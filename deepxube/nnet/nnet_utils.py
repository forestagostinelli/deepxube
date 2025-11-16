from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass

from deepxube.utils.data_utils import SharedNDArray, np_to_shnd
import numpy as np
from numpy.typing import NDArray
import os
import torch
from torch import nn
from collections import OrderedDict
import re
from torch import Tensor
from torch.multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess


# training
def to_pytorch_input(states_nnet: List[NDArray[Any]], device: torch.device) -> List[Tensor]:
    states_nnet_tensors = []
    for tensor_np in states_nnet:
        tensor = torch.tensor(tensor_np, device=device)
        states_nnet_tensors.append(tensor)

    return states_nnet_tensors


# pytorch device
def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cpu")
    devices: List[int] = []
    on_gpu: bool = False
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 0)
        devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
        on_gpu = True
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(8)

    return device, devices, on_gpu


# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file, weights_only=True)
    else:
        state_dict = torch.load(model_file, map_location=device, weights_only=False)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub(r'^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


def get_available_gpu_nums() -> List[int]:
    gpu_nums: List[int] = []
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

    return gpu_nums


def nnet_batched(nnet: nn.Module, inputs: List[NDArray[Any]], batch_size: Optional[int],
                 device: torch.device) -> NDArray[np.float64]:
    outputs_l: List[NDArray[np.float64]] = []

    num_states: int = inputs[0].shape[0]

    batch_size_inst: int = num_states
    if batch_size is not None:
        batch_size_inst = batch_size

    start_idx: int = 0
    while start_idx < num_states:
        # get batch
        end_idx: int = min(start_idx + batch_size_inst, num_states)
        inputs_batch = [x[start_idx:end_idx] for x in inputs]

        # get nnet output
        inputs_batch_t = to_pytorch_input(inputs_batch, device)

        outputs_batch: NDArray[np.float64] = nnet(inputs_batch_t).cpu().data.numpy()
        outputs_l.append(outputs_batch)

        start_idx = end_idx

    outputs: NDArray[np.float64] = np.concatenate(outputs_l, axis=0).astype(np.float64)
    assert (outputs.shape[0] == num_states)

    return outputs


@dataclass
class NNetParInfo:
    nnet_i_q: Queue
    nnet_o_q: Queue
    proc_id: int


# parallel neural networks
def nnet_fn_runner(nnet_i_q: Queue, nnet_o_qs: List[Queue], model_file: str, device: torch.device, on_gpu: bool,
                   gpu_num: int, get_nnet: Callable[[], nn.Module], batch_size: Optional[int]) -> None:
    if (gpu_num is not None) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    torch.set_num_threads(1)
    nnet: nn.Module = get_nnet()
    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    # if on_gpu:
    #    nnet = nn.DataParallel(nnet)

    while True:
        # get from input q
        inputs_nnet_shm: Optional[List[SharedNDArray]]
        proc_id, inputs_nnet_shm = nnet_i_q.get()
        if proc_id is None:
            break

        # get outputs
        inputs_nnet: List[NDArray] = []
        for inputs_idx in range(len(inputs_nnet_shm)):
            inputs_nnet.append(inputs_nnet_shm[inputs_idx].array)

        outputs: NDArray[np.float64] = nnet_batched(nnet, inputs_nnet, batch_size, device)

        # send outputs
        outputs_shm: SharedNDArray = np_to_shnd(outputs)
        nnet_o_qs[proc_id].put(outputs_shm)

        for arr_shm in inputs_nnet_shm + [outputs_shm]:
            arr_shm.close()


def start_nnet_fn_runners(get_nnet: Callable[[], nn.Module], num_procs: int, model_file: str, device: torch.device,
                          on_gpu: bool,
                          batch_size: Optional[int] = None) -> Tuple[List[NNetParInfo], List[BaseProcess]]:
    ctx = get_context("spawn")

    nnet_i_q: Queue = ctx.Queue()
    nnet_o_qs: List[Queue] = []
    nnet_par_infos: List[NNetParInfo] = []
    for proc_id in range(num_procs):
        nnet_o_q: Queue = ctx.Queue(1)
        nnet_o_qs.append(nnet_o_q)
        nnet_par_infos.append(NNetParInfo(nnet_i_q, nnet_o_q, proc_id))

    # initialize heuristic procs
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
    else:
        gpu_nums = [-1]

    nnet_procs: List[BaseProcess] = []
    for gpu_num in gpu_nums:
        nnet_fn_procs = ctx.Process(target=nnet_fn_runner,
                                    args=(nnet_i_q, nnet_o_qs, model_file, device, on_gpu, gpu_num, get_nnet,
                                          batch_size))
        nnet_fn_procs.daemon = True
        nnet_fn_procs.start()
        nnet_procs.append(nnet_fn_procs)

    return nnet_par_infos, nnet_procs


def stop_nnet_runners(nnet_fn_procs: List[BaseProcess], nnet_par_infos: List[NNetParInfo]) -> None:
    for _ in nnet_fn_procs:
        nnet_par_infos[0].nnet_i_q.put((None, None))

    for heur_fn_proc in nnet_fn_procs:
        heur_fn_proc.join()


NNetCallable = Callable[..., Any]
NNetFn = TypeVar('NNetFn', bound=NNetCallable)


class NNetPar(ABC, Generic[NNetFn]):
    @abstractmethod
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> NNetFn:
        pass

    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> NNetFn:
        pass

    @abstractmethod
    def get_nnet(self) -> nn.Module:
        pass


def get_nnet_par_out(inputs_nnet: List[NDArray], nnet_par_info: NNetParInfo) -> NDArray:
    inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i)
                                            for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

    nnet_par_info.nnet_i_q.put((nnet_par_info.proc_id, inputs_nnet_shm))

    out_shm: SharedNDArray = nnet_par_info.nnet_o_q.get()
    out: NDArray = out_shm.array.copy()

    for arr_shm in inputs_nnet_shm + [out_shm]:
        arr_shm.close()
        arr_shm.unlink()

    return out
