from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Callable, TypeVar, Generic, cast
from dataclasses import dataclass

from deepxube.utils.data_utils import SharedNDArray, np_to_shnd, combine_l_l
import numpy as np
from numpy.typing import NDArray
import os
import torch
from torch import nn, Tensor
from collections import OrderedDict
import re
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
        state_dict = torch.load(model_file, map_location=torch.device('cpu'), weights_only=True)
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
                 device: torch.device) -> List[NDArray[np.float64]]:
    outputs_l_l: List[List[NDArray[np.float64]]] = []

    num_states: int = inputs[0].shape[0]

    batch_size_inst: int = num_states
    if batch_size is not None:
        batch_size_inst = batch_size

    start_idx: int = 0
    num_outputs: Optional[int] = None
    while start_idx < num_states:
        # get batch
        end_idx: int = min(start_idx + batch_size_inst, num_states)
        inputs_batch = [x[start_idx:end_idx] for x in inputs]

        # get nnet output
        inputs_batch_t = to_pytorch_input(inputs_batch, device)

        outputs_batch_t_l: List[Tensor] = nnet(inputs_batch_t)
        outputs_batch_l: List[NDArray[np.float64]] = [outputs_batch_t.cpu().data.numpy()
                                                      for outputs_batch_t in outputs_batch_t_l]
        if num_outputs is None:
            num_outputs = len(outputs_batch_l)
        else:
            assert len(outputs_batch_l) == num_outputs, f"{len(outputs_batch_l)} != {num_outputs}"

        for out_idx in range(len(outputs_batch_l)):
            outputs_batch_l[out_idx] = outputs_batch_l[out_idx].astype(np.float64)
        outputs_l_l.append(outputs_batch_l)

        start_idx = end_idx

    outputs_l: List[NDArray[np.float64]] = combine_l_l(outputs_l_l, "concat")
    for out_idx in range(len(outputs_l)):
        assert (outputs_l[out_idx].shape[0] == num_states)

    return outputs_l


@dataclass
class NNetParInfo:
    nnet_i_q: Queue
    nnet_o_q: Queue
    proc_id: int


def nnet_in_out_shared_q(nnet: nn.Module, inputs_nnet_shm: List[SharedNDArray], batch_size: Optional[int],
                         device: torch.device, nnet_o_q: Queue) -> None:
    # get outputs
    inputs_nnet: List[NDArray] = []
    for inputs_idx in range(len(inputs_nnet_shm)):
        inputs_nnet.append(inputs_nnet_shm[inputs_idx].array)

    outputs_l: List[NDArray[np.float64]] = nnet_batched(nnet, inputs_nnet, batch_size, device)

    # send outputs
    outputs_l_shm: List[SharedNDArray] = [np_to_shnd(outputs) for outputs in outputs_l]
    nnet_o_q.put(outputs_l_shm)

    for arr_shm in inputs_nnet_shm + outputs_l_shm:
        arr_shm.close()


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

        # nnet in/out
        assert inputs_nnet_shm is not None
        nnet_in_out_shared_q(nnet, inputs_nnet_shm, batch_size, device, nnet_o_qs[proc_id])


def get_nnet_par_infos(num_procs: int) -> List[NNetParInfo]:
    ctx = get_context("spawn")

    nnet_i_q: Queue = ctx.Queue()
    nnet_o_qs: List[Queue] = []
    nnet_par_infos: List[NNetParInfo] = []
    for proc_id in range(num_procs):
        nnet_o_q: Queue = ctx.Queue(1)
        nnet_o_qs.append(nnet_o_q)
        nnet_par_infos.append(NNetParInfo(nnet_i_q, nnet_o_q, proc_id))

    return nnet_par_infos


def start_nnet_fn_runners(get_nnet: Callable[[], nn.Module], nnet_par_infos: List[NNetParInfo], model_file: str,
                          device: torch.device, on_gpu: bool,
                          batch_size: Optional[int] = None) -> List[BaseProcess]:
    ctx = get_context("spawn")

    nnet_i_q: Queue = nnet_par_infos[0].nnet_i_q
    nnet_o_qs: List[Queue] = [nnet_par_info.nnet_o_q for nnet_par_info in nnet_par_infos]

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

    return nnet_procs


def stop_nnet_fn_runners(nnet_fn_procs: List[BaseProcess], nnet_par_infos: List[NNetParInfo]) -> None:
    for _ in nnet_fn_procs:
        nnet_par_infos[0].nnet_i_q.put((None, None))

    for heur_fn_proc in nnet_fn_procs:
        heur_fn_proc.join()


def get_nnet_par_out(inputs_nnet: List[NDArray], nnet_par_info: NNetParInfo) -> List[NDArray]:
    inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i)
                                            for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

    nnet_par_info.nnet_i_q.put((nnet_par_info.proc_id, inputs_nnet_shm))

    out_shm_l: List[SharedNDArray] = nnet_par_info.nnet_o_q.get()
    out_l: List[NDArray] = [out_shm.array.copy() for out_shm in out_shm_l]

    for arr_shm in inputs_nnet_shm + out_shm_l:
        arr_shm.close()
        arr_shm.unlink()

    return out_l


NNetCallable = Callable[..., Any]
NNF_T = TypeVar('NNF_T', bound=NNetCallable)
PROCESSED_T = TypeVar("PROCESSED_T")


@dataclass(frozen=True)
class ProcessedInput(Generic[PROCESSED_T]):
    inputs_nnet: List[NDArray]
    processed: PROCESSED_T


class NNetPar(ABC, Generic[NNF_T, PROCESSED_T]):
    def __init__(self, **kwargs: Any):
        self.nnet_file: Optional[str] = None
        self.nnet_par_fn: Optional[NNF_T] = None
        self.nnet_par_info: Optional[NNetParInfo] = None
        self.nnet_par_info_l: Optional[List[NNetParInfo]] = None
        self.nnet_runner_proc_l: Optional[List[BaseProcess]] = None

    """ A neural network that can be called from other processes """
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device) -> NNF_T:
        """

        :param nnet: Neural network module, assumed to take a List of NDArrays as input
        :param batch_size: Maximum number of inputs to pass to neural network at a time. Loops until all inputs are given. If None, all inputs are given
        at once, no matter how large.
        :param device: Device that the neural network is on
        :return: Neural network function
        """
        nnet.eval()

        def nnet_fn(*args: Any) -> Any:
            processed_input: ProcessedInput[PROCESSED_T] = self.process_inputs(*args)
            outs: List[NDArray[np.float64]] = nnet_batched(nnet, processed_input.inputs_nnet, batch_size, device)

            return self.process_outputs(outs, processed_input.processed)

        return cast(NNF_T, nnet_fn)

    def get_nnet_par_fn_w_info(self, nnet_par_info: NNetParInfo) -> NNF_T:
        """

        :param nnet_par_info: Information of neural network which may exist on a different process
        :return: Neural network function
        """
        def nnet_fn(*args: Any) -> Any:
            processed_input: ProcessedInput[PROCESSED_T] = self.process_inputs(*args)
            outs: List[NDArray[np.float64]] = get_nnet_par_out(processed_input.inputs_nnet, nnet_par_info)

            return self.process_outputs(outs, processed_input.processed)

        return cast(NNF_T, nnet_fn)

    @abstractmethod
    def get_nnet(self) -> nn.Module:
        """

        :return: Neural network module, assumed to take a List of NDArrays as input
        """
        pass

    @abstractmethod
    def process_inputs(self, *args: Any) -> ProcessedInput[PROCESSED_T]:
        pass

    @abstractmethod
    def process_outputs(self, outs: List[NDArray], processed: PROCESSED_T) -> Any:
        pass

    @abstractmethod
    def get_default_fn(self) -> NNF_T:
        pass

    def set_nnet_file(self, nnet_file: str) -> None:
        self.nnet_file = nnet_file

    def get_nnet_par_infos(self, proc_idx: int) -> NNetParInfo:
        assert self.nnet_par_info_l is not None
        return self.nnet_par_info_l[proc_idx]

    def get_nnet_par_fn(self) -> NNF_T:
        assert self.nnet_par_fn is not None
        return self.nnet_par_fn

    def set_nnet_par_info_l(self, num_procs: int) -> None:
        assert self.nnet_runner_proc_l is None
        self.nnet_par_info_l = get_nnet_par_infos(num_procs)

    def set_nnet_par_info(self, nnet_par_info: NNetParInfo) -> None:
        assert self.nnet_par_info is None
        self.nnet_par_info = nnet_par_info

    def start_nnet_runners(self, device: torch.device, on_gpu: bool, nnet_batch_size: Optional[int]) -> None:
        assert self.nnet_par_info_l is not None
        assert self.nnet_runner_proc_l is None
        assert self.nnet_file is not None
        self.nnet_runner_proc_l = start_nnet_fn_runners(self.get_nnet, self.nnet_par_info_l, self.nnet_file, device, on_gpu, batch_size=nnet_batch_size)

    def stop_nnet_runners(self) -> None:
        assert self.nnet_runner_proc_l is not None
        assert self.nnet_par_info_l is not None
        stop_nnet_fn_runners(self.nnet_runner_proc_l, self.nnet_par_info_l)

        self.nnet_runner_proc_l = None

    def init_nnet_par_fn(self) -> None:
        assert self.nnet_par_fn is None
        assert self.nnet_par_info is not None
        self.nnet_par_fn = self.get_nnet_par_fn_w_info(self.nnet_par_info)

    def clear_nnet_fn(self) -> None:
        assert self.nnet_par_fn is not None
        self.nnet_par_fn = None

    def __repr__(self) -> str:
        nnet: nn.Module = self.get_nnet()
        num_trainable = sum(p.numel() for p in nnet.parameters() if p.requires_grad)
        return f"{type(self).__name__}()\n{nnet}\nNumber of trainable parameters: {format(num_trainable, ',')}"
