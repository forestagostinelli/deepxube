from typing import List, Tuple, Optional, Callable, Any, Union, cast
from deepxube.environments.environment_abstract import Environment, State, Goal, HeurFnNNet
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd
import numpy as np
from numpy.typing import NDArray
import os
import torch
from torch import nn
from collections import OrderedDict
import re
from torch import Tensor
from torch.nn import DataParallel
from torch.multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess

HeurFN_T = Callable[[Union[List[State], List[NDArray[Any]]], Optional[List[Goal]]], NDArray[np.float64]]


# training
def to_pytorch_input(states_nnet: List[NDArray[Any]], device) -> List[Tensor]:
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
    else:
        torch.set_num_threads(8)

    return device, devices, on_gpu


# loading nnet
def load_nnet(model_file: str, nnet: HeurFnNNet, device: Optional[torch.device] = None) -> HeurFnNNet:
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


# heuristic
def get_heuristic_fn(nnet: nn.Module, device: torch.device, env: Environment, clip_zero: bool = False,
                     batch_size: Optional[int] = None, is_v: bool = False) -> HeurFN_T:
    nnet.eval()

    def heuristic_fn(states: Union[List[State], List[NDArray[Any]]],
                     goals: Optional[List[Goal]]) -> NDArray[np.float64]:
        cost_to_go_l: List[NDArray[np.float64]] = []

        num_states: int
        is_nnet_format: bool
        if goals is not None:
            num_states = len(states)
            is_nnet_format = False
        else:
            num_states = cast(List[NDArray[Any]], states)[0].shape[0]
            is_nnet_format = True

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # convert to nnet input
            if not is_nnet_format:
                states_batch: List[State] = cast(List[State], states)[start_idx:end_idx]
                goals_batch: List[Goal] = cast(List[Goal], goals)[start_idx:end_idx]

                states_goals_nnet_batch = env.states_goals_to_nnet_input(states_batch, goals_batch)
            else:
                states_goals_nnet_batch = [x[start_idx:end_idx] for x in cast(List[NDArray[Any]], states)]

            # get nnet output
            states_goals_nnet_batch_tensors = to_pytorch_input(states_goals_nnet_batch, device)

            cost_to_go_batch: NDArray[np.float64] = nnet(states_goals_nnet_batch_tensors).cpu().data.numpy()
            if is_v:
                cost_to_go_batch = cost_to_go_batch[:, 0]
            cost_to_go_l.append(cost_to_go_batch)

            start_idx = end_idx

        cost_to_go = np.concatenate(cost_to_go_l, axis=0).astype(np.float64)
        assert (cost_to_go.shape[0] == num_states)

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_available_gpu_nums() -> List[int]:
    gpu_nums: List[int] = []
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

    return gpu_nums


def load_heuristic_fn(model_file: str, device: torch.device, on_gpu: bool, nnet: HeurFnNNet,
                      env: Environment, clip_zero: bool = False, gpu_num: Optional[int] = None,
                      batch_size: Optional[int] = None) -> HeurFN_T:
    if (gpu_num is not None) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    is_v: bool = nnet.nnet_type.upper() == "V"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)

    nnet_heur: Union[HeurFnNNet, DataParallel[HeurFnNNet]]
    if on_gpu:
        nnet_heur = nn.DataParallel(nnet)
    else:
        nnet_heur = nnet

    heuristic_fn = get_heuristic_fn(nnet_heur, device, env, clip_zero=clip_zero, batch_size=batch_size, is_v=is_v)

    return heuristic_fn


# parallel training
def heuristic_fn_runner(heuristic_fn_input_queue: Queue, heuristic_fn_output_queues: List[Queue],
                        model_file: str, device, on_gpu: bool, gpu_num: int,
                        env: Environment, heur_type: str, all_zeros: bool, clip_zero: bool, batch_size: Optional[int]):
    nnet: HeurFnNNet
    if heur_type.upper() == "V":
        nnet = env.get_v_nnet()
    elif heur_type.upper() == "Q":
        nnet = env.get_q_nnet()
    else:
        raise ValueError(f"Unknown heuristic type {heur_type}")
    heuristic_fn: Optional[HeurFN_T] = None
    if not all_zeros:
        heuristic_fn = load_heuristic_fn(model_file, device, on_gpu, nnet, env, gpu_num=gpu_num,
                                         clip_zero=clip_zero, batch_size=batch_size)

    while True:
        inputs_nnet_shm: Optional[List[SharedNDArray]]
        proc_id, inputs_nnet_shm = heuristic_fn_input_queue.get()
        if proc_id is None:
            break

        inputs_nnet: List[NDArray] = []
        for inputs_idx in range(len(inputs_nnet_shm)):
            inputs_nnet.append(inputs_nnet_shm[inputs_idx].array)
        if all_zeros:
            heurs = np.zeros(inputs_nnet[0].shape[0], dtype=float)
        else:
            heurs = cast(HeurFN_T, heuristic_fn)(inputs_nnet, None)

        shm_name: str = f"{proc_id}_out"
        heurs_shm: SharedNDArray = np_to_shnd(heurs, shm_name)
        heuristic_fn_output_queues[proc_id].put(heurs_shm)

        for arr_shm in inputs_nnet_shm + [heurs_shm]:
            arr_shm.close()

    return heuristic_fn


class HeurFnQ:
    def __init__(self, heur_fn_i_q, heur_fn_o_q, proc_id: int):
        self.heur_fn_i_q = heur_fn_i_q
        self.heur_fn_o_q = heur_fn_o_q
        self.proc_id: int = proc_id

    def get_heuristic_fn(self, env: Environment) -> HeurFN_T:
        def heuristic_fn(states: List[State], goals: List[Goal]) -> NDArray:
            inputs_nnet: List[NDArray] = env.states_goals_to_nnet_input(states, goals)
            inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i, f"{input_idx}_{self.proc_id}")
                                                    for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

            self.heur_fn_i_q.put((self.proc_id, inputs_nnet_shm))

            heurs_shm: SharedNDArray = self.heur_fn_o_q.get()
            heurs: NDArray = heurs_shm.array.copy()

            for arr_shm in inputs_nnet_shm + [heurs_shm]:
                arr_shm.close()
                arr_shm.unlink()

            return heurs

        return heuristic_fn


def start_heur_fn_runners(num_procs: int, model_file: str, device, on_gpu: bool, env: Environment, heur_type: str,
                          all_zeros: bool = False, clip_zero: bool = False,
                          batch_size: Optional[int] = None) -> Tuple[List[HeurFnQ], List[BaseProcess]]:
    ctx = get_context("spawn")

    heur_fn_i_q: Queue = ctx.Queue()
    heur_fn_o_qs: List[Queue] = []
    heur_fn_qs: List[HeurFnQ] = []
    for proc_id in range(num_procs):
        heur_fn_o_q: Queue = ctx.Queue(1)
        heur_fn_o_qs.append(heur_fn_o_q)
        heur_fn_qs.append(HeurFnQ(heur_fn_i_q, heur_fn_o_q, proc_id))

    # initialize heuristic procs
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
    else:
        gpu_nums = [-1]

    heur_procs: List[BaseProcess] = []
    for gpu_num in gpu_nums:
        heur_fn_proc = ctx.Process(target=heuristic_fn_runner,
                                   args=(heur_fn_i_q, heur_fn_o_qs, model_file, device, on_gpu, gpu_num, env, heur_type,
                                         all_zeros, clip_zero, batch_size))
        heur_fn_proc.daemon = True
        heur_fn_proc.start()
        heur_procs.append(heur_fn_proc)

    return heur_fn_qs, heur_procs


def stop_heuristic_fn_runners(heur_fn_procs, heur_fn_qs: List[HeurFnQ]):
    for _ in heur_fn_procs:
        heur_fn_qs[0].heur_fn_i_q.put((None, None))

    for heur_fn_proc in heur_fn_procs:
        heur_fn_proc.join()
