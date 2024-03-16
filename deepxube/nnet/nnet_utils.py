from typing import List, Tuple, Optional, Callable
from deepxube.environments.environment_abstract import Environment, State, HeurFnNNet
import numpy as np
import os
import torch
from torch import nn
from collections import OrderedDict
import re
from torch import Tensor
from torch.multiprocessing import Queue, get_context


# training
def to_pytorch_input(states_nnet: List[np.ndarray], device) -> List[Tensor]:
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
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


# heuristic
def get_heuristic_fn(nnet: nn.Module, device: torch.device, env: Environment, clip_zero: bool = False,
                     batch_size: Optional[int] = None, is_v: bool = False):
    nnet.eval()

    def heuristic_fn(states: List, goals: List, is_nnet_format: bool = False) -> np.ndarray:
        cost_to_go_l: List[np.ndarray] = []

        if not is_nnet_format:
            num_states: int = len(states)
        else:
            num_states: int = states[0].shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # convert to nnet input
            if not is_nnet_format:
                states_batch: List = states[start_idx:end_idx]
                goals_batch: List = goals[start_idx:end_idx]

                states_nnet_batch = env.states_to_nnet_input(states_batch)
                goals_nnet_batch = env.goals_to_nnet_input(goals_batch)
            else:
                states_nnet_batch = [x[start_idx:end_idx] for x in states]
                goals_nnet_batch = [x[start_idx:end_idx] for x in goals]

            # get nnet output
            states_nnet_batch_tensors = to_pytorch_input(states_nnet_batch, device)
            goal_nnet_batch_tensors = to_pytorch_input(goals_nnet_batch, device)

            cost_to_go_batch: np.ndarray = nnet(states_nnet_batch_tensors, goal_nnet_batch_tensors).cpu().data.numpy()
            if is_v:
                cost_to_go_batch = cost_to_go_batch[:, 0]
            cost_to_go_l.append(cost_to_go_batch)

            start_idx: int = end_idx

        cost_to_go = np.concatenate(cost_to_go_l, axis=0)
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
                      batch_size: Optional[int] = None):
    if (gpu_num is not None) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    is_v: bool = nnet.nnet_type.upper() == "V"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(nnet, device, env, clip_zero=clip_zero, batch_size=batch_size, is_v=is_v)

    return heuristic_fn


# parallel training
def heuristic_fn_runner(heuristic_fn_input_queue: Queue, heuristic_fn_output_queues, model_file: str,
                        device, on_gpu: bool, gpu_num: int, nnet: HeurFnNNet, env: Environment, all_zeros: bool,
                        clip_zero: bool, batch_size: Optional[int]):
    heuristic_fn = None
    if not all_zeros:
        heuristic_fn = load_heuristic_fn(model_file, device, on_gpu, nnet, env, gpu_num=gpu_num,
                                         clip_zero=clip_zero, batch_size=batch_size)

    while True:
        proc_id, states_nnet, goals_nnet = heuristic_fn_input_queue.get()
        if proc_id is None:
            break

        if all_zeros:
            heuristics = np.zeros(states_nnet[0].shape[0], dtype=float)
        else:
            heuristics = heuristic_fn(states_nnet, goals_nnet, is_nnet_format=True)

        heuristic_fn_output_queues[proc_id].put(heuristics)

    return heuristic_fn


class HeurFnQ:
    def __init__(self, heur_fn_i_q, heur_fn_o_q, proc_id: int):
        self.heur_fn_i_q = heur_fn_i_q
        self.heur_fn_o_q = heur_fn_o_q
        self.proc_id: int = proc_id

    def get_heuristic_fn(self, env: Environment) -> Callable:
        def heuristic_fn(states, goals):
            if isinstance(states[0], State):
                states_nnet = env.states_to_nnet_input(states)
                goals_nnet = env.goals_to_nnet_input(goals)
            else:
                states_nnet = states
                goals_nnet = goals

            self.heur_fn_i_q.put((self.proc_id, states_nnet, goals_nnet))
            heuristics = self.heur_fn_o_q.get()

            return heuristics

        return heuristic_fn


def start_heur_fn_runners(num_procs: int, model_file: str, device, on_gpu: bool, nnet: HeurFnNNet, env: Environment,
                          all_zeros: bool = False, clip_zero: bool = False, batch_size: Optional[int] = None):
    ctx = get_context("spawn")

    heur_fn_i_q: ctx.Queue = ctx.Queue()
    heur_fn_o_qs: List[ctx.Queue] = []
    heur_fn_qs: List[HeurFnQ] = []
    for proc_id in range(num_procs):
        heur_fn_o_q: ctx.Queue = ctx.Queue(1)
        heur_fn_o_qs.append(heur_fn_o_q)
        heur_fn_qs.append(HeurFnQ(heur_fn_i_q, heur_fn_o_q, proc_id))

    # initialize heuristic procs
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
    else:
        gpu_nums = [-1]

    heur_procs: List[ctx.Process] = []
    for gpu_num in gpu_nums:
        heur_fn_proc = ctx.Process(target=heuristic_fn_runner,
                                   args=(heur_fn_i_q, heur_fn_o_qs, model_file, device, on_gpu, gpu_num, nnet,
                                         env, all_zeros, clip_zero, batch_size))
        heur_fn_proc.daemon = True
        heur_fn_proc.start()
        heur_procs.append(heur_fn_proc)

    return heur_fn_qs, heur_procs


def stop_heuristic_fn_runners(heur_fn_procs, heur_fn_qs: List[HeurFnQ]):
    for _ in heur_fn_procs:
        heur_fn_qs[0].heur_fn_i_q.put((None, None, None))

    for heur_fn_proc in heur_fn_procs:
        heur_fn_proc.join()
