from typing import List, Dict, Tuple, Any, Optional, Generic, TypeVar
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess

import numpy as np
import torch
from numpy.typing import NDArray

from deepxube.nnet.nnet_utils import NNetParInfo
from deepxube.base.environment import Environment, Action
from deepxube.base.heuristic import HeurNNet, HeurFn, HeurFnV, HeurFnQ
from deepxube.base.heuristic import HeurNNetV, HeurNNetQ
from deepxube.nnet import nnet_utils
from deepxube.base.pathfinding import PathFind, PathFindV, PathFindQ, Instance, InstArgs
from deepxube.pathfinding.pathfinding_utils import PathFindPerf
from deepxube.training.train_utils import ReplayBuffer
from deepxube.utils.data_utils import SharedNDArray
from deepxube.utils.misc_utils import split_evenly_w_max
from deepxube.utils.timing_utils import Times


@dataclass
class UpHeurArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data
    balance.

    :param up_itrs: How many iterations to wait for updating target network
    :param up_gen_itrs: How many iterations worth of data to generate per udpate
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_search_itrs: Maximum number of pathfinding iterationos to take from generated problem instances
    :param up_batch_size: Maximum number of searches to do at a time. Helps manage memory.
    Decrease if memory is running out during updater.
    :param up_nnet_batch_size: Batch size of each nnet used for each process updater. Make smaller if running out
    of memory.
    Increasing this number could make the heuristic function more robust to depression regions.
    """
    up_itrs: int
    up_gen_itrs: int
    up_procs: int
    up_search_itrs: int
    up_batch_size: int
    up_nnet_batch_size: int


def get_data_from_procs(num_gen: int, from_q: Queue, to_q: Queue, procs: List[BaseProcess],
                        inputs_nnet_shm: List[SharedNDArray], ctgs_shm: SharedNDArray, rb: ReplayBuffer,
                        start_time_gen: float) -> Dict[int, PathFindPerf]:
    # getting data from processes
    times_up: Times = Times()
    display_counts: NDArray[np.int_] = np.linspace(0, num_gen, 10, dtype=int)
    num_gen_curr: int = 0
    while num_gen_curr < num_gen:
        start_idx, end_idx = from_q.get()
        start_time = time.time()
        inputs_nnet_get: List[NDArray] = []
        for inputs_idx in range(len(inputs_nnet_shm)):
            inputs_nnet_get.append(inputs_nnet_shm[inputs_idx][start_idx:end_idx].copy())
        ctgs_shm_get: NDArray = ctgs_shm[start_idx:end_idx].copy()
        rb.add(inputs_nnet_get + [ctgs_shm_get])
        num_gen_curr += (end_idx - start_idx)
        times_up.record_time("rb", time.time() - start_time)
        if num_gen_curr >= min(display_counts):
            print(f"{num_gen_curr}/{num_gen} instances (%.2f%%) "
                  f"(Tot time: %.2f)" % (100 * num_gen_curr / num_gen, time.time() - start_time_gen))
            display_counts = display_counts[num_gen_curr < display_counts]

    # sending stop signal
    for _ in procs:
        to_q.put((None, None))

    # get summary from processes
    step_to_pathperf: Dict[int, PathFindPerf] = dict()
    for _ in procs:
        times_up_i, step_to_pathperf_i = from_q.get()
        times_up.add_times(times_up_i)
        for step_num_perf, pathperf in step_to_pathperf_i.items():
            if step_num_perf not in step_to_pathperf.keys():
                step_to_pathperf[step_num_perf] = PathFindPerf()
            step_to_pathperf[step_num_perf] = step_to_pathperf[step_num_perf].comb_perf(pathperf)

    # cost-to-go summary
    mean_ctg = ctgs_shm.array.mean()
    min_ctg = ctgs_shm.array.min()
    max_ctg = ctgs_shm.array.max()
    print(f"Generated {format(num_gen_curr, ',')} training instances, "
          f"Replay buffer size: {format(rb.size(), ',')}")
    print("Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))
    print(f"Times - {times_up.get_time_str()}")

    return step_to_pathperf


def _put_to_shm(inputs_nnet_shm: List[SharedNDArray], ctgs_shm: SharedNDArray, inputs_nnet: List[NDArray],
                ctgs_backup: List[float], start_idx: int, times: Times) -> int:
    start_time = time.time()
    end_idx = start_idx + len(ctgs_backup)
    for input_idx in range(len(inputs_nnet)):
        inputs_nnet_shm[input_idx][start_idx:end_idx] = inputs_nnet[input_idx].copy()
    ctgs_shm[start_idx:end_idx] = np.array(ctgs_backup).copy()
    start_idx = end_idx
    times.record_time("put", time.time() - start_time)
    return start_idx


def _update_perf(insts_rem: List[Instance], step_to_pathperf: Dict[int, PathFindPerf]):
    for inst_rem in insts_rem:
        step_num_inst: int = int(inst_rem.inst_info[0])
        if step_num_inst not in step_to_pathperf.keys():
            step_to_pathperf[step_num_inst] = PathFindPerf()
        step_to_pathperf[step_num_inst].update_perf(inst_rem)


H = TypeVar('H', bound=HeurFn)
P = TypeVar('P', bound=PathFind)


class UpdaterHeur(ABC, Generic[H, P]):
    def __init__(self, env: Environment, heur_nnet: HeurNNet, up_args: UpHeurArgs):
        self.env: Environment = env
        self.heur_nnet: HeurNNet = heur_nnet
        self.up_args: UpHeurArgs = up_args

    @abstractmethod
    def get_pathfind(self) -> P:
        pass

    @abstractmethod
    def step_get_in_out_np(self, pathfind: P, heur_fn: H,
                           times: Times) -> Tuple[List[NDArray], List[float]]:
        pass

    @abstractmethod
    def get_input_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        pass

    def update_runner(self, gen_step_max: int, nnet_par_info: NNetParInfo, to_q: Queue, from_q: Queue,
                      step_probs: NDArray, inputs_nnet_shm: List[SharedNDArray], ctgs_shm: SharedNDArray):
        times: Times = Times()

        heur_fn: H = self.heur_nnet.get_nnet_par_fn(nnet_par_info)
        step_to_pathperf: Dict[int, PathFindPerf] = dict()
        while True:
            batch_size, start_idx = to_q.get()
            if batch_size is None:
                break

            pathfind: P = self.get_pathfind()

            insts_rem: List[Instance] = []
            start_idx_batch: int = start_idx
            for _ in range(self.up_args.up_search_itrs):
                # add instances
                self._add_instances(pathfind, insts_rem, gen_step_max, batch_size, step_probs, heur_fn, times)

                # step and to_np
                inputs_nnet, ctgs_backup = self.step_get_in_out_np(pathfind, heur_fn, times)
                assert len(ctgs_backup) == batch_size

                # put
                start_idx = _put_to_shm(inputs_nnet_shm, ctgs_shm, inputs_nnet, ctgs_backup, start_idx, times)

                # remove instances
                insts_rem = pathfind.remove_finished_instances(self.up_args.up_search_itrs)

                # pathfinding performance
                _update_perf(insts_rem, step_to_pathperf)

            from_q.put((start_idx_batch, start_idx))
            times.add_times(pathfind.times, path=["pathfinding"])

        from_q.put((times, step_to_pathperf))
        for arr_shm in inputs_nnet_shm + [ctgs_shm]:
            arr_shm.close()

    def _add_instances(self, pathfind: P, insts_rem: List[Instance], gen_step_max: int, batch_size: int,
                       step_probs: NDArray, heur_fn: H, times: Times):
        if (len(pathfind.instances) == 0) or (len(insts_rem) > 0):
            times_states: Times = Times()
            # get steps generate
            start_time = time.time()
            steps_gen: List[int]
            if len(pathfind.instances) == 0:
                steps_gen = list(np.random.choice(gen_step_max + 1, size=batch_size, p=step_probs))
            else:
                steps_gen = [int(inst.inst_info[0]) for inst in insts_rem]
            times_states.record_time("steps_gen", time.time() - start_time)

            # generate states
            states_gen, goals_gen = self.env.get_start_goal_pairs(steps_gen, times=times_states)
            times.add_times(times_states, ["get_states"])

            # get instance information and kwargs
            start_time = time.time()
            inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
            times.record_time("inst_info", time.time() - start_time)

            # add instances
            pathfind.add_instances(states_gen, goals_gen, heur_fn, self._get_inst_args(len(states_gen)),
                                   inst_infos=inst_infos, compute_init_heur=True)

    @abstractmethod
    def _get_inst_args(self, num: int) -> List[InstArgs]:
        pass

    def get_update_data(self, heur_file: str, all_zeros: bool, step_max: int, step_probs: NDArray, num_gen: int,
                        rb: ReplayBuffer, device: torch.device, on_gpu: bool) -> Dict[int, PathFindPerf]:
        start_time_gen = time.time()
        # put work information on to_q
        ctx = get_context("spawn")
        to_q: Queue = self._send_work_to_q(num_gen, ctx)

        # parallel heuristic functions
        nnet_par_infos, nnet_procs = nnet_utils.start_nnet_fn_runners(self.heur_nnet.get_nnet, self.up_args.up_procs,
                                                                      heur_file, device, on_gpu, all_zeros=all_zeros,
                                                                      clip_zero=True,
                                                                      batch_size=self.up_args.up_nnet_batch_size)

        # shared memory
        inputs_nnet_shm, ctgs_shm = self._init_shared_mem(num_gen)

        # starting processes
        from_q: Queue = ctx.Queue()
        procs: List[BaseProcess] = []
        for proc_id, nnet_par_info in enumerate(nnet_par_infos):
            proc: BaseProcess = ctx.Process(target=self.update_runner, args=(step_max, nnet_par_info, to_q, from_q,
                                                                             step_probs, inputs_nnet_shm, ctgs_shm))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        # getting data from procs
        step_to_pathperf: Dict[int, PathFindPerf] = get_data_from_procs(num_gen, from_q, to_q, procs, inputs_nnet_shm,
                                                                        ctgs_shm, rb, start_time_gen)

        # clean up clean up everybody do your share
        nnet_utils.stop_nnet_runners(nnet_procs, nnet_par_infos)
        for proc in procs:
            proc.join()
        for arr_shm in inputs_nnet_shm + [ctgs_shm]:
            arr_shm.close()
            arr_shm.unlink()

        return step_to_pathperf

    def _init_shared_mem(self, num_gen: int) -> Tuple[List[SharedNDArray], SharedNDArray]:
        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = self.get_input_shapes_dtypes()
        inputs_nnet_shm: List[SharedNDArray] = []
        for shape, dtype in shapes_dtypes:
            inputs_nnet_shm.append(SharedNDArray((num_gen,) + shape, dtype, None, True))
        ctgs_shm = SharedNDArray((num_gen,), np.float64, None, True)

        return inputs_nnet_shm, ctgs_shm

    def _send_work_to_q(self, num_gen: int, ctx) -> Queue:
        num_searches: int = num_gen // self.up_args.up_search_itrs
        print(f"Generating {format(num_gen, ',')} training instances with {format(num_searches, ',')} searches")

        assert num_gen % self.up_args.up_search_itrs == 0, (f"Number of instances to generate per for this updater "
                                                            f"{num_gen} is not divisible by the max number of "
                                                            f"pathfinding iterations to take during the "
                                                            f"updater ({self.up_args.up_search_itrs})")
        to_q: Queue = ctx.Queue()
        num_to_send_per: List[int] = split_evenly_w_max(num_searches, self.up_args.up_procs, self.up_args.up_batch_size)
        start_idx: int = 0
        for num_to_send_per_i in num_to_send_per:
            if num_to_send_per_i > 0:
                to_q.put((num_to_send_per_i, start_idx))
                start_idx += (num_to_send_per_i * self.up_args.up_search_itrs)
        assert start_idx == num_gen

        return to_q


HV = TypeVar('HV', bound=HeurFnV)
PV = TypeVar('PV', bound=PathFindV)


class UpdateHeurV(UpdaterHeur[HV, PV]):
    def __init__(self, env: Environment, heur_nnet: HeurNNetV, up_args: UpHeurArgs):
        super().__init__(env, heur_nnet, up_args)
        self.heur_nnet: HeurNNetV = heur_nnet

    @abstractmethod
    def get_pathfind(self) -> PV:
        pass

    def step_get_in_out_np(self, pathfind: PV, heur_fn: HV,
                           times: Times) -> Tuple[List[NDArray], List[float]]:
        # take a step
        states, goals, ctgs_backup = pathfind.step(heur_fn)

        # to np
        start_time = time.time()
        inputs_np: List[NDArray] = self.heur_nnet.to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)
        return inputs_np, ctgs_backup

    def get_input_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.env.get_start_goal_pairs([0])
        inputs_nnet: List[NDArray[Any]] = self.heur_nnet.to_np(states, goals)

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dypes


HQ = TypeVar('HQ', bound=HeurFnQ)
PQ = TypeVar('PQ', bound=PathFindQ)


class UpdateHeurQ(UpdaterHeur[HQ, PQ]):
    def __init__(self, env: Environment, heur_nnet: HeurNNetQ, up_args: UpHeurArgs):
        super().__init__(env, heur_nnet, up_args)
        self.heur_nnet: HeurNNetQ = heur_nnet

    @abstractmethod
    def get_pathfind(self) -> PQ:
        pass

    def step_get_in_out_np(self, pathfind: PQ, heur_fn: HQ, times: Times) -> Tuple[List[NDArray], List[float]]:
        # take a step
        states, goals, actions, ctgs_backup = pathfind.step(heur_fn)

        # to_np
        start_time = time.time()
        inputs_np: List[NDArray] = self.heur_nnet.to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)
        return inputs_np, ctgs_backup

    def get_input_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.env.get_start_goal_pairs([0])
        actions: List[Action] = self.env.get_state_action_rand(states)
        inputs_nnet: List[NDArray[Any]] = self.heur_nnet.to_np(states, goals, [[action] for action in actions])

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dypes
