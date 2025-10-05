from typing import List, Dict, Tuple, Any, cast
import os
import time
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess

import numpy as np
import torch
from numpy.typing import NDArray

from deepxube.nnet.nnet_utils import NNetParInfo
from deepxube.base.environment import Environment, State, Goal, Action
from deepxube.base.heuristic import HeurNNet, NNetType, HeurNNetV, NNetParQ
from deepxube.nnet import nnet_utils
from deepxube.pathfinding.v.bwas import BWAS
from deepxube.base.pathfinding import PathFind, Instance, PathFindV, PathFindQ
from deepxube.pathfinding.pathfinding_utils import PathFindPerf
from deepxube.training.train_utils import ReplayBuffer, get_single_nnet_input
from deepxube.utils.data_utils import SharedNDArray
from deepxube.utils.misc_utils import split_evenly_w_max
from deepxube.utils.timing_utils import Times


@dataclass
class UpdateArgs:
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


def update_runner(gen_step_max: int, search_step_max: int, env: Environment, nnet_par_info: NNetParInfo,
                  heur_nnet: HeurNNet, nnet_type: NNetType, to_q: Queue, from_q: Queue, step_probs: NDArray,
                  inputs_nnet_shm: List[SharedNDArray], ctgs_shm: SharedNDArray):
    times: Times = Times()

    heur_fn = heur_nnet.get_nnet_par_fn(nnet_par_info)
    step_to_search_perf: Dict[int, PathFindPerf] = dict()
    while True:
        batch_size, start_idx = to_q.get()
        if batch_size is None:
            break

        if nnet_type is NNetType.V:
            search: PathFind = BWAS(env)
        elif nnet_type is NNetType.Q:
            search: PathFind = BWQS(env)
            pass

        insts_rem: List[Instance] = []
        start_idx_batch: int = start_idx
        for _ in range(search_step_max):
            # add instances
            if (len(search.instances) == 0) or (len(insts_rem) > 0):
                times_states: Times = Times()
                # get steps generate
                start_time = time.time()
                steps_gen: List[int]
                if len(search.instances) == 0:
                    steps_gen = list(np.random.choice(gen_step_max + 1, size=batch_size, p=step_probs))
                else:
                    steps_gen = [int(inst.inst_info[0]) for inst in insts_rem]
                times_states.record_time("steps_gen", time.time() - start_time)

                # generate states
                states_gen, goals_gen = env.get_start_goal_pairs(steps_gen, times=times_states)
                times.add_times(times_states, ["get_states"])

                # get instance information and kwargs
                start_time = time.time()
                inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
                kwargs: Dict[str, Any] = dict()
                times.record_time("inst_info", time.time() - start_time)

                search.add_instances(states_gen, goals_gen, heur_fn, inst_infos=inst_infos, compute_init_heur=True,
                                     **kwargs)

            # take a step
            search_ret: Any = search.step(heur_fn)

            # to nnet
            start_time = time.time()
            inputs_nnet: List[NDArray]
            ctgs_bellman: List[float]
            if isinstance(search, PathFindV):
                states, goals, ctgs_bellman = cast(Tuple[List[State], List[Goal], List[float]], search_ret)
                inputs_nnet = cast(HeurNNetV, heur_nnet).to_np(states, goals)
            elif isinstance(search, PathFindQ):
                states, goals, actions, ctgs_bellman = cast(Tuple[List[State], List[Goal], List[Action], List[float]],
                                                            search_ret)
                actions_l: List[List[Action]] = [[action] for action in actions]
                inputs_nnet = cast(NNetParQ, heur_nnet).to_nnet(states, goals, actions_l)
            else:
                raise ValueError(f"Unknown pathfinding class {search.__class__}")
            times.record_time("to_nnet", time.time() - start_time)

            # put
            start_time = time.time()
            end_idx = start_idx + len(ctgs_bellman)
            assert len(ctgs_bellman) == batch_size
            for input_idx in range(len(inputs_nnet)):
                inputs_nnet_shm[input_idx][start_idx:end_idx] = inputs_nnet[input_idx].copy()
            ctgs_shm[start_idx:end_idx] = np.array(ctgs_bellman).copy()
            start_idx = end_idx
            times.record_time("put", time.time() - start_time)

            # remove instances
            insts_rem: List[Instance] = search.remove_finished_instances(search_step_max)

            # pathfinding performance
            for inst_rem in insts_rem:
                step_num_inst: int = int(inst_rem.inst_info[0])
                if step_num_inst not in step_to_search_perf.keys():
                    step_to_search_perf[step_num_inst] = PathFindPerf()
                step_to_search_perf[step_num_inst].update_perf(inst_rem)

        from_q.put((start_idx_batch, start_idx))
        times.add_times(search.times, path=["pathfinding"])

    from_q.put((times, step_to_search_perf))
    for arr_shm in inputs_nnet_shm + [ctgs_shm]:
        arr_shm.close()


def get_update_data(env: Environment, heur_nnet: HeurNNet, step_max: int, step_probs: NDArray, num_gen: int,
                    up_args: UpdateArgs, rb: ReplayBuffer, nnet_type: NNetType, targ_file: str, device: torch.device,
                    on_gpu: bool) -> Dict[int, PathFindPerf]:
    start_time_gen = time.time()
    num_searches: int = num_gen // up_args.up_search_itrs
    print(f"Generating {format(num_gen, ',')} training instances with {format(num_searches, ',')} searches")
    # updater heuristic functions
    all_zeros: bool = not os.path.isfile(targ_file)
    nnet_par_infos, nnet_procs = nnet_utils.start_nnet_fn_runners(heur_nnet.get_nnet, up_args.up_procs, targ_file,
                                                                  device, on_gpu, all_zeros=all_zeros, clip_zero=True,
                                                                  batch_size=up_args.up_nnet_batch_size)

    # shared memory
    inputs_nnet: List[NDArray] = get_single_nnet_input(env, heur_nnet)
    inputs_nnet_shm: List[SharedNDArray] = []
    for nnet_idx, inputs_nnet_i in enumerate(inputs_nnet):
        inputs_nnet_shm.append(SharedNDArray((num_gen,) + inputs_nnet_i[0].shape, inputs_nnet_i.dtype,
                                             None, True))
    ctgs_shm = SharedNDArray((num_gen,), np.float64, None, True)

    # sending index data to processes
    ctx = get_context("spawn")
    assert num_gen % up_args.up_search_itrs == 0, (f"Number of instances to generate per for this updater {num_gen} is not "
                                                f"divisible by the max number of pathfinding iterations to take during the "
                                                f"updater ({up_args.up_search_itrs})")
    to_q: Queue = ctx.Queue()
    num_to_send_per: List[int] = split_evenly_w_max(num_searches, up_args.up_procs, up_args.up_batch_size)
    start_idx: int = 0
    for num_to_send_per_i in num_to_send_per:
        if num_to_send_per_i > 0:
            to_q.put((num_to_send_per_i, start_idx))
            start_idx += (num_to_send_per_i * up_args.up_search_itrs)
    assert start_idx == num_gen

    # starting processes
    from_q: Queue = ctx.Queue()
    procs: List[BaseProcess] = []

    for proc_id, nnet_par_info in enumerate(nnet_par_infos):
        proc = ctx.Process(target=update_runner, args=(step_max, up_args.up_search_itrs, env, nnet_par_info,
                                                       heur_nnet, nnet_type, to_q, from_q, step_probs,
                                                       inputs_nnet_shm, ctgs_shm))
        proc.daemon = True
        proc.start()
        procs.append(proc)

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
    step_to_search_perf: Dict[int, PathFindPerf] = dict()
    for _ in procs:
        times_up_i, step_to_search_perf_i  = from_q.get()
        times_up.add_times(times_up_i)
        for step_num_perf, search_perf in step_to_search_perf_i.items():
            if step_num_perf not in step_to_search_perf.keys():
                step_to_search_perf[step_num_perf] = PathFindPerf()
            step_to_search_perf[step_num_perf] = step_to_search_perf[step_num_perf].comb_perf(search_perf)

    # cost-to-go summary
    mean_ctg = ctgs_shm.array.mean()
    min_ctg = ctgs_shm.array.min()
    max_ctg = ctgs_shm.array.max()
    print(f"Generated {format(num_gen_curr, ',')} training instances, "
          f"Replay buffer size: {format(rb.size(), ',')}")
    print(f"Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))
    print(f"Times - {times_up.get_time_str()}")

    # clean up
    nnet_utils.stop_nnet_runners(nnet_procs, nnet_par_infos)
    for proc in procs:
        proc.join()
    for arr_shm in inputs_nnet_shm + [ctgs_shm]:
        arr_shm.close()
        arr_shm.unlink()

    return step_to_search_perf
