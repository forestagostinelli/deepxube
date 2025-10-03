from typing import List, Tuple, Dict, Any, cast
from dataclasses import dataclass

from deepxube.training.train_utils import ReplayBuffer, train_heur, TrainArgs
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.nnet.nnet_utils import HeurFnQ
from deepxube.environments.environment_abstract import State, Environment, Goal

from deepxube.search.search_abstract_v import SearchV, InstanceV
from deepxube.search.bwas import BWAS
from deepxube.search.greedy_policy import Greedy, InstanceGrV
from deepxube.search.search_utils import SearchPerf, search_test
from deepxube.utils.timing_utils import Times
from deepxube.utils.data_utils import SharedNDArray
from deepxube.utils.misc_utils import split_evenly_w_max

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess
from torch.utils.tensorboard import SummaryWriter

import os
import pickle

import numpy as np
from numpy.typing import NDArray
import time

import sys
import shutil
import random


@dataclass
class UpdateArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data
    balance.

    :param up_itrs: How many iterations worth of data to generate per udpate
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_batch_size: Helps manage memory. Decrease if memory is running out during update.
    :param up_nnet_batch_size: Batch size of each nnet used for each process update. Make smaller if running out
    of memory.
    :param up_search: greedy or astar
    :param up_step_max: Maximum number of search steps to take from generated start states to generate additional data.
    Increasing this number could make the heuristic function more robust to depression regions.
    :param up_eps_max_greedy: epsilon greedy policy max. Each start/goal pair will have an eps uniformly distributed
    between 0 and greedy_update_eps_max
    :param up_epochs: do up_epochs * up_itrs iterations worth of training before checking for update. Can decrease data
    generation time, but can increase risk of overfitting between updates checks.
    :param up_test: greedy: update when greedy policy improves, const: update every update check
    """
    up_itrs: int
    up_procs: int
    up_batch_size: int
    up_nnet_batch_size: int
    up_search: str
    up_step_max: int
    up_eps_max_greedy: float
    up_epochs: int = 1
    up_test: str = "greedy"


class Status:
    def __init__(self, env: Environment, step_max: int, num_test_per_step: int, num_procs: int):
        self.itr: int = 0
        self.update_num: int = 0

        # generate data
        self.state_t_steps_l: List[int] = []
        for step in range(step_max + 1):
            self.state_t_steps_l.extend([step] * num_test_per_step)
        random.shuffle(self.state_t_steps_l)

        self.states_start_t: List[State]
        self.goals_t: List[Goal]
        print(f"Generating {num_test_per_step} test states per step ({step_max} steps, "
              f"{format(len(self.state_t_steps_l), ',')} total test states)")
        self.states_start_t, self.goals_t = env.get_start_goal_pairs(self.state_t_steps_l)

        # Initialize per_solved_best
        print("Initializing per solved best")
        start_time = time.time()
        per_solved, is_solved_all = search_test(env, self.states_start_t, self.goals_t, self.state_t_steps_l, num_procs,
                                                "", None, torch.device("cpu"), False, "greedy", 1)
        print("Greedy policy solved: %.2f%%" % per_solved)
        print("Test time: %.2f" % (time.time() - start_time))
        self.per_solved_best: float = per_solved

        self.step_probs: NDArray = np.zeros(step_max + 1)/(step_max + 1)
        self.step_max: int = step_max
        self.update_step_probs(is_solved_all)

    def update_step_probs(self, is_solved_all: NDArray):
        per_solved_per_step_l: List[float] = []
        for step in range(self.step_max + 1):
            step_idxs: NDArray = np.where(np.array(self.state_t_steps_l) == step)[0]
            per_solved_per_step_l.append(100.0 * float(np.mean(is_solved_all[step_idxs])))
        per_solved_per_step: NDArray = np.array(per_solved_per_step_l)

        num_no_soln: int = np.sum(per_solved_per_step == 0)
        if num_no_soln == 0:
            self.step_probs: NDArray = per_solved_per_step / per_solved_per_step.sum()
        else:
            num_w_soln_eff: float = per_solved_per_step.sum() / 100.0
            num_tot_eff: float = num_w_soln_eff + 1
            self.step_probs: NDArray = num_w_soln_eff * per_solved_per_step / per_solved_per_step.sum() / num_tot_eff
            self.step_probs[per_solved_per_step == 0] = 1 / num_tot_eff / num_no_soln


def update_runner(gen_step_max: int, heur_fn_q: HeurFnQ, env: Environment, to_q: Queue, data_q: Queue,
                  up_args: UpdateArgs, step_probs: NDArray, inputs_nnet_shm: List[SharedNDArray],
                  ctgs_shm: SharedNDArray):
    times: Times = Times()

    up_search: str = up_args.up_search.upper()
    heur_fn = heur_fn_q.get_heuristic_fn(env)
    step_to_search_perf: Dict[int, SearchPerf] = dict()
    while True:
        batch_size, start_idx = to_q.get()
        if batch_size is None:
            break

        search: SearchV
        if up_search == "GREEDY":
            search: SearchV = Greedy(env)
        elif up_search == "ASTAR":
            search: SearchV = BWAS(env)
        else:
            raise ValueError(f"Unknown search method {up_args.up_search}")

        insts_rem: List[InstanceV] = []
        start_idx_batch: int = start_idx
        for _ in range(up_args.up_step_max):
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
                if up_search == "GREEDY":
                    if len(search.instances) == 0:
                        kwargs['eps_l'] = list(np.random.rand(batch_size) * up_args.up_eps_max_greedy)
                    else:
                        kwargs['eps_l'] = [cast(InstanceGrV, inst).eps for inst in insts_rem]
                times.record_time("inst_info", time.time() - start_time)

                search.add_instances(states_gen, goals_gen, heur_fn, inst_infos=inst_infos, compute_init_heur=False,
                                     **kwargs)

            # take a step
            states, goals, ctgs_bellman = search.step(heur_fn)

            # to nnet
            start_time = time.time()
            states_goals_nnet: List[NDArray] = env.states_goals_to_nnet_input(states, goals)
            times.record_time("to_nnet", time.time() - start_time)

            # put
            start_time = time.time()
            end_idx = start_idx + len(states)
            assert len(states) == batch_size
            for input_idx in range(len(states_goals_nnet)):
                inputs_nnet_shm[input_idx][start_idx:end_idx] = states_goals_nnet[input_idx].copy()
            ctgs_shm[start_idx:end_idx] = ctgs_bellman.copy()
            start_idx = end_idx
            times.record_time("put", time.time() - start_time)

            # remove instances
            insts_rem: List[InstanceV] = search.remove_finished_instances(up_args.up_step_max)

            # search performance
            for inst_rem in insts_rem:
                step_num_inst: int = int(inst_rem.inst_info[0])
                if step_num_inst not in step_to_search_perf.keys():
                    step_to_search_perf[step_num_inst] = SearchPerf()
                step_to_search_perf[step_num_inst].update_perf(inst_rem)

        data_q.put((start_idx_batch, start_idx))
        times.add_times(search.times, path=["search"])

    data_q.put((times, step_to_search_perf))
    for arr_shm in inputs_nnet_shm + [ctgs_shm]:
        arr_shm.close()


def load_data(model_dir: str, nnet_file: str, env: Environment, num_test_per_step: int,
              step_max: int, num_procs: int) -> Tuple[nn.Module, Status]:
    status_file: str = "%s/status.pkl" % model_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_v_nnet())
    else:
        nnet = env.get_v_nnet()

    status: Status
    if os.path.isfile(status_file):
        status = pickle.load(open("%s/status.pkl" % model_dir, "rb"))
        print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}, "
              f"per_solved_best: {status.per_solved_best}")
    else:
        status = Status(env, step_max, num_test_per_step, num_procs)
        # noinspection PyTypeChecker
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return nnet, status


def get_update_data(env: Environment, step_max: int, up_args: UpdateArgs, train_args: TrainArgs, status: Status,
                    rb: ReplayBuffer, targ_file: str, device: torch.device, on_gpu: bool, writer: SummaryWriter):
    # update heuristic functions
    num_gen_up: int = train_args.batch_size * up_args.up_itrs
    all_zeros: bool = not os.path.isfile(targ_file)
    heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(up_args.up_procs, targ_file, device, on_gpu, env, "V",
                                                              all_zeros=all_zeros, clip_zero=True,
                                                              batch_size=up_args.up_nnet_batch_size)

    # shared memory
    states, goals = env.get_start_goal_pairs([0])
    inputs_nnet: List[NDArray] = env.states_goals_to_nnet_input(states, goals)
    inputs_nnet_shm: List[SharedNDArray] = []
    for nnet_idx, inputs_nnet_i in enumerate(inputs_nnet):
        inputs_nnet_shm.append(SharedNDArray((num_gen_up,) + inputs_nnet_i[0].shape, inputs_nnet_i.dtype,
                                             None, True))
    ctgs_shm = SharedNDArray((num_gen_up,), np.float64, None, True)

    # sending index data to processes
    ctx = get_context("spawn")
    assert num_gen_up % up_args.up_step_max == 0, (f"Number of instances to generate per update "
                                                   f"(batch_size * up_itrs = {num_gen_up}), is not divisible by "
                                                   f"the max number of search steps to take during the "
                                                   f"update ({up_args.up_step_max})")
    to_q: Queue = ctx.Queue()
    num_searches: int = num_gen_up // up_args.up_step_max
    num_to_send_per: List[int] = split_evenly_w_max(num_searches, up_args.up_procs, up_args.up_batch_size)
    start_idx: int = 0
    for num_to_send_per_i in num_to_send_per:
        if num_to_send_per_i > 0:
            to_q.put((num_to_send_per_i, start_idx))
            start_idx += (num_to_send_per_i * up_args.up_step_max)
    assert start_idx == num_gen_up

    # starting processes
    data_q: Queue = ctx.Queue()
    procs: List[BaseProcess] = []

    step_prob_str: str = ', '.join([f'{step_num}:{step_prob:.2f}'
                                    for step_num, step_prob in zip(range(status.step_max + 1), status.step_probs)])
    print(f"Step probs: {step_prob_str}")
    for proc_id, heur_fn_q in enumerate(heur_fn_qs):
        proc = ctx.Process(target=update_runner, args=(step_max, heur_fn_q, env, to_q, data_q, up_args,
                                                       status.step_probs, inputs_nnet_shm, ctgs_shm))
        proc.daemon = True
        proc.start()
        procs.append(proc)

    # getting data from processes
    times_up: Times = Times()
    print(f"Generating {format(num_gen_up, ',')} training instances with {num_searches} searches")
    display_counts: NDArray[np.int_] = np.linspace(0, num_gen_up, 10, dtype=int)
    start_time_gen = time.time()
    num_gen_curr: int = 0
    while num_gen_curr < num_gen_up:
        start_idx, end_idx = data_q.get()
        start_time = time.time()
        inputs_nnet_get: List[NDArray] = []
        for inputs_idx in range(len(inputs_nnet_shm)):
            inputs_nnet_get.append(inputs_nnet_shm[inputs_idx][start_idx:end_idx].copy())
        ctgs_shm_get: NDArray = ctgs_shm[start_idx:end_idx].copy()
        rb.add(inputs_nnet_get + [ctgs_shm_get])
        num_gen_curr += (end_idx - start_idx)
        times_up.record_time("rb", time.time() - start_time)
        if num_gen_curr >= min(display_counts):
            print(f"{num_gen_curr}/{num_gen_up} instances (%.2f%%) "
                  f"(Tot time: %.2f)" % (100 * num_gen_curr / num_gen_up, time.time() - start_time_gen))
            display_counts = display_counts[num_gen_curr < display_counts]

    # sending stop signal
    for _ in procs:
        to_q.put((None, None))

    # get summary from processes
    step_to_search_perf: Dict[int, SearchPerf] = dict()
    for _ in procs:
        times_up_i, step_to_search_perf_i  = data_q.get()
        times_up.add_times(times_up_i)
        for step_num_perf, search_perf in step_to_search_perf_i.items():
            if step_num_perf not in step_to_search_perf.keys():
                step_to_search_perf[step_num_perf] = SearchPerf()
            step_to_search_perf[step_num_perf] = step_to_search_perf[step_num_perf].comb_perf(search_perf)

    # print summary
    mean_ctg = ctgs_shm.array.mean()
    min_ctg = ctgs_shm.array.min()
    max_ctg = ctgs_shm.array.max()
    print(f"Generated {format(num_gen_curr, ',')} training instances, "
          f"Replay buffer size: {format(rb.size(), ',')}")
    print(f"Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))
    per_solved_l: List[float] = []
    path_cost_ave_l: List[float] = []
    search_itrs_ave_l: List[float] = []
    for search_perf in step_to_search_perf.values():
        per_solved_i, path_cost_ave_i, search_itrs_ave_i = search_perf.stats()
        per_solved_l.append(per_solved_i)
        if per_solved_i > 0.0:
            path_cost_ave_l.append(path_cost_ave_i)
            search_itrs_ave_l.append(search_itrs_ave_i)
    per_solved_ave: float = float(np.mean(per_solved_l))
    path_costs_ave: float = float(np.mean(path_cost_ave_l))
    search_itrs_ave: float = float(np.mean(search_itrs_ave_l))
    print(f"%solved: {per_solved_ave:.2f}, path_costs: {path_costs_ave:.3f}, "
          f"search_itrs: {search_itrs_ave:.3f} (equally weighted across step numbers)")
    writer.add_scalar("solved (update)", per_solved_ave, status.itr)
    writer.add_scalar("path_cost (update)", path_costs_ave, status.itr)
    writer.add_scalar("search_itrs (update)", search_itrs_ave, status.itr)
    print(f"Times - {times_up.get_time_str()}")

    # clean up
    nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)
    for proc in procs:
        proc.join()
    for arr_shm in inputs_nnet_shm + [ctgs_shm]:
        arr_shm.close()
        arr_shm.unlink()


def train(env: Environment, step_max: int, nnet_dir: str, train_args: TrainArgs, up_args: UpdateArgs,
          rb_past_up: int = 10, num_test_per_step: int = 30, debug: bool = False):
    """ Train a deep neural network heuristic (DNN) function with deep approximate value iteration (DAVI).
    A target DNN is maintained for computing the updated heuristic values. When the greedy policy improves on a fixed
    test set, the target DNN is updated to be the current DNN. The number of steps taken for testing the greedy policy
    is the minimum between the number of target DNN updates and step_max.
    This makes the test a lot faster in the earlier stages, espeicially when step_max is large.

    For more information see:
    - Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and search."
    Nature Machine Intelligence 1.8 (2019): 356-363.
    - Bertsekas, D. P. & Tsitsiklis, J. N. Neuro-dynamic Programming (Athena Scientific, 1996).

    :param env: an Environment object
    :param step_max: maximum number of steps to take to generate start/goal pairs
    :param nnet_dir: directory where DNN will be saved
    :param train_args: training arguments
    :param up_args: update arguments
    :param rb_past_up: amount of data from previous update checks to keep in replay buffer. Total replay buffer size
    will then be train_args.batch_size * up_args.up_itrs * rb_past_up. The replay buffer is cleared after the
    target network is updated.
    :param num_test_per_step: Number of test states for each step between 0 and step_max
    :param debug: Turns off logging to make typing during breakpoints easier
    :return: None
    """
    # Initialization
    targ_file: str = f"{nnet_dir}/target.pt"
    curr_file = f"{nnet_dir}/current.pt"
    output_save_loc = "%s/output.txt" % nnet_dir
    writer: SummaryWriter = SummaryWriter(nnet_dir)

    if not os.path.exists(nnet_dir):
        os.makedirs(nnet_dir)

    if not debug:
        sys.stdout = data_utils.Logger(output_save_loc, "a")  # type: ignore

    # Print basic info
    # print("HOST: %s" % os.uname()[1])
    print(f"Train args: {train_args}")
    print(f"Update args: {up_args}")
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # load nnet
    print("Loading nnet and status")
    nnet, status = load_data(nnet_dir, curr_file, env, num_test_per_step, step_max, up_args.up_procs)
    nnet.to(device)
    nnet = nn.DataParallel(nnet)

    # initialize replay buffer
    states, goals = env.get_start_goal_pairs([0])
    inputs_nnet: List[NDArray] = env.states_goals_to_nnet_input(states, goals)
    rb_shapes: List[Tuple[int, ...]] = []
    rb_dtypes: List[np.dtype] = []
    for nnet_idx, inputs_nnet_i in enumerate(inputs_nnet):
        rb_shapes.append(inputs_nnet_i[0].shape)
        rb_dtypes.append(inputs_nnet_i.dtype)
    rb_shapes.append(tuple())
    rb_dtypes.append(np.dtype(np.float64))
    rb: ReplayBuffer = ReplayBuffer(train_args.batch_size * up_args.up_itrs * rb_past_up, rb_shapes, rb_dtypes)

    # training
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=train_args.lr)
    criterion = nn.MSELoss()
    while status.itr < train_args.max_itrs:
        # update
        get_update_data(env, step_max, up_args, train_args, status, rb, targ_file, device, on_gpu, writer)

        # get batches
        print("Getting training batches")
        start_time = time.time()
        batches: List[Tuple[List[NDArray], NDArray]] = []
        for _ in range(up_args.up_itrs * up_args.up_epochs):
            arrays_samp: List[NDArray] = rb.sample(train_args.batch_size)
            inputs_batch_np: List[NDArray] = arrays_samp[:-1]
            ctgs_batch_np: NDArray = np.expand_dims(arrays_samp[-1].astype(np.float32), 1)
            batches.append((inputs_batch_np, ctgs_batch_np))
        print(f"Time: {time.time() - start_time}")

        # train nnet
        print("Training model for update number %i for %i iterations" % (status.update_num, len(batches)))
        last_loss = train_heur(nnet, batches, optimizer, criterion, device, status.itr, train_args)
        status.itr += len(batches)

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

        # test
        start_time = time.time()
        max_solve_steps: int = min(status.update_num + 2, step_max)

        print("Testing greedy policy with %i states and %i steps" % (len(status.states_start_t), max_solve_steps))
        per_solved, is_solved_all = search_test(env, status.states_start_t, status.goals_t, status.state_t_steps_l,
                                                up_args.up_procs, curr_file, up_args.up_nnet_batch_size, device, on_gpu,
                                                "greedy", max_solve_steps)
        writer.add_scalar("per_solved (greedy)", per_solved, status.itr)
        print("Greedy policy solved (best): %.2f%% (%.2f%%)" % (per_solved, status.per_solved_best))
        status.update_step_probs(is_solved_all)

        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        update_nnet: bool
        if up_args.up_test.upper() == "GREEDY":
            update_nnet = per_solved > status.per_solved_best
        elif up_args.up_test.upper() == "CONST":
            update_nnet = True
        else:
            raise ValueError(f"Unknown update test {up_args.up_test}")

        print("Last loss was %f" % last_loss)
        status.per_solved_best = max(status.per_solved_best, per_solved)
        if update_nnet:
            # Update nnet
            print("Updating target network")
            rb.clear()
            shutil.copy(curr_file, targ_file)
            status.update_num = status.update_num + 1

        # noinspection PyTypeChecker
        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
