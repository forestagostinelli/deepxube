from typing import List, Tuple, Union, cast, Dict, Any
from dataclasses import dataclass
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.nnet.nnet_utils import HeurFnQ
from deepxube.environments.environment_abstract import State, Environment, Goal

from deepxube.search.search_abstract import Search, Instance
from deepxube.search.bwas import BWAS
from deepxube.search.greedy_policy import Greedy
from deepxube.search.search_utils import SearchPerf, search_test
from deepxube.utils.misc_utils import split_evenly
from deepxube.utils.data_utils import sel_l
from deepxube.utils.timing_utils import Times
from deepxube.utils.data_utils import SharedNDArray

import torch
from torch import Tensor
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess

import os
import pickle

import numpy as np
from numpy.typing import NDArray
import time

import sys
import shutil
import random


TrainData = Tuple[List[NDArray], NDArray]


@dataclass
class TrainArgs:
    """
    :param batch_size: Batch size
    :param lr: Initial learning rate
    :param lr_d: Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)
    :param max_itrs: Maximum number of iterations
    :param display: Number of iterations to display progress. No display if 0.
    """
    batch_size: int
    lr: float
    lr_d: float
    max_itrs: int
    display: bool


@dataclass
class UpdateArgs:
    """ Search performance is printed after update. This will most likely be higher than greedy since the steps taken
    to generate instances are balanced according to solve performance and because, each time an instance is solved, a
    new one is created with the same number of steps to maintain training data balance.

    :param up_itrs: How many iterations to do before checking if target network should be updated
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_nnet_batch_size: Batch size of each nnet used for each process update. Make smaller if running out
    of memory.
    :param up_search: greedy or astar
    :param up_step_max: Maximum number of search steps to take from generated start states to generate additional data.
    Increasing this number could make the heuristic function more robust to depression regions.
    :param up_eps_max_greedy: epsilon greedy policy max. Each start/goal pair will have an eps uniformly distributed
    between 0 and greedy_update_eps_max
    """
    up_itrs: int
    up_procs: int
    up_nnet_batch_size: int
    up_search: str
    up_step_max: int
    up_eps_max_greedy: float


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


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.inputs: List[NDArray] = []
        self.ctgs: NDArray = np.empty(0)
        self.max_size: int = max_size
        self.curr_size: int = 0
        self.add_idx: int = 0

    def add(self, inputs_add: List[NDArray], ctgs_add: NDArray):
        self.curr_size = min(self.curr_size + ctgs_add.shape[0], self.max_size)
        if self.ctgs.shape[0] == 0:
            # first add
            start_time = time.time()
            print("Input array sizes:")
            for input_idx in range(len(inputs_add)):
                print(f"index: {input_idx}, dtype: {inputs_add[input_idx].dtype}, shape:",
                      inputs_add[input_idx].shape[1:])
                self.inputs.append(inputs_add[input_idx][:self.max_size])
            self.ctgs = ctgs_add[:self.max_size]

            self.add_idx = self.curr_size

            print(f"Initializing replay buffer with max size {format(self.max_size, ',')}")
            rep_num: int = self.max_size - self.curr_size
            for input_idx in range(len(self.inputs)):
                inputs_add_idx_rep: NDArray = np.repeat(inputs_add[input_idx][0:1], rep_num, axis=0)
                self.inputs[input_idx] = np.concatenate((self.inputs[input_idx], inputs_add_idx_rep), axis=0)

            ctgs_add_rep: NDArray = np.repeat(ctgs_add[0:1], rep_num, axis=0)
            self.ctgs = np.concatenate((self.ctgs, ctgs_add_rep), axis=0)
            print(f"Replay buffer initialized. Time: {time.time() - start_time}")
        else:
            self._add_circular(inputs_add, ctgs_add)

    def sample(self, num: int) -> TrainData:
        sel_idxs: NDArray = np.random.randint(self.size(), size=num)

        inputs_samp: List[NDArray] = sel_l(self.inputs, sel_idxs)
        ctgs_samp: NDArray = self.ctgs[sel_idxs]

        return inputs_samp, ctgs_samp

    def size(self) -> int:
        return self.curr_size

    def clear(self):
        self.curr_size: int = 0
        self.add_idx: int = 0

    def _add_circular(self, inputs_add: List[NDArray], ctgs_add: NDArray):
        start_idx: int = 0
        assert len(self.inputs) == len(inputs_add), "should have same number of arrays"
        while start_idx < ctgs_add.shape[0]:
            num_add: int = min(ctgs_add.shape[0] - start_idx, self.max_size - self.add_idx)
            end_idx: int = start_idx + num_add
            add_idx_end: int = self.add_idx + num_add

            for input_idx in range(len(self.inputs)):
                self.inputs[input_idx][self.add_idx:add_idx_end] = inputs_add[input_idx][start_idx:end_idx]
            self.ctgs[self.add_idx:add_idx_end] = ctgs_add[start_idx:end_idx]

            start_idx = end_idx
            self.add_idx = add_idx_end
            if self.add_idx == self.max_size:
                self.add_idx = 0


def update_runner(gen_step_max: int, heur_fn_q: HeurFnQ, env: Environment, data_q: Queue,
                  up_args: UpdateArgs, step_probs: NDArray, inputs_nnet_shm: SharedNDArray,
                  ctgs_shm: SharedNDArray, batch_size: int, start_idx: int):
    times: Times = Times()

    up_search: str = up_args.up_search.upper()
    heur_fn = heur_fn_q.get_heuristic_fn(env)
    search_perf: SearchPerf = SearchPerf()

    search: Search
    if up_search == "GREEDY":
        search: Search = Greedy(env)
    elif up_search == "ASTAR":
        search: Search = BWAS(env)
    else:
        raise ValueError(f"Unknown search method {up_args.up_search}")

    insts_rem: List[Instance] = []
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

            # get kwargs and instance information
            start_time = time.time()
            kwargs: Dict[str, Any] = dict()
            inst_infos: List[Any]
            if up_search == "GREEDY":
                if len(search.instances) == 0:
                    eps_gen_l = list(np.random.rand(batch_size) * up_args.up_eps_max_greedy)
                else:
                    eps_gen_l = [float(inst.inst_info[1]) for inst in insts_rem]
                inst_infos: List[Tuple[int, float]] = [(step_gen, eps_gen)
                                                       for step_gen, eps_gen in zip(steps_gen, eps_gen_l)]
                kwargs['eps_l'] = eps_gen_l
            elif up_search == "ASTAR":
                inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
            times.record_time("inst_info", time.time() - start_time)

            search.add_instances(states_gen, goals_gen, heur_fn, inst_infos=inst_infos, **kwargs)

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
            inputs_nnet_shm[input_idx][start_idx:end_idx] = states_goals_nnet[input_idx]
        ctgs_shm[start_idx:end_idx] = ctgs_bellman
        data_q.put((start_idx, end_idx))
        start_idx = end_idx
        times.record_time("put", time.time() - start_time)

        # remove instances
        insts_rem: List[Instance] = search.remove_finished_instances(up_args.up_step_max)
        for inst_rem in insts_rem:
            search_perf.update_perf(inst_rem)

    times.add_times(search.times, path=["search"])

    data_q.put((times, search_perf))


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


def train_nnet(nnet: nn.Module, rb: ReplayBuffer, optimizer: Optimizer, device: torch.device,  num_itrs: int,
               train_itr: int, train_args: TrainArgs) -> float:
    # optimization
    criterion = nn.MSELoss()

    # initialize status tracking
    start_time = time.time()

    # train network
    nnet.train()
    max_itrs: int = train_itr + num_itrs

    last_loss: float = np.inf
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = train_args.lr * (train_args.lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        inputs_batch_np, ctgs_batch_np = rb.sample(train_args.batch_size)
        ctgs_batch_np = ctgs_batch_np.astype(np.float32)

        # send data to device
        inputs_batch: List[Tensor] = nnet_utils.to_pytorch_input(inputs_batch_np, device)
        ctgs_batch: Tensor = torch.tensor(ctgs_batch_np, device=device)

        # forward
        ctgs_nnet: Tensor = nnet(inputs_batch)

        # loss
        loss = criterion(ctgs_nnet[:, 0], ctgs_batch)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        last_loss = loss.item()
        # display progress
        if (train_args.display > 0) and (train_itr % train_args.display == 0):
            print("Itr: %i, lr: %.2E, loss: %.2E, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), ctgs_batch.mean().item(),
                      ctgs_nnet.mean().item(), time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

    return last_loss


def get_update_data(env: Environment, step_max: int, up_args: UpdateArgs, train_args: TrainArgs, status: Status,
                    rb: ReplayBuffer, targ_file: str, device: torch.device, on_gpu: bool):
    # update
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
                                             f"input{nnet_idx}", True))
    ctgs_shm = SharedNDArray((num_gen_up,), np.float64, f"ctgs", True)

    # sending index data to processes
    ctx = get_context("spawn")
    assert num_gen_up % up_args.up_step_max == 0, (f"Number of instances to generate per update "
                                                   f"(batch_size * up_itrs = {num_gen_up}), is not divisible by "
                                                   f"the max number of search steps to take during the "
                                                   f"update ({up_args.up_step_max})")
    # starting processes
    data_q: Queue = ctx.Queue()
    procs: List[BaseProcess] = []

    step_prob_str: str = ', '.join([f'{step_num}:{step_prob:.2f}'
                                    for step_num, step_prob in zip(range(status.step_max + 1), status.step_probs)])
    num_to_send_procs: int = num_gen_up // up_args.up_step_max
    num_send_per_proc: List[int] = split_evenly(num_to_send_procs, up_args.up_procs)
    start_idx: int = 0
    print(f"Step probs: {step_prob_str}")
    for proc_id, heur_fn_q in enumerate(heur_fn_qs):
        num_send_proc: int = num_send_per_proc[proc_id]
        if num_send_proc == 0:
            continue
        proc = ctx.Process(target=update_runner, args=(step_max, heur_fn_q, env, data_q, up_args, status.step_probs,
                                                       inputs_nnet_shm, ctgs_shm, num_send_proc, start_idx))
        proc.daemon = True
        proc.start()
        procs.append(proc)
        start_idx += (num_send_proc * up_args.up_step_max)

    # getting data from processes
    times_up: Times = Times()
    search_perf: SearchPerf = SearchPerf()
    print(f"Generating {format(num_gen_up, ',')} training instances")
    display_counts: NDArray[np.int_] = np.linspace(0, num_gen_up, 10, dtype=int)
    start_time_gen = time.time()
    num_procs_done: int = 0
    num_gen_curr: int = 0
    while num_procs_done < len(procs):
        start_time = time.time()
        data_get: Union[Tuple[Times, SearchPerf], Tuple[int, int]] = data_q.get()
        times_up.record_time("get", time.time() - start_time)
        if type(data_get[0]) is Times:
            times_up.add_times(data_get[0])
            search_perf = search_perf.comb_perf(data_get[1])
            num_procs_done += 1
        else:
            start_time = time.time()
            start_idx: int = cast(int, data_get[0])
            end_idx: int = cast(int, data_get[1])
            inputs_nnet_get: List[NDArray] = []
            for inputs_idx in range(len(inputs_nnet_shm)):
                inputs_nnet_get.append(inputs_nnet_shm[inputs_idx][start_idx:end_idx].copy())
            ctgs_shm_get: NDArray = ctgs_shm[start_idx:end_idx].copy()
            rb.add(inputs_nnet_get, ctgs_shm_get)
            num_gen_curr += (end_idx - start_idx)
            times_up.record_time("rb", time.time() - start_time)
            if num_gen_curr >= min(display_counts):
                print(f"{num_gen_curr}/{num_gen_up} instances (%.2f%%) "
                      f"(Tot time: %.2f)" % (100 * num_gen_curr / num_gen_up, time.time() - start_time_gen))
                display_counts = display_counts[num_gen_curr < display_counts]

    mean_ctg = ctgs_shm.array.mean()
    min_ctg = ctgs_shm.array.min()
    max_ctg = ctgs_shm.array.max()
    print(f"Generated {format(num_gen_curr, ',')} training instances, "
          f"Replay buffer size: {format(rb.size(), ',')}")
    print(f"Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))
    print(search_perf.to_string())
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
    :param rb_past_up: amount of data from previous update checks to keep in replay buffer. Totaly replay buffer size
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

    # training
    rb: ReplayBuffer = ReplayBuffer(train_args.batch_size * up_args.up_itrs * rb_past_up)
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=train_args.lr)
    while status.itr < train_args.max_itrs:
        # update
        get_update_data(env, step_max, up_args, train_args, status, rb, targ_file, device, on_gpu)

        # train nnet
        print("Training model for update number %i for %i iterations" % (status.update_num, up_args.up_itrs))
        last_loss = train_nnet(nnet, rb, optimizer, device, up_args.up_itrs, status.itr, train_args)
        status.itr += up_args.up_itrs

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

        # test
        start_time = time.time()
        max_solve_steps: int = min(status.update_num + 2, step_max)

        print("Testing greedy policy with %i states and %i steps" % (len(status.states_start_t), max_solve_steps))
        per_solved, is_solved_all = search_test(env, status.states_start_t, status.goals_t, status.state_t_steps_l,
                                                up_args.up_procs, curr_file, up_args.up_nnet_batch_size, device, on_gpu,
                                                "greedy", max_solve_steps)
        print("Greedy policy solved (best): %.2f%% (%.2f%%)" % (per_solved, status.per_solved_best))
        status.update_step_probs(is_solved_all)

        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        update_nnet: bool
        if per_solved > status.per_solved_best:
            print("Updating target network")
            status.per_solved_best = per_solved
            update_nnet = True
        else:
            update_nnet = False

        print("Last loss was %f" % last_loss)
        if update_nnet:
            # Update nnet
            rb.clear()
            shutil.copy(curr_file, targ_file)
            status.update_num = status.update_num + 1

        # noinspection PyTypeChecker
        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
