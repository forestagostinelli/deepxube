from typing import List, Tuple, Union, cast
from dataclasses import dataclass
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.nnet.nnet_utils import HeurFnQ
from deepxube.environments.environment_abstract import State, Environment, Goal
from deepxube.search.greedy_policy import Greedy, greedy_test
from deepxube.search.greedy_policy import Instance as InstanceGreedy
from deepxube.utils.data_utils import sel_l
from deepxube.utils.timing_utils import Times
from deepxube.utils import misc_utils

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
    """
    :param up_itrs: How many iterations to do before checking if target network should be updated
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_nnet_batch_size: Batch size of each nnet used for each process update. Make smaller if running out
    of memory.
    :param up_step_max: Maximum number of search steps to take from generated start states to generate additional data.
    Increasing this number could make the heuristic function more robust to depression regions.
    Max number of steps taken has an upper bound of up_itrs.
    :param up_eps_max_greedy: epsilon greedy policy max. Each start/goal pair will have an eps uniformly distributed
    between 0 and greedy_update_eps_max
    """
    up_itrs: int
    up_procs: int
    up_nnet_batch_size: int
    up_step_max: int
    up_eps_max_greedy: float


@dataclass
class SearchPerf:
    def __init__(self):
        self.is_solved_l: List[bool] = []
        self.path_costs: List[float] = []
        self.search_itrs_l: List[int] = []

    def update_perf(self, is_solved: bool, path_cost: float, search_itrs: int):
        self.is_solved_l.append(is_solved)
        if is_solved:
            self.path_costs.append(path_cost)
            self.search_itrs_l.append(search_itrs)

    def comb_perf(self, search_perf2: 'SearchPerf') -> 'SearchPerf':
        search_perf_new: SearchPerf = SearchPerf()
        search_perf_new.is_solved_l = self.is_solved_l + search_perf2.is_solved_l
        search_perf_new.path_costs = self.path_costs + search_perf2.path_costs
        search_perf_new.search_itrs_l = self.search_itrs_l + search_perf2.search_itrs_l

        return search_perf_new

    def stats(self) -> Tuple[float, float, float]:
        path_cost_ave: float = 0.0
        if len(self.path_costs) > 0:
            path_cost_ave: float = float(np.mean(self.path_costs))
        search_itrs_ave: float = float(np.mean(self.search_itrs_l))
        per_solved: float = 100.0 * float(np.mean(self.is_solved_l))

        return per_solved, path_cost_ave, search_itrs_ave

    def to_string(self) -> str:
        per_solved, path_cost_ave, search_itrs_ave = self.stats()
        return f"%solved: {per_solved:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f}"


class Status:
    def __init__(self, env: Environment, step_max: int, num_test_per_step: int):
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
        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(1, "", torch.device("cpu"), False, env, "V",
                                                                  all_zeros=True)
        per_solved: float = greedy_test(self.states_start_t, self.goals_t, self.state_t_steps_l, env,
                                        heur_fn_qs, max_solve_steps=1)
        print("Greedy policy solved: %.2f%%" % per_solved)
        self.per_solved_best: float = per_solved


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


def update_runner(batch_size: int, num_batches: int, step_max: int, heur_fn_q: HeurFnQ, env: Environment,
                  result_queue: Queue, up_args: UpdateArgs):
    times: Times = Times()

    def remove_instance_fn(inst_in: InstanceGreedy) -> bool:
        if inst_in.is_solved:
            return True
        if inst_in.step_num >= up_args.up_step_max:
            return True
        return False

    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    greedy: Greedy = Greedy(env)
    search_perf: SearchPerf = SearchPerf()
    for _ in range(num_batches):
        # remove instances
        insts_rem: List[InstanceGreedy] = greedy.remove_instances(remove_instance_fn)
        for inst in insts_rem:
            search_perf.update_perf(inst.is_solved, inst.curr_node.path_cost, inst.step_num)

        # add instances
        if (len(greedy.instances) < batch_size) or (len(insts_rem) > 0):
            steps_gen: List[int]
            eps_gen_l: List[float]
            if len(insts_rem) == 0:
                steps_gen = list(np.random.choice(step_max + 1, size=batch_size))
                eps_gen_l = list(np.random.rand(batch_size) * up_args.up_eps_max_greedy)
            else:
                steps_gen = [int(inst.inst_info) for inst in insts_rem]
                eps_gen_l = [inst.eps for inst in insts_rem]

            times_states: Times = Times()
            states_gen, goals_gen = env.get_start_goal_pairs(steps_gen, times=times_states)
            times.add_times(times_states, ["get_states"])

            greedy.add_instances(states_gen, goals_gen, eps_l=eps_gen_l, inst_infos=steps_gen)

        # take a step
        states, goals, ctgs_backup = greedy.step(heuristic_fn, times=times, rand_seen=True)

        # put in queue
        start_time = time.time()
        states_goals_nnet: List[NDArray] = env.states_goals_to_nnet_input(states, goals)
        times.record_time("to_nnet", time.time() - start_time)

        result_queue.put((states_goals_nnet, ctgs_backup))

    insts_rem: List[InstanceGreedy] = greedy.remove_instances(remove_instance_fn)
    for inst in insts_rem:
        search_perf.update_perf(inst.is_solved, inst.curr_node.path_cost, inst.step_num)
    result_queue.put((times, search_perf))


def load_data(model_dir: str, nnet_file: str, env: Environment, num_test_per_step: int,
              step_max: int) -> Tuple[nn.Module, Status]:
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
        status = Status(env, step_max, num_test_per_step)
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
    nnet, status = load_data(nnet_dir, curr_file, env, num_test_per_step, step_max)
    nnet.to(device)
    nnet = nn.DataParallel(nnet)

    # training
    rb: ReplayBuffer = ReplayBuffer(train_args.batch_size * up_args.up_itrs * rb_past_up)
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=train_args.lr)
    while status.itr < train_args.max_itrs:
        # update
        all_zeros: bool = not os.path.isfile(targ_file)
        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(up_args.up_procs, targ_file, device, on_gpu, env, "V",
                                                                  all_zeros=all_zeros, clip_zero=True,
                                                                  batch_size=up_args.up_nnet_batch_size)

        ctx = get_context("spawn")
        data_q: Queue = ctx.Queue()
        batches_per_procs: List[int] = misc_utils.split_evenly(up_args.up_itrs, up_args.up_procs)
        procs: List[BaseProcess] = []
        for proc_id, heur_fn_q in enumerate(heur_fn_qs):
            num_batches: int = batches_per_procs[proc_id]
            if num_batches == 0:
                continue
            proc = ctx.Process(target=update_runner, args=(train_args.batch_size, num_batches, step_max, heur_fn_q, env,
                                                           data_q, up_args))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        times_up: Times = Times()
        search_perf: SearchPerf = SearchPerf()
        ctgs_up: NDArray = np.zeros(0)
        num_inst_gen: int = up_args.up_itrs * train_args.batch_size
        print(f"Generating {format(num_inst_gen, ',')} training instances")
        display_counts: NDArray[np.int_] = np.linspace(0, num_inst_gen, 10, dtype=int)
        start_time_gen = time.time()
        num_procs_done: int = 0
        while num_procs_done < len(procs):
            data_get: Union[Tuple[Times, SearchPerf], TrainData] = data_q.get()
            if type(data_get[0]) is Times:
                times_up.add_times(data_get[0])
                search_perf = search_perf.comb_perf(data_get[1])
                num_procs_done += 1
            else:
                train_data: TrainData = cast(TrainData, data_get)
                rb.add(train_data[0], train_data[1])
                ctgs_up = np.concatenate((ctgs_up, train_data[1]), axis=0)
                num_gen_curr: int = ctgs_up.shape[0]
                if num_gen_curr >= min(display_counts):
                    print(f"{num_gen_curr}/{num_inst_gen} instances (%.2f%%) "
                          f"(Data time: %.2f)" % (100 * num_gen_curr / num_inst_gen, time.time() - start_time_gen))
                    display_counts = display_counts[num_gen_curr < display_counts]


        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)
        for proc in procs:
            proc.join()

        mean_ctg = ctgs_up.mean()
        min_ctg = ctgs_up.min()
        max_ctg = ctgs_up.max()
        print(f"Generated {format(ctgs_up.shape[0], ',')} training instances, "
              f"Replay buffer size: {format(rb.size(), ',')}")
        print(f"Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))
        print(f"Times - {times_up.get_time_str()}")

        # train nnet
        print("Training model for update number %i for %i iterations" % (status.update_num, up_args.up_itrs))
        last_loss = train_nnet(nnet, rb, optimizer, device, up_args.up_itrs, status.itr, train_args)
        status.itr += up_args.up_itrs

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

        # test
        start_time = time.time()

        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(up_args.up_procs, curr_file, device, on_gpu, env, "V",
                                                                  all_zeros=False, clip_zero=False,
                                                                  batch_size=up_args.up_nnet_batch_size)

        max_solve_steps: int = min(status.update_num + 1, step_max)

        print("Testing greedy policy with %i states and %i steps" % (len(status.states_start_t), max_solve_steps))
        per_solved: float = greedy_test(status.states_start_t, status.goals_t, status.state_t_steps_l, env,
                                        heur_fn_qs, max_solve_steps=max_solve_steps)
        print("Greedy policy solved (best): %.2f%% (%.2f%%)" % (per_solved, status.per_solved_best))

        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)
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
