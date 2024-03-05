from typing import List, Tuple
from deepxube.utils import data_utils, nnet_utils, misc_utils
from deepxube.utils.timing_utils import Times
from deepxube.utils.nnet_utils import HeurFnQ
from deepxube.environments.environment_abstract import State, Environment, Goal
from deepxube.search_state.greedy_policy import Greedy, greedy_test, Instance

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
import time

import sys
import shutil


class Status:
    def __init__(self, env: Environment, step_max: int, num_test_per_step: int):
        self.itr: int = 0
        self.update_num: int = 0
        self.per_solved_best: float = 0.0

        self.state_t_steps_l: List[int] = []

        for step in range(step_max + 1):
            self.state_t_steps_l.extend([step] * num_test_per_step)

        self.states_start_t: List[State]
        self.goals_t: List[Goal]
        print(f"Generating {num_test_per_step} test states per step ({step_max} steps, "
              f"{format(len(self.state_t_steps_l), ',')} total test states)")
        self.states_start_t, self.goals_t = env.get_start_goal_pairs(self.state_t_steps_l)


def update_runner(batch_size: int, num_batches: int, step_max: int, heur_fn_q: HeurFnQ, env: Environment,
                  result_queue: Queue, solve_steps: int, eps_max: float):
    times: Times = Times()

    def remove_instance_fn(inst: Instance) -> bool:
        if inst.is_solved:
            return True
        if inst.step_num > solve_steps:
            return True
        return False

    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    greedy: Greedy = Greedy(env)
    for _ in range(num_batches):
        # remove instances
        greedy.remove_instances(remove_instance_fn)

        # add instances
        if len(greedy.instances) < batch_size:
            num_gen: int = batch_size - len(greedy.instances)
            gen_steps: np.array = np.random.choice(step_max + 1, size=num_gen)

            times_states: Times = Times()
            states_gen, goals_gen = env.get_start_goal_pairs(list(gen_steps), times=times_states)
            times.add_times(times_states, ["get_states"])

            eps_gen_l: List[float] = list(np.random.rand(len(states_gen)) * eps_max)
            greedy.add_instances(states_gen, goals_gen, eps_l=eps_gen_l)

        # take a step
        states, goals, ctgs_backup = greedy.step(heuristic_fn, times=times, rand_seen=True)

        # put in queue
        start_time = time.time()
        states_nnet = env.states_to_nnet_input(states)
        times.record_time("states_to_nnet", time.time() - start_time)

        start_time = time.time()
        goals_nnet = env.goals_to_nnet_input(goals)
        times.record_time("goals_to_nnet", time.time() - start_time)

        result_queue.put((states_nnet, goals_nnet, ctgs_backup))


def load_data(model_dir: str, nnet_file: str, env: Environment, num_test_per_step: int,
              step_max: int) -> Tuple[nn.Module, Status]:
    status_file: str = "%s/status.pkl" % model_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_v_nnet())
    else:
        nnet = env.get_v_nnet()

    if os.path.isfile(status_file):
        status: Status = pickle.load(open("%s/status.pkl" % model_dir, "rb"))
        print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}, "
              f"per_solved_best: {status.per_solved_best}")
    else:
        status: Status = Status(env, step_max, num_test_per_step)
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return nnet, status


def sample_list_np(data: List[np.ndarray], samp_idxs) -> List[np.ndarray]:
    list_len: int = len(data)
    data_samp: List[np.ndarray] = []
    for nnet_rep_idx in range(list_len):
        data_samp.append(data[nnet_rep_idx][samp_idxs].copy())

    return data_samp


class ReplayBuffer:
    def __init__(self, maxsize: int):
        self.states_nnet: List[np.ndarray] = []
        self.goals_nnet: List[np.ndarray] = []
        self.targets: np.array = np.zeros(0)

        self.curr_size: int = 0
        self.maxsize: int = maxsize
        self.empty_buff: bool = True

    def add(self, states_nnet: List[np.ndarray], goals_nnet: List[np.ndarray], targets: np.array):
        num_add: int = targets.shape[0]

        # initialize rb
        if self.empty_buff:
            samp_idxs: np.array = np.random.randint(0, num_add, size=self.maxsize)

            self.states_nnet = sample_list_np(states_nnet, samp_idxs)
            self.goals_nnet = sample_list_np(goals_nnet, samp_idxs)
            self.targets = targets[samp_idxs].copy()

            self.empty_buff = False

        # add data
        num_below_max: int = min(self.maxsize - self.curr_size, num_add)
        num_above_max: int = num_add - num_below_max

        buff_idxs = np.concatenate((np.arange(self.curr_size, self.curr_size + num_below_max),
                                    np.random.randint(0, self.maxsize, size=num_above_max)), axis=0)
        state_nnet_rep_len: int = len(states_nnet)
        for nnet_rep_idx in range(state_nnet_rep_len):
            self.states_nnet[nnet_rep_idx][buff_idxs] = states_nnet[nnet_rep_idx]

        goal_nnet_rep_len: int = len(goals_nnet)
        for nnet_rep_idx in range(goal_nnet_rep_len):
            self.goals_nnet[nnet_rep_idx][buff_idxs] = goals_nnet[nnet_rep_idx]

        self.targets[buff_idxs] = targets

        self.curr_size += num_add

    def sample(self, num: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.array]:
        samp_idxs: np.array = self._sample_valid_idxs(num)

        states_nnet = sample_list_np(self.states_nnet, samp_idxs)
        goals_nnet = sample_list_np(self.goals_nnet, samp_idxs)
        targets = self.targets[samp_idxs].copy()

        return states_nnet, goals_nnet, targets

    def _sample_valid_idxs(self, num: int) -> np.array:
        samp_idxs: np.array = np.random.randint(0, self.curr_size, size=num)

        return samp_idxs


def train_nnet(nnet: nn.Module, rb: ReplayBuffer, data_q: Queue, device: torch.device, batch_size: int, num_itrs: int,
               train_itr: int, lr: float, lr_d: float, on_gpu: bool, display_itrs: int) -> float:
    # optimization
    criterion = nn.MSELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    times = Times()

    # train network
    nnet.train()
    max_itrs: int = train_itr + num_itrs

    last_loss: float = np.inf
    while train_itr < max_itrs:
        # zero the parameter gradients
        start_time = time.time()
        nnet.train()
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("zero_grad", time.time() - start_time)

        # get data
        start_time = time.time()
        states_nnet_get, goals_nnet_get, ctgs_backup_get = data_q.get()
        rb.add(states_nnet_get, goals_nnet_get, ctgs_backup_get)
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("get_data", time.time() - start_time)

        # get training data
        start_time = time.time()
        states_batch_np, goals_batch_np, ctgs_batch_np = rb.sample(batch_size)
        ctgs_batch_np = np.expand_dims(ctgs_batch_np.astype(np.float32), 1)
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("rb_samp", time.time() - start_time)

        # send training data to device
        start_time = time.time()
        states_batch: List[Tensor] = nnet_utils.to_pytorch_input(states_batch_np, device)
        goals_batch: List[Tensor] = nnet_utils.to_pytorch_input(goals_batch_np, device)
        ctgs_batch: Tensor = torch.tensor(ctgs_batch_np, device=device)[:, 0]
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("to_gpu", time.time() - start_time)

        # forward
        start_time = time.time()
        ctgs_nnet: Tensor = nnet(states_batch, goals_batch)

        loss = criterion(ctgs_nnet[:, 0], ctgs_batch)
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("forward", time.time() - start_time)

        # backward
        start_time = time.time()
        loss.backward()
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("backward", time.time() - start_time)

        # step
        start_time = time.time()
        optimizer.step()
        if on_gpu:
            torch.cuda.synchronize()
        times.record_time("step", time.time() - start_time)

        last_loss = loss.item()
        # display progress
        if (display_itrs > 0) and (train_itr % display_itrs == 0):
            print("Itr: %i, lr: %.2E, loss: %.2E, targ_ctg: %.2f, nnet_ctg: %.2f" % (
                      train_itr, lr_itr, loss.item(), ctgs_batch.mean().item(),
                      ctgs_nnet.mean().item()))
            print(times.get_time_str())

            times.reset_times()
        train_itr = train_itr + 1

    return last_loss


def train(env: Environment, step_max: int, nnet_dir: str, num_test_per_step: int = 30, itrs_per_update: int = 5000,
          num_update_procs: int = 1, update_nnet_batch_size: int = 10000, greedy_update_step_max: int = 1,
          greedy_update_eps_max: int = 0.1, lr: float = 0.001, lr_d: float = 0.9999993, max_itrs: int = 1000000,
          batch_size: int = 1000, display: int = 100, debug: bool = False):
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
    :param num_test_per_step: Number of test states for each step between 0 and step_max
    :param itrs_per_update: How many iterations to do before checking if target network should be updated
    :param num_update_procs: Number of parallel workers used to compute updated cost-to-go values
    :param update_nnet_batch_size: Batch size of each nnet used for each process update. Make smaller if running out
    of memory.
    :param greedy_update_step_max: Maximum number of epsilon greedy policy steps (update_steps) to take from generated
    start states to generate additional data. Increasing this number could make
    the heuristic function more robust to depression regions. The number of steps taken for the update is the
    minimum between the number of target DNN updates and greedy_update_step_max to not bias towards states further
    away from goal.
    :param greedy_update_eps_max: epsilon greedy policy max. Each start/goal pair will have an eps uniformly distributed
    between 0 and greedy_update_eps_max
    :param lr: Initial learning rate
    :param lr_d: Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)
    :param max_itrs: Maximum number of iterations
    :param batch_size: Batch size
    :param display: Number of iterations to display progress. No display if 0.
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
        sys.stdout = data_utils.Logger(output_save_loc, "a")

    # Print basic info
    print("HOST: %s" % os.uname()[1])
    print("CPU: %i" % num_update_procs)
    print("Batch size: %i" % batch_size)
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
    states_per_update: int = itrs_per_update * batch_size
    rb: ReplayBuffer = ReplayBuffer(states_per_update)
    while status.itr < max_itrs:
        # greedy policy data generation
        all_zeros: bool = not os.path.isfile(targ_file)
        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(num_update_procs, targ_file,
                                                                  device, on_gpu, env.get_v_nnet(), env,
                                                                  all_zeros=all_zeros, clip_zero=True,
                                                                  batch_size=update_nnet_batch_size)

        solve_steps: int = int(min(status.update_num + 1, greedy_update_step_max))
        ctx = get_context("spawn")
        data_q: Queue = ctx.Queue()
        num_batches_per_proc: List[int] = misc_utils.split_evenly(itrs_per_update, num_update_procs)
        procs: List[BaseProcess] = []
        for proc_id, heur_fn_q in enumerate(heur_fn_qs):
            num_batches_proc: int = num_batches_per_proc[proc_id]
            if num_batches_proc == 0:
                continue
            # update_runner(batch_size, num_batches_proc, step_max, heur_fn_q, env, data_q, solve_steps,
            #              greedy_update_eps_max)
            proc = ctx.Process(target=update_runner, args=(batch_size, num_batches_proc, step_max, heur_fn_q, env,
                                                           data_q, solve_steps, greedy_update_eps_max))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        # train nnet
        print("Training model for update number %i for %i iterations" % (status.update_num, itrs_per_update))
        last_loss = train_nnet(nnet, rb, data_q, device, batch_size, itrs_per_update,
                               status.itr, lr, lr_d, on_gpu, display)
        status.itr += itrs_per_update

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

        # stop parallel processes
        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)
        for proc in procs:
            proc.join()

        # test
        start_time = time.time()

        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(num_update_procs, curr_file,
                                                                  device, on_gpu, env.get_v_nnet(), env,
                                                                  all_zeros=False, clip_zero=False,
                                                                  batch_size=update_nnet_batch_size)

        max_solve_steps: int = min(status.update_num + 1, step_max)

        print("Testing greedy policy with %i states and %i steps" % (len(status.states_start_t), max_solve_steps))
        per_solved: float = greedy_test(status.states_start_t, status.goals_t, status.state_t_steps_l, env,
                                        heur_fn_qs, max_solve_steps=max_solve_steps)
        print("Greedy policy solved (best): %.2f%% (%.2f%%)" % (per_solved, status.per_solved_best))

        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)
        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        if per_solved > status.per_solved_best:
            print("Updating target network")
            status.per_solved_best = per_solved
            update_nnet: bool = True
        else:
            update_nnet: bool = False

        print("Last loss was %f" % last_loss)
        if update_nnet:
            # Update nnet
            rb: ReplayBuffer = ReplayBuffer(states_per_update)
            shutil.copy(curr_file, targ_file)
            status.update_num = status.update_num + 1

        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
