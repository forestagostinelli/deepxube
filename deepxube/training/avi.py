from typing import List, Tuple, Any
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.nnet.nnet_utils import HeurFnQ
from deepxube.environments.environment_abstract import State, Environment, Goal
from deepxube.updaters.updater import Updater
from deepxube.search.greedy_policy import greedy_test

import torch
from torch import Tensor
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn

import os
import pickle

import numpy as np
from numpy.typing import NDArray
import time

import sys
import shutil
import random


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
        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(1, "", torch.device("cpu"), False, env.get_v_nnet(),
                                                                  env, all_zeros=True)
        per_solved: float = greedy_test(self.states_start_t, self.goals_t, self.state_t_steps_l, env,
                                        heur_fn_qs, max_solve_steps=1)
        print("Greedy policy solved: %.2f%%" % per_solved)
        self.per_solved_best: float = per_solved


def do_update(step_max: int, update_num: int, env: Environment, step_update_max: int, num_states: int,
              eps_max: float, heur_fn_qs: List[HeurFnQ],
              update_batch_size: int) -> Tuple[List[NDArray[Any]], NDArray[np.float64]]:
    update_steps: int = int(min(update_num + 1, step_update_max))
    # num_states: int = int(np.ceil(num_states / update_steps))

    if update_num == 0:
        eps_max = 1.0

    # Do updates
    output_time_start = time.time()

    print(f"Updating cost-to-go with value iteration. Generating {format(num_states, ',')} training states.")
    step_probs: List[float] = list(np.ones(1 + step_max) / (step_max + 1))
    if step_update_max > 1:
        print("Using greedy policy with %i step(s)" % update_steps)

    updater: Updater = Updater(env, num_states, step_max, step_probs, heur_fn_qs, update_steps,
                               "greedy", update_batch_size=update_batch_size, eps_max=eps_max)

    nnet_rep: List[NDArray[Any]]
    ctgs: NDArray[np.float64]
    nnet_rep, ctgs, is_solved = updater.update()

    # Print stats
    # per_solved = 100.0 * np.mean(is_solved)
    print("Produced %s states, (%.2f seconds)" % (format(ctgs.shape[0], ","), time.time() - output_time_start))

    mean_ctg = ctgs[:, 0].mean()
    min_ctg = ctgs[:, 0].min()
    max_ctg = ctgs[:, 0].max()
    print("Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))

    return nnet_rep, ctgs


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
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return nnet, status


def make_batches(nnet_rep: List[NDArray[Any]], ctgs: NDArray[np.float64],
                 batch_size: int) -> List[Tuple[List[NDArray[Any]], NDArray[np.float64]]]:
    num_examples = ctgs.shape[0]
    rand_idxs = np.random.choice(num_examples, num_examples, replace=False)
    ctgs = ctgs.astype(np.float32)

    start_idx = 0
    batches = []
    while (start_idx + batch_size) <= num_examples:
        end_idx = start_idx + batch_size

        idxs = rand_idxs[start_idx:end_idx]

        inputs_batch = [x[idxs] for x in nnet_rep]
        ctgs_batch = ctgs[idxs]

        batches.append((inputs_batch, ctgs_batch))

        start_idx = end_idx

    return batches


def train_nnet(nnet: nn.Module, nnet_rep: List[NDArray[Any]], ctgs: NDArray[np.float64], device: torch.device,
               batch_size: int, num_itrs: int, train_itr: int, lr: float, lr_d: float, display_itrs: int) -> float:
    # optimization
    criterion = nn.MSELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    start_time = time.time()

    # train network
    batches = make_batches(nnet_rep, ctgs, batch_size)

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    last_loss: float = np.inf
    batch_idx: int = 0
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        inputs_batch_np, ctgs_batch_np = batches[batch_idx]
        ctgs_batch_np = ctgs_batch_np.astype(np.float32)

        # send data to device
        inputs_batch: List[Tensor] = nnet_utils.to_pytorch_input(inputs_batch_np, device)
        ctgs_batch: Tensor = torch.tensor(ctgs_batch_np, device=device)[:, 0]

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
        if (display_itrs > 0) and (train_itr % display_itrs == 0):
            print("Itr: %i, lr: %.2E, loss: %.2E, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), ctgs_batch.mean().item(),
                      ctgs_nnet.mean().item(), time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

        batch_idx += 1
        if batch_idx >= len(batches):
            random.shuffle(batches)
            batch_idx = 0

    return last_loss


def train(env: Environment, step_max: int, nnet_dir: str, num_test_per_step: int = 30,
          itrs_per_update: int = 5000, epochs_per_update: int = 1, num_update_procs: int = 1,
          update_batch_size: int = 10000, update_nnet_batch_size: int = 10000, greedy_update_step_max: int = 1,
          greedy_update_eps_max: float = 0.1, lr: float = 0.001, lr_d: float = 0.9999993, max_itrs: int = 1000000,
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
    :param epochs_per_update: How many epochs for which train. Making this greater than 1 could increase risk of
    overfitting, however, one can train for more iterations without having to generate more data.)
    :param num_update_procs: Number of parallel workers used to compute updated cost-to-go values
    :param update_batch_size: Maximum number of start/goal pairs when computing updated cost-to-go that each
    parallel worker generates at a time. Since multiprocessing is used, memory issues can happen if trying to send
    objects that are too large across processes. Therefore, if you update step freezes for no apparent reason,
    it could be a memory issue. Lowering this value or making the DNN representation more memory efficient could solve
    it.
    :param update_nnet_batch_size: Batch size of each nnet used for each process update. Make smaller if running out
    of memory.
    :param greedy_update_step_max: Maximum number of epsilon greedy policy steps (update_steps) to take from generated
    start states to generate additional data. Value of 1 is the same as basic DAVI. Increasing this number could make
    the heuristic function more robust to depression regions. The number of steps taken for the update is the
    minimum between the number of target DNN updates and greedy_update_step_max. Number of start/goal pairs is
    states_per_update/update_steps.
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
        sys.stdout = data_utils.Logger(output_save_loc, "a")  # type: ignore

    # Print basic info
    # print("HOST: %s" % os.uname()[1])
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
    while status.itr < max_itrs:
        # update
        all_zeros: bool = not os.path.isfile(targ_file)
        heur_fn_qs, heur_procs = nnet_utils.start_heur_fn_runners(num_update_procs, targ_file,
                                                                  device, on_gpu, env.get_v_nnet(), env,
                                                                  all_zeros=all_zeros, clip_zero=True,
                                                                  batch_size=update_nnet_batch_size)

        states_per_update: int = itrs_per_update * batch_size
        num_update_states: int = int(states_per_update)

        nnet_rep, ctgs = do_update(step_max, status.update_num, env, greedy_update_step_max, num_update_states,
                                   greedy_update_eps_max, heur_fn_qs, update_batch_size)

        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_qs)

        # train nnet
        num_train_itrs: int = epochs_per_update * np.ceil(ctgs.shape[0] / batch_size)
        print("Training model for update number %i for %i iterations" % (status.update_num, num_train_itrs))
        last_loss = train_nnet(nnet, nnet_rep, ctgs, device, batch_size, num_train_itrs,
                               status.itr, lr, lr_d, display)
        status.itr += num_train_itrs

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

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
            shutil.copy(curr_file, targ_file)
            status.update_num = status.update_num + 1

        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
