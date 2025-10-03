from typing import List, Tuple, Dict

from deepxube.search.search_utils import SearchPerf, print_search_perf
from deepxube.training.train_utils import ReplayBuffer, train_heur, TrainArgs
from deepxube.update.updater import UpdateArgs, get_update_data
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.environments.environment_abstract import Environment

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import pickle

import numpy as np
from numpy.typing import NDArray
import time

import sys
import shutil


class Status:
    def __init__(self, step_max: int):
        self.itr: int = 0
        self.update_num: int = 0
        self.step_max: int = step_max
        # self.step_probs: NDArray = np.zeros(self.step_max + 1)
        # self.step_probs[0] = 0.5
        # self.step_probs[1:] = 0.5/self.step_max
        self.step_probs: NDArray = np.ones(self.step_max + 1)/(step_max + 1)

    def update_step_probs(self, step_to_search_perf: Dict[int, SearchPerf]):
        per_solved_per_step_l: List[float] = []
        for step in range(self.step_max + 1):
            if step not in step_to_search_perf.keys():
                per_solved_per_step_l.append(0.0)
            else:
                search_perf: SearchPerf = step_to_search_perf[step]
                per_solved_per_step_l.append(search_perf.per_solved())
        per_solved_per_step: NDArray = np.array(per_solved_per_step_l)

        num_no_soln: int = np.sum(per_solved_per_step == 0)
        if num_no_soln == 0:
            self.step_probs: NDArray = per_solved_per_step / per_solved_per_step.sum()
        else:
            num_w_soln_eff: float = per_solved_per_step.sum() / 100.0
            num_tot_eff: float = num_w_soln_eff + 1
            self.step_probs: NDArray = num_w_soln_eff * per_solved_per_step / per_solved_per_step.sum() / num_tot_eff
            self.step_probs[per_solved_per_step == 0] = 1 / num_tot_eff / num_no_soln


def load_data(model_dir: str, nnet_file: str, env: Environment, step_max: int) -> Tuple[nn.Module, Status]:
    status_file: str = "%s/status.pkl" % model_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_v_nnet())
    else:
        nnet = env.get_v_nnet()

    status: Status
    if os.path.isfile(status_file):
        status = pickle.load(open("%s/status.pkl" % model_dir, "rb"))
        print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}")
    else:
        status = Status(step_max)
        # noinspection PyTypeChecker
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return nnet, status


def print_update_summary(step_to_search_perf: Dict[int, SearchPerf], writer: SummaryWriter, status: Status):
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

    print_search_perf(step_to_search_perf)


def train(env: Environment, step_max: int, nnet_dir: str, train_args: TrainArgs, up_args: UpdateArgs,
          rb_past_up: int = 10, debug: bool = False):
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
    :param rb_past_up: amount of data generated from previous updates to keep in replay buffer. Total replay buffer size
    will then be train_args.batch_size * up_args.up_gen_itrs * rb_past_up
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
    nnet, status = load_data(nnet_dir, curr_file, env, step_max)
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
    rb: ReplayBuffer = ReplayBuffer(train_args.batch_size * up_args.up_gen_itrs * rb_past_up, rb_shapes, rb_dtypes)

    # training
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=train_args.lr)
    criterion = nn.MSELoss()
    while status.itr < train_args.max_itrs:
        # update
        # start_time = time.time()
        # steps_show: List[int] = list(np.unique(np.linspace(0, status.step_max, 30, dtype=int)))
        # step_prob_str: str = ', '.join([f'{step}:{status.step_probs[step]:.2E}' for step in steps_show])
        # print(f"Step probs: {step_prob_str}")
        num_gen: int = train_args.batch_size * up_args.up_gen_itrs
        step_to_search_perf: Dict[int, SearchPerf] = get_update_data(env, step_max, status.step_probs, num_gen, up_args,
                                                                     rb, targ_file, device, on_gpu)
        print_update_summary(step_to_search_perf, writer, status)
        # status.update_step_probs(step_to_search_perf)

        # get batches
        print("Getting training batches")
        start_time = time.time()
        batches: List[Tuple[List[NDArray], NDArray]] = []
        for _ in range(up_args.up_itrs):
            arrays_samp: List[NDArray] = rb.sample(train_args.batch_size)
            inputs_batch_np: List[NDArray] = arrays_samp[:-1]
            ctgs_batch_np: NDArray = np.expand_dims(arrays_samp[-1].astype(np.float32), 1)
            batches.append((inputs_batch_np, ctgs_batch_np))
        print(f"Time: {time.time() - start_time}")

        # train nnet
        print("Training model for update number %i for %i iterations" % (status.update_num, len(batches)))
        last_loss = train_heur(nnet, batches, optimizer, criterion, device, status.itr, train_args)
        print("Last loss was %f" % last_loss)
        status.itr += len(batches)

        # save nnet
        torch.save(nnet.state_dict(), curr_file)

        # clear cuda memory
        torch.cuda.empty_cache()

        # Update nnet
        shutil.copy(curr_file, targ_file)
        status.update_num = status.update_num + 1

        # noinspection PyTypeChecker
        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
