from typing import List, Tuple, Dict

from deepxube.base.updater import UpdaterHeur
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, print_pathfindperf
from deepxube.training.train_utils import ReplayBuffer, train_heur_nnet, TrainArgs
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils

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
    def __init__(self, step_max: int, balance_steps: bool):
        self.itr: int = 0
        self.update_num: int = 0
        self.step_max: int = step_max
        self.step_probs: NDArray
        if balance_steps:
            self.step_probs = np.zeros(self.step_max + 1)
            self.step_probs[0] = 0.5
            self.step_probs[1:] = 0.5/self.step_max
        else:
            self.step_probs = np.ones(self.step_max + 1)/(step_max + 1)

    def update_step_probs(self, step_to_search_perf: Dict[int, PathFindPerf]):
        per_solved_per_step_l: List[float] = []
        for step in range(self.step_max + 1):
            if step not in step_to_search_perf.keys():
                per_solved_per_step_l.append(0.0)
            else:
                search_perf: PathFindPerf = step_to_search_perf[step]
                per_solved_per_step_l.append(search_perf.per_solved())
        per_solved_per_step: NDArray = np.array(per_solved_per_step_l)

        num_no_soln: int = np.sum(per_solved_per_step == 0)
        if num_no_soln == 0:
            self.step_probs = per_solved_per_step / per_solved_per_step.sum()
        else:
            num_w_soln_eff: float = per_solved_per_step.sum() / 100.0
            num_tot_eff: float = num_w_soln_eff + 1
            self.step_probs = num_w_soln_eff * per_solved_per_step / per_solved_per_step.sum() / num_tot_eff
            self.step_probs[per_solved_per_step == 0] = 1 / num_tot_eff / num_no_soln


def load_data(model_dir: str, curr_file: str, targ_file: str, nnet: nn.Module,
              step_max: int, train_args: TrainArgs) -> Tuple[nn.Module, Status]:
    status_file: str = "%s/status.pkl" % model_dir
    if os.path.isfile(curr_file):
        nnet = nnet_utils.load_nnet(curr_file, nnet)
    else:
        torch.save(nnet.state_dict(), targ_file)

    status: Status
    if os.path.isfile(status_file):
        status = pickle.load(open("%s/status.pkl" % model_dir, "rb"))
        print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}")
    else:
        status = Status(step_max, train_args.balance_steps)
        # noinspection PyTypeChecker
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return nnet, status


def print_update_summary(step_to_search_perf: Dict[int, PathFindPerf], writer: SummaryWriter, status: Status):
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

    print_pathfindperf(step_to_search_perf)


def train(updater: UpdaterHeur, step_max: int, nnet_dir: str, train_args: TrainArgs, rb_past_up: int = 10,
          debug: bool = False):
    """ Train a deep neural network heuristic (DNN) function with deep approximate value iteration (DAVI).
    A target DNN is maintained for computing the updated heuristic values. When the greedy policy improves on a fixed
    test set, the target DNN is updated to be the current DNN. The number of steps taken for testing the greedy policy
    is the minimum between the number of target DNN updates and step_max.
    This makes the test a lot faster in the earlier stages, espeicially when step_max is large.

    For more information see:
    - Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and pathfinding."
    Nature Machine Intelligence 1.8 (2019): 356-363.
    - Bertsekas, D. P. & Tsitsiklis, J. N. Neuro-dynamic Programming (Athena Scientific, 1996).

    :param updater: an Updater object
    :param step_max: maximum number of steps to take to generate start/goal pairs
    :param nnet_dir: directory where DNN will be saved
    :param train_args: training arguments
    :param rb_past_up: amount of data generated from previous updates to keep in replay buffer. Total replay buffer size
    will then be train_args.batch_size * up_args.up_gen_itrs * rb_past_up
    :param debug: Turns off logging to make typing during breakpoints easier
    :return: None
    """
    # Initialization
    nnet: nn.Module = updater.heur_nnet.get_nnet()
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
    print(f"Update args: {updater.up_args}")
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # load nnet
    print("Loading nnet and status")
    nnet, status = load_data(nnet_dir, curr_file, targ_file, nnet, step_max, train_args)
    nnet.to(device)
    nnet = nn.DataParallel(nnet)

    # initialize replay buffer
    shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = updater.get_input_shapes_dtypes()
    rb_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes] + [tuple()]
    rb_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes] + [np.dtype(np.float64)]
    rb: ReplayBuffer = ReplayBuffer(train_args.batch_size * updater.up_args.up_gen_itrs * rb_past_up, rb_shapes,
                                    rb_dtypes)

    # training
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=train_args.lr)
    criterion = nn.MSELoss()
    while status.itr < train_args.max_itrs:
        # updater
        # start_time = time.time()
        if train_args.balance_steps:
            steps_show: List[int] = list(np.unique(np.linspace(0, status.step_max, 30, dtype=int)))
            step_prob_str: str = ', '.join([f'{step}:{status.step_probs[step]:.2E}' for step in steps_show])
            print(f"Step probs: {step_prob_str}")
        num_gen: int = train_args.batch_size * updater.up_args.up_gen_itrs
        all_zeros: bool = status.update_num == 0
        step_to_search_perf, ctgs_mean, ctgs_min, ctgs_max = updater.get_update_data(targ_file, all_zeros, step_max,
                                                                                     status.step_probs, num_gen, rb,
                                                                                     device, on_gpu)
        writer.add_scalar("ctgs (mean)", ctgs_mean, status.itr)
        writer.add_scalar("ctgs (min)", ctgs_min, status.itr)
        writer.add_scalar("ctgs (max)", ctgs_max, status.itr)

        print_update_summary(step_to_search_perf, writer, status)
        if train_args.balance_steps:
            status.update_step_probs(step_to_search_perf)

        # get batches
        print("Getting training batches")
        start_time = time.time()
        batches: List[Tuple[List[NDArray], NDArray]] = []
        for _ in range(updater.up_args.up_itrs):
            arrays_samp: List[NDArray] = rb.sample(train_args.batch_size)
            inputs_batch_np: List[NDArray] = arrays_samp[:-1]
            ctgs_batch_np: NDArray = np.expand_dims(arrays_samp[-1].astype(np.float32), 1)
            batches.append((inputs_batch_np, ctgs_batch_np))
        print(f"Time: {time.time() - start_time}")

        # train nnet
        print("Training model for updater number %i for %i iterations" % (status.update_num, len(batches)))
        last_loss = train_heur_nnet(nnet, batches, optimizer, criterion, device, status.itr, train_args)
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
