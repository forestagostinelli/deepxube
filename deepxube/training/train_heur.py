from typing import List, Dict

from deepxube.base.updater import UpdateHeur
from deepxube.pathfinding.pathfinding_utils import PathFindPerf
from deepxube.training.train_utils import TrainArgs
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.training.trainers import TrainHeur

import torch
from torch.utils.tensorboard import SummaryWriter

import os
import pickle

import numpy as np
from numpy.typing import NDArray

import sys


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

    def update_step_probs(self, step_to_search_perf: Dict[int, PathFindPerf]) -> None:
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


def load_data(model_dir: str, step_max: int, train_args: TrainArgs) -> Status:
    status_file: str = "%s/status.pkl" % model_dir
    status: Status
    if os.path.isfile(status_file):
        status = pickle.load(open("%s/status.pkl" % model_dir, "rb"))
        print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}")
    else:
        status = Status(step_max, train_args.balance_steps)
        # noinspection PyTypeChecker
        pickle.dump(status, open(status_file, "wb"), protocol=-1)

    return status


def train(updater: UpdateHeur, step_max: int, nnet_dir: str, train_args: TrainArgs, rb_past_up: int = 1,
          debug: bool = False) -> None:
    """ Train a deep neural network heuristic (DNN) function with deep reinforcement learning.

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
    heur_file = f"{nnet_dir}/heur.pt"
    output_save_loc = "%s/output.txt" % nnet_dir
    writer: SummaryWriter = SummaryWriter(nnet_dir)

    if not os.path.exists(nnet_dir):
        os.makedirs(nnet_dir)

    if not debug:
        sys.stdout = data_utils.Logger(output_save_loc, "a")

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

    print("Loading nnet and status")
    train_heur: TrainHeur = TrainHeur(updater, heur_file, device, on_gpu, writer, train_args, rb_past_up)
    status: Status = load_data(nnet_dir, step_max, train_args)

    # training
    while status.itr < train_args.max_itrs:
        # step probs
        if train_args.balance_steps:
            steps_show: List[int] = list(np.unique(np.linspace(0, status.step_max, 30, dtype=int)))
            step_prob_str: str = ', '.join([f'{step}:{status.step_probs[step]:.2E}' for step in steps_show])
            print(f"Step probs: {step_prob_str}")

        # train
        step_to_search_perf: Dict[int, PathFindPerf] = train_heur.update_step(step_max, status.step_probs.tolist(),
                                                                              status.itr)
        if train_args.balance_steps:
            status.update_step_probs(step_to_search_perf)

        status.itr += updater.up_args.up_itrs

        # save nnet
        train_heur.save_nnet()

        # clear cuda memory
        torch.cuda.empty_cache()

        # Update nnet
        status.update_num = status.update_num + 1

        # noinspection PyTypeChecker
        pickle.dump(status, open("%s/status.pkl" % nnet_dir, "wb"), protocol=-1)

    print("Done")
