from deepxube.base.updater import UpdateHeur
from deepxube.training.train_utils import TrainArgs
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.training.trainers import TrainHeur

import torch
from torch.utils.tensorboard import SummaryWriter

import os

import sys


def train(updater: UpdateHeur, step_max: int, nnet_dir: str, train_args: TrainArgs,
          rb_past_up: int = 1, debug: bool = False) -> None:
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
    heur_targ_file = f"{nnet_dir}/heur_targ.pt"
    status_file: str = f"{nnet_dir}/status.pkl"
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

    train_heur: TrainHeur = TrainHeur(updater, step_max, heur_file, heur_targ_file, status_file, device, on_gpu, writer,
                                      train_args, rb_past_up)

    # training
    while train_heur.status.itr < train_args.max_itrs:
        # train
        train_heur.update_step()

        # clear cuda memory
        torch.cuda.empty_cache()

    print("Done")
