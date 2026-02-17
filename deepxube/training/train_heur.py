from typing import Optional, List
from dataclasses import dataclass

from deepxube.base.domain import State, Goal
from deepxube.base.heuristic import HeurNNet, HeurNNetPar, HeurFn
from deepxube.base.pathfinding import PathFind, Instance, PathFindHeur
from deepxube.pathfinding.utils.performance import PathFindPerf
from deepxube.base.updater import UpdateHeur
from deepxube.training.train_utils import TrainArgs
from deepxube.utils.command_line_utils import get_pathfind_from_arg
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.training.trainers import TrainHeur

import torch
from torch.utils.tensorboard import SummaryWriter
import time

import os

import sys


@dataclass
class TestArgs:
    test_states: List[State]
    test_goals: List[Goal]
    search_itrs: int
    pathfinds: List[str]
    test_nnet_batch_size: int
    test_up_freq: int
    test_init: bool

    def __repr__(self) -> str:
        return (f"TestArgs(instances={len(self.test_states)}, search_itrs={self.search_itrs}, "
                f"pathfinds={self.pathfinds}, test_nnet_batch_size={self.test_nnet_batch_size}, "
                f"test_up_freq={self.test_up_freq}, test_init={self.test_init})")


def get_pathfind_w_instances(heur_nnet_par: HeurNNetPar, heur_type: str, updater: UpdateHeur, train_heur: TrainHeur, test_args: TestArgs,
                             pathfind_arg: str) -> PathFind:
    pathfind: PathFind = get_pathfind_from_arg(updater.domain, heur_type, pathfind_arg)[0]
    assert isinstance(pathfind, PathFindHeur)

    heur_fn: HeurFn = heur_nnet_par.get_nnet_fn(train_heur.nnet, test_args.test_nnet_batch_size, train_heur.device, None)
    pathfind.set_heur_fn(heur_fn)

    instances: List[Instance] = pathfind.make_instances(test_args.test_states, test_args.test_goals, None, True)
    pathfind.add_instances(instances)

    return pathfind


def train(heur_nnet_par: HeurNNetPar, heur_type: str, updater: UpdateHeur, nnet_dir: str, train_args: TrainArgs, test_args: Optional[TestArgs] = None,
          debug: bool = False) -> None:
    """ Train a deep neural network heuristic (DNN) function with deep reinforcement learning.

    For more information see:
    - Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and pathfinding."
    Nature Machine Intelligence 1.8 (2019): 356-363.
    - Bertsekas, D. P. & Tsitsiklis, J. N. Neuro-dynamic Programming (Athena Scientific, 1996).

    :param heur_nnet_par: heur_nnet_par object to be used with updater
    :param heur_type: heuristic function type
    :param updater: an Updater object
    :param nnet_dir: directory where DNN will be saved
    :param train_args: training arguments
    :param test_args: test arguments
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
    updater.set_heur_nnet(heur_nnet_par)
    updater.set_heur_file(heur_targ_file)

    heur_nnet: HeurNNet = heur_nnet_par.get_nnet()
    print(heur_nnet)
    print(updater.domain)
    print(updater.get_pathfind())
    print(f"{train_args}")
    print(f"{updater.get_up_args_repr()}")
    if test_args is not None:
        print(f"{test_args}")
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    to_main_q, from_main_qs = updater.start_procs(train_args.rb * train_args.batch_size * updater.up_args.up_itrs)
    train_heur: TrainHeur = TrainHeur(heur_nnet, updater, to_main_q, from_main_qs, heur_file, heur_targ_file, status_file, device,
                                      on_gpu, writer, train_args)

    # training
    up_itr_performed: bool = False
    while train_heur.status.itr < train_args.max_itrs:
        # test
        do_test: bool = False
        if test_args is not None:
            if train_heur.status.update_num > 0:
                do_test = train_heur.status.update_num % test_args.test_up_freq == 0
            elif train_heur.status.update_num == 0:
                do_test = test_args.test_init

        if do_test:
            assert test_args is not None
            test(heur_nnet_par, heur_type, updater, train_heur, test_args, writer)

        # train
        train_heur.update_step()

        # clear cuda memory
        torch.cuda.empty_cache()

        up_itr_performed = True

    if (test_args is not None) and up_itr_performed:
        test(heur_nnet_par, heur_type, updater, train_heur, test_args, writer)

    updater.stop_procs()

    print("Done")


def test(heur_nnet_par: HeurNNetPar, heur_type: str, updater: UpdateHeur, train_heur: TrainHeur, test_args: TestArgs, writer: SummaryWriter) -> None:
    print(f"Testing - itr: {train_heur.status.itr}, update_itr: {train_heur.status.update_num}, "
          f"targ_update: {train_heur.status.targ_update_num}, num_inst: {len(test_args.test_states)}, "
          f"num_pathfinds: {len(test_args.pathfinds)}")
    for pathfind_idx in range(len(test_args.pathfinds)):
        start_time = time.time()
        # get pathfinding alg with test instances
        pathfind_arg: str = test_args.pathfinds[pathfind_idx]
        pathfind: PathFind = get_pathfind_w_instances(heur_nnet_par, heur_type, updater, train_heur, test_args, pathfind_arg)

        # attempt to solve
        for _ in range(test_args.search_itrs):
            pathfind.step()

        # get performacne
        pathfind_perf: PathFindPerf = PathFindPerf()
        for instance in pathfind.instances:
            pathfind_perf.update_perf(instance)
        test_time = time.time() - start_time

        # log
        per_solved_ave, path_cost_ave, search_itrs_ave = pathfind_perf.stats()
        test_info_l: List[str] = [f"%solved: {per_solved_ave:.2f}", f"path_costs: {path_cost_ave:.3f}",
                                  f"search_itrs: {search_itrs_ave:.3f}",
                                  f"test_time: {test_time:.2f}"]
        writer.add_scalar(f"val/{pathfind_arg}/solved", per_solved_ave, train_heur.status.itr)
        writer.add_scalar(f"val/{pathfind_arg}/path_cost", path_cost_ave, train_heur.status.itr)
        writer.add_scalar(f"val/{pathfind_arg}/search_itrs", search_itrs_ave, train_heur.status.itr)
        print(f"Test {pathfind_arg} - {', '.join(test_info_l)}")
