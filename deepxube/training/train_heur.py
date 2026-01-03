from typing import Optional, List
from dataclasses import dataclass

from deepxube.base.domain import State, Goal
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ, HeurFnV, HeurFnQ
from deepxube.base.pathfinding import PathFind, Node
from deepxube.pathfinding.pathfinding_utils import PathFindPerf
from deepxube.pathfinding.q.bwqs import BWQSEnum, InstanceBWQS
from deepxube.pathfinding.v.bwas import BWAS, InstanceBWAS
from deepxube.base.updater import UpdateHeur
from deepxube.training.train_utils import TrainArgs
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
    search_weights: List[float]
    test_nnet_batch_size: int
    test_up_freq: int
    test_init: bool

    def __repr__(self) -> str:
        return (f"TestArgs(instances={len(self.test_states)}, search_itrs={self.search_itrs}, "
                f"search_weights={self.search_weights}, test_nnet_batch_size={self.test_nnet_batch_size}, "
                f"test_up_freq={self.test_up_freq}, test_init={self.test_init})")


def get_pathfind_w_instances(updater: UpdateHeur, train_heur: TrainHeur, test_args: TestArgs, param_idx: int) -> PathFind:
    heur_nnet: HeurNNetPar = updater.get_heur_nnet()
    if isinstance(heur_nnet, HeurNNetParV):
        heur_fn_v: HeurFnV = heur_nnet.get_nnet_fn(train_heur.nnet, test_args.test_nnet_batch_size, train_heur.device, None)
        pathfind_v: BWAS = BWAS(updater.domain)
        pathfind_v.set_heur_fn(heur_fn_v)
        instances_v: List[InstanceBWAS] = pathfind_v.make_instances(test_args.test_states, test_args.test_goals, batch_size=1,
                                                                    weight=test_args.search_weights[param_idx], eps=0.0)
        pathfind_v.add_instances(instances_v)
        return pathfind_v
    elif isinstance(heur_nnet, HeurNNetParQ):
        heur_fn_q: HeurFnQ = heur_nnet.get_nnet_fn(train_heur.nnet, test_args.test_nnet_batch_size,
                                                   train_heur.device, None)
        pathfind_q: BWQSEnum = BWQSEnum(updater.domain, heur_fn_q)
        root_nodes_q: List[Node] = pathfind_q.create_root_nodes(test_args.test_states, test_args.test_goals)
        instances_q: List[InstanceBWQS] = [InstanceBWQS(root_node, 1, test_args.search_weights[param_idx], 0.0, None)
                                           for root_node in root_nodes_q]
        pathfind_q.add_instances(instances_q)
        return pathfind_q

    else:
        raise ValueError(f"Unknown heuristic function type {heur_nnet}")


def train(updater: UpdateHeur, nnet_dir: str, train_args: TrainArgs, test_args: Optional[TestArgs] = None,
          debug: bool = False) -> None:
    """ Train a deep neural network heuristic (DNN) function with deep reinforcement learning.

    For more information see:
    - Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and pathfinding."
    Nature Machine Intelligence 1.8 (2019): 356-363.
    - Bertsekas, D. P. & Tsitsiklis, J. N. Neuro-dynamic Programming (Athena Scientific, 1996).

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
    updater.set_heur_file(heur_targ_file)
    print(updater.get_heur_nnet().get_nnet())
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

    to_main_q, from_main_qs = updater.start_procs()
    train_heur: TrainHeur = TrainHeur(updater, to_main_q, from_main_qs, heur_file, heur_targ_file, status_file, device,
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
            test(updater, train_heur, test_args, writer)

        # train
        train_heur.update_step()

        # clear cuda memory
        torch.cuda.empty_cache()

        up_itr_performed = True

    if (test_args is not None) and up_itr_performed:
        test(updater, train_heur, test_args, writer)

    updater.stop_procs()

    print("Done")


def test(updater: UpdateHeur, train_heur: TrainHeur, test_args: TestArgs, writer: SummaryWriter) -> None:
    print(f"Testing - itr: {train_heur.status.itr}, update_itr: {train_heur.status.update_num}, "
          f"targ_update: {train_heur.status.targ_update_num}, num_inst: {len(test_args.test_states)}, "
          f"num_search_params: {len(test_args.search_weights)}")
    for param_idx in range(len(test_args.search_weights)):
        start_time = time.time()
        # get pathfinding alg with test instances
        pathfind: PathFind = get_pathfind_w_instances(updater, train_heur, test_args, param_idx)

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
        w_val: float = test_args.search_weights[param_idx]
        test_info_l: List[str] = [f"%solved: {per_solved_ave:.2f}", f"path_costs: {path_cost_ave:.3f}",
                                  f"search_itrs: {search_itrs_ave:.3f}",
                                  f"test_time: {test_time:.2f}"]
        writer.add_scalar(f"val/w{w_val}/solved", per_solved_ave, train_heur.status.itr)
        writer.add_scalar(f"val/w{w_val}/path_cost", path_cost_ave, train_heur.status.itr)
        writer.add_scalar(f"val/w{w_val}/search_itrs", search_itrs_ave, train_heur.status.itr)
        print(f"Test w{w_val} - {', '.join(test_info_l)}")
