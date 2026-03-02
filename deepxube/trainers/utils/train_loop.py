from typing import Optional, List
from dataclasses import dataclass

from deepxube.base.domain import Domain, State, Goal
from deepxube.base.heuristic import HeurNNet, PolicyNNet, HeurNNetPar, PolicyNNetPar, HeurFn, PolicyFn
from deepxube.base.pathfinding import PathFind, Instance, PathFindHasHeur, PathFindHasPolicy
from deepxube.base.updater import UpdateHasHeur, UpdateHasPolicy, UpdateHeur, UpdatePolicy
from deepxube.base.trainer import TrainArgs
from deepxube.pathfinding.utils.performance import PathFindPerf
from deepxube.trainers.train_heur import TrainHeur
from deepxube.trainers.train_policy import TrainPolicy
from deepxube.utils.command_line_utils import get_pathfind_from_arg
from deepxube.nnet import nnet_utils
from deepxube.utils import data_utils

from torch.utils.tensorboard import SummaryWriter

import sys
import os

import torch
import time


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


def get_curr_itr(train_heur: Optional[TrainHeur], train_policy: Optional[TrainPolicy]) -> int:
    if train_heur is not None:
        return train_heur.status.itr
    else:
        assert train_policy is not None
        return train_policy.status.itr


def get_curr_update_num(train_heur: Optional[TrainHeur], train_policy: Optional[TrainPolicy]) -> int:
    if train_policy is not None:
        return train_policy.status.update_num
    else:
        assert train_heur is not None
        return train_heur.status.update_num


def train(domain: Domain, heur_nnet_par: Optional[HeurNNetPar], update_heur: Optional[UpdateHeur], policy_nnet_par: Optional[PolicyNNetPar],
          update_policy: Optional[UpdatePolicy], nnet_dir: str, train_args: TrainArgs, test_args: Optional[TestArgs] = None,
          debug: bool = False) -> None:
    if not os.path.exists(nnet_dir):
        os.makedirs(nnet_dir)

    # logging
    if not debug:
        output_save_loc = f"{nnet_dir}/output.txt"
        sys.stdout = data_utils.Logger(output_save_loc, "a")

    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    writer: SummaryWriter = SummaryWriter(nnet_dir)

    heur_file: str = f"{nnet_dir}/heur.pt"
    heur_targ_file: str = f"{nnet_dir}/heur_targ.pt"
    heur_status_file: str = f"{nnet_dir}/heur_status.pkl"

    policy_file = f"{nnet_dir}/policy.pt"
    policy_targ_file = f"{nnet_dir}/policy_targ.pt"
    policy_status_file: str = f"{nnet_dir}/policy_status.pkl"

    train_heur: Optional[TrainHeur] = None
    train_policy: Optional[TrainPolicy] = None
    if heur_nnet_par is not None:
        assert update_heur is not None

        for updater in [update_heur, update_policy]:
            if (updater is not None) and isinstance(updater, UpdateHasHeur):
                updater.set_heur_nnet(heur_nnet_par)
                updater.set_heur_file(heur_targ_file)

        heur_nnet: HeurNNet = heur_nnet_par.get_nnet()
        print("Update Heur:")
        print(heur_nnet)
        print(update_heur.get_pathfind())
        print(f"{update_heur.get_up_args_repr()}")
        to_main_q, from_main_qs = update_heur.start_procs(train_args.rb * train_args.batch_size * update_heur.up_args.up_itrs)
        train_heur = TrainHeur(heur_nnet, update_heur, to_main_q, from_main_qs, heur_file, heur_targ_file, heur_status_file, device, on_gpu, writer, train_args)
    if policy_nnet_par is not None:
        assert update_policy is not None

        for updater in [update_heur, update_policy]:
            if (updater is not None) and isinstance(updater, UpdateHasPolicy):
                updater.set_policy_nnet(policy_nnet_par)
                updater.set_policy_file(policy_targ_file)

        policy_nnet: PolicyNNet = policy_nnet_par.get_nnet()
        print("Update Policy:")
        print(policy_nnet)
        print(update_policy.get_pathfind())
        print(f"{update_policy.get_up_args_repr()}")
        to_main_q, from_main_qs = update_policy.start_procs(train_args.rb * train_args.batch_size * update_policy.up_args.up_itrs)
        train_policy = TrainPolicy(policy_nnet, update_policy, to_main_q, from_main_qs, policy_file, policy_targ_file, policy_status_file, device, on_gpu,
                                   writer, train_args)

    print(f"{train_args}")
    print(domain)
    if test_args is not None:
        print(f"{test_args}")

    # training
    up_itr_performed: bool = False

    curr_itr: int = get_curr_itr(train_heur, train_policy)
    while curr_itr < train_args.max_itrs:
        # test
        do_test: bool = False
        if test_args is not None:
            update_num: int = get_curr_update_num(train_heur, train_policy)
            if update_num > 0:
                do_test = update_num % test_args.test_up_freq == 0
            elif update_num == 0:
                do_test = test_args.test_init

        if do_test:
            assert test_args is not None
            test(domain, heur_nnet_par, train_heur, policy_nnet_par, train_policy, test_args, writer, curr_itr)

        # train
        for train_obj in [train_heur, train_policy]:
            if (train_obj is None) or (train_obj.status.itr > curr_itr):
                continue
            train_obj.update_step()
            torch.cuda.empty_cache()

        up_itr_performed = True
        curr_itr = get_curr_itr(train_heur, train_policy)

    if (test_args is not None) and up_itr_performed:
        test(domain, heur_nnet_par, train_heur, policy_nnet_par, train_policy, test_args, writer, curr_itr)

    # stop procs
    for updater in [update_heur, update_policy]:
        if updater is not None:
            updater.stop_procs()

    print("Done")


def get_pathfind_w_instances(domain: Domain, heur_nnet_par: Optional[HeurNNetPar], train_heur: Optional[TrainHeur], policy_nnet_par: Optional[PolicyNNetPar],
                             train_policy: Optional[TrainPolicy], test_args: TestArgs, pathfind_arg: str) -> PathFind:
    pathfind: PathFind = get_pathfind_from_arg(domain, pathfind_arg)[0]
    if heur_nnet_par is not None:
        assert isinstance(pathfind, PathFindHasHeur)
        assert train_heur is not None
        heur_fn: HeurFn = heur_nnet_par.get_nnet_fn(train_heur.nnet, test_args.test_nnet_batch_size, train_heur.device, None)
        pathfind.set_heur_fn(heur_fn)
    if policy_nnet_par is not None:
        assert isinstance(pathfind, PathFindHasPolicy)
        assert train_policy is not None
        policy_fn: PolicyFn = policy_nnet_par.get_nnet_fn(train_policy.nnet, test_args.test_nnet_batch_size, train_policy.device, None)
        pathfind.set_policy_fn(policy_fn)

    instances: List[Instance] = pathfind.make_instances(test_args.test_states, test_args.test_goals, None, True)
    pathfind.add_instances(instances)

    return pathfind


def test(domain: Domain, heur_nnet_par: Optional[HeurNNetPar], train_heur: Optional[TrainHeur], policy_nnet_par: Optional[PolicyNNetPar],
         train_policy: Optional[TrainPolicy], test_args: TestArgs, writer: SummaryWriter, curr_itr: int) -> None:
    print(f"Testing - itr: num_inst: {len(test_args.test_states)}, num_pathfinds: {len(test_args.pathfinds)}")
    for pathfind_idx in range(len(test_args.pathfinds)):
        start_time = time.time()
        # get pathfinding alg with test instances
        pathfind_arg: str = test_args.pathfinds[pathfind_idx]
        pathfind: PathFind = get_pathfind_w_instances(domain, heur_nnet_par, train_heur, policy_nnet_par, train_policy, test_args, pathfind_arg)

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
        writer.add_scalar(f"val/{pathfind_arg}/solved", per_solved_ave, curr_itr)
        writer.add_scalar(f"val/{pathfind_arg}/path_cost", path_cost_ave, curr_itr)
        writer.add_scalar(f"val/{pathfind_arg}/search_itrs", search_itrs_ave, curr_itr)
        print(f"Test {pathfind_arg} - {', '.join(test_info_l)}")
