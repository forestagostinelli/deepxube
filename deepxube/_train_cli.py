from typing import Dict
import argparse
from argparse import ArgumentParser

import torch

from deepxube.nnet.nnet_utils import NNetParRunner
from deepxube.base.nnet_par_fn import HeurNNetPar, HeurVNNetPar, HeurQNNetPar, PolicyNNetPar, HeurVNNetParRunner, HeurQNNetParRunner, PolicyNNetParRunner
from deepxube.base.updater import Update
from deepxube.base.trainer import Train, TrainArgs
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.factories.updater_factory import get_updater_from_args
from deepxube.nnet import nnet_utils
from deepxube.factories.nnet_par_fn_factory import get_heur_nnet_par_from_arg, get_policy_nnet_par_from_arg
from deepxube.utils.data_utils import Logger
from deepxube.trainers.train_heur import TrainHeurV
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os
import sys


def parser_train(parser: ArgumentParser) -> None:
    # domain
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    # nnets and corresponding functions
    parser.add_argument('--heur', type=str, default=None, help="Heuristic neural network and arguments.")
    parser.add_argument('--heur_type', type=str, default=None, help="V, QFix, QIn. V maps state/goal tuples to cost-to-go. "
                                                                    "QFix maps state/goal tuples to q_values for a fixed action space. "
                                                                    "QIn maps state/goal/action tuples to q_value (can be used in arbitrary action spaces).")

    parser.add_argument('--policy', type=str, default=None, help="Policy neural network and arguments.")
    parser.add_argument('--policy_samp', type=int, default=10, help="Number to actions to sample from policy")

    # pathfinding
    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments. Batch size of any pathfinding algorithm should be 1 "
                                                                    "since updater assumes 1 instance is generated per iteration.")

    # updater
    parser.add_argument('--up', type=str, required=True, help="Updater algorithm and arguments.")

    # train args
    parser.add_argument('--dir', type=str, required=True, help="Directory to save neural networks.")

    train_group = parser.add_argument_group('train')
    train_group.add_argument('--batch_size', type=int, default=1000, help="Batch size.")
    train_group.add_argument('--up_itrs', type=int, default=100, help="Number of iterations to check for update.")
    train_group.add_argument('--up_gen_itrs', type=int, default=None, help="Number of iterations for which to generate training data per update check. "
                                                                           "If None then defaults to up_itrs.")
    train_group.add_argument('--max_itrs', type=int, default=100000, help="Maximum training iterations.")
    train_group.add_argument('--accum', type=int, default=1, help="Number of gradient accumulation steps to use to split batch. This argument does not change "
                                                                  "the given batch size, only the number of accumulation steps used to do the forward pass.")
    train_group.add_argument('--chkpt', type=int, default=0, help="Save checkpoint file of network being trained at initialization and at every given "
                                                                  "number of update checks. Checkpoint number given is training iteration, not update number."
                                                                  "If 0 then checkpointing is not done.")
    train_group.add_argument('--display', type=int, default=100, help="Display frequency for nnet training info. 0 for no display.")
    train_group.add_argument('--bal', action='store_true', default=False, help="Set to balance of number of steps to take to generate problem instances based "
                                                                               "on percentage of states solved.")
    train_group.add_argument('--rb', type=int, default=0, help="Number of updates worth of data to keep in replay buffer. If 0 then no replay buffer is used "
                                                               "and training waits for update to finish to get data and randomly sample from that data. "
                                                               "No replay buffer results in faster updates due to not having to use a separate network to "
                                                               "compute the update, but is more susceptible to instability due to shifts in the distribution "
                                                               "of states seen during search.")
    train_group.add_argument('--up_lt', type=float, default=np.inf, help="Loss must be below this threshold for update.")

    # test args
    test_group = parser.add_argument_group('test')
    test_group.add_argument('--t_file', type=str, default=None, help="File to use when testing.")
    test_group.add_argument('--t_search_itrs', type=int, default=100, help="Number of search iterations when testing.")
    test_group.add_argument('--t_up_freq', type=int, default=10, help="Test every t_up_freq updates.")
    test_group.add_argument('--t_pathfinds', type=str, default="bwas", help="Comma separated list of pathfinding algorithms to use when testing.")
    test_group.add_argument('--t_init', action='store_true', default=False, help="Set for testing before training begins.")

    # other
    parser.add_argument('--debug', action='store_true', default=False, help="Set for debug mode.")
    parser.set_defaults(func=train_cli)


def train_cli(args: argparse.Namespace) -> None:
    # logging
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if not args.debug:
        output_save_loc = f"{args.dir}/output.txt"
        sys.stdout = Logger(output_save_loc, "a")

    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # parse domain and heur_nnet
    domain, domain_name = get_domain_from_arg(args.domain)

    # parse nnet par runners
    # TODO set nnet files outside
    nnet_par_run_dict: Dict[str, NNetParRunner] = dict()
    if args.heur is not None:
        heur_nnet_par: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]
        heur_targ_name: str = f"{args.dir}/heur_targ.pt"
        if args.heur_type == "V":
            assert isinstance(heur_nnet_par, HeurVNNetPar)
            nnet_par_run_dict["heurv"] = HeurVNNetParRunner(heur_nnet_par, heur_targ_name)
        else:
            assert isinstance(heur_nnet_par, HeurQNNetPar)
            nnet_par_run_dict["heurq"] = HeurQNNetParRunner(heur_nnet_par, heur_targ_name)
        print(heur_nnet_par)

    if args.policy is not None:
        policy_nnet_par: PolicyNNetPar = get_policy_nnet_par_from_arg(domain, domain_name, args.policy, args.policy_samp)[0]
        policy_targ_name: str = f"{args.dir}/policy_targ.pt"
        nnet_par_run_dict["policy"] = PolicyNNetParRunner(policy_nnet_par, policy_targ_name)
        print(policy_nnet_par)

    # updater
    updater: Update = get_updater_from_args(domain, args.pathfind, nnet_par_run_dict, args.up)[0]
    print(f"{updater}")

    # train args
    train_args: TrainArgs = TrainArgs(args.batch_size, args.max_itrs, args.bal, up_itrs=args.up_itrs, up_gen_itrs=args.up_gen_itrs, rb=args.rb,
                                      loss_thresh=args.up_lt, checkpoint=args.chkpt, grad_accum=args.accum, display=args.display)

    print(f"{train_args}")
    print(domain)

    # TODO print pathfind

    """
    # test args
    test_args: Optional[TestArgs] = None
    if args.t_file is not None:
        data = pickle.load(open(args.t_file, "rb"))
        states: List[State] = data['states']
        goals: List[Goal] = data['goals']
        test_args = TestArgs(states, goals, args.t_search_itrs, args.t_pathfinds.split(","), args.up_nnet_batch_size, args.t_up_freq, args.t_init)
        print(f"{test_args}")
    """

    writer: SummaryWriter = SummaryWriter(args.dir)
    trainer: Train = TrainHeurV(args.dir, updater, device, on_gpu, writer, train_args)

    trainer.train_loop()
