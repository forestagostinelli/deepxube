from typing import Optional
import argparse
from argparse import ArgumentParser

from deepxube.factories.updater_factory import get_updater

from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train, TestArgs
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg, get_pathfind_name_kwargs

import os
import pickle


def parser_train(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    parser.add_argument('--heur', type=str, required=True, help="Heuristic neural network and arguments.")
    parser.add_argument('--heur_type', type=str, required=True, help="V, QFix, QIn.")
    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments.")

    parser.add_argument('--dir', type=str, required=True, help="Directory to save neural networks.")

    # train args
    train_group = parser.add_argument_group('train')
    train_group.add_argument('--batch_size', type=int, default=1000, help="Batch size.")
    train_group.add_argument('--lr', type=float, default=0.001, help=" Learning rate.")
    train_group.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay.")
    train_group.add_argument('--max_itrs', type=int, default=100000, help="Maximum training iterations.")
    train_group.add_argument('--display', type=int, default=0, help="Display frequency for nnet training.")
    train_group.add_argument('--bal', action='store_true', default=False, help="Set to balance of number of steps to take to generate problem instances based "
                                                                               "on percentage of states solved.")

    # updater args
    update_group = parser.add_argument_group('update')
    update_group.add_argument('--procs', type=int, default=1, help="Number of processes to generate update data.")
    update_group.add_argument('--step_max', type=int, required=True, help="Maximum number of steps to take when generating problem instnaces.")
    update_group.add_argument('--up_itrs', type=int, default=100, help="Number of iterations to check for update.")
    update_group.add_argument('--search_itrs', type=int, default=1000, help="Number of search iterations to take when generating data.")
    update_group.add_argument('--up_batch_size', type=int, default=100, help="Maximum number of problem instances to generate at a time. Lower if running out "
                                                                             "of memory.")
    update_group.add_argument('--up_nnet_batch_size', type=int, default=20000, help="Maximum number of inputs to give to any nnet at a time during update. "
                                                                                    "Lower if running out of memory.")
    update_group.add_argument('--sync_main', action='store_true', default=False, help="Use main nnet to search during update.")
    update_group.add_argument('--up_v', action='store_true', default=False, help="Verbose update.")

    # update heur args
    update_group.add_argument('--backup', type=int, default=1, help="1 for Bellman backup, -1 for limited horizon bellman lookahead (LHBL)")

    # test args
    test_group = parser.add_argument_group('test')
    test_group.add_argument('--t_search_itrs', type=int, default=1000, help="Number of search iterations when testing.")
    test_group.add_argument('--t_up_freq', type=int, default=10, help="Test every t_up_freq updates.")

    # other
    parser.add_argument('--debug', action='store_true', default=False, help="Set for debug mode.")
    parser.set_defaults(func=train_cli)


def train_cli(args: argparse.Namespace) -> None:
    # parse domain and heur_nnet
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]
    pathfind_name, pathfind_kwargs = get_pathfind_name_kwargs(args.pathfind)

    # update args
    up_args: UpArgs = UpArgs(args.procs, args.up_itrs, args.step_max, args.search_itrs,
                             up_batch_size=args.up_batch_size, nnet_batch_size=args.up_nnet_batch_size,
                             sync_main=args.sync_main, v=args.up_v)
    up_heur_args: UpHeurArgs = UpHeurArgs(False, args.backup)

    # updater
    updater: UpdateHeur = get_updater(domain, heur_nnet, pathfind_name, pathfind_kwargs, up_args, up_heur_args)

    # train args
    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.bal,
                                      display=args.display)

    # test args
    valid_file: str = f"data/{args.domain}/valid.pkl"
    test_args: Optional[TestArgs]
    if os.path.isfile(valid_file):
        states, goals = pickle.load(open(valid_file, "rb"))
        test_args = TestArgs(states, goals, args.t_search_itrs, [0.0], args.up_nnet_batch_size, args.t_up_freq,
                             False)
    else:
        test_args = None

    # test args
    train(updater, args.dir, train_args, test_args=test_args, debug=args.debug)
