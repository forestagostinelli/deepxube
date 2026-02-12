from typing import Optional, List
import argparse
from argparse import ArgumentParser

from deepxube.factories.updater_factory import get_updater

from deepxube.base.domain import State, Goal
from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train, TestArgs
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg, get_pathfind_name_kwargs, get_pathfind_from_arg

import pickle


def parser_train(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    parser.add_argument('--heur', type=str, required=True, help="Heuristic neural network and arguments.")
    parser.add_argument('--heur_type', type=str, default=None, help="V, QFix, QIn. V maps state/goal tuples to cost-to-go. "
                                                                    "QFix maps state/goal tuples to q_values for a fixed action space. "
                                                                    "QIn maps state/goal/action tuples to q_value (can be used in arbitrary action spaces).")
    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments. Batch size of any pathfinding algorithm should be 1 "
                                                                    "since updater assumes 1 instance is generated per iteration.")

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
    update_group.add_argument('--her', action='store_true', default=False, help="If problem instance not solved during search, do hindsight experience replay "
                                                                                "(HER) by relabeling deepest node in search tree as a goal state and "
                                                                                "sampling a goal from it.")
    update_group.add_argument('--sync_main', action='store_true', default=False, help="Use main nnet to search during update.")
    update_group.add_argument('--up_v', action='store_true', default=False, help="Verbose update.")

    # update heur args
    update_group.add_argument('--backup', type=int, default=1, help="1 for Bellman backup, -1 for limited horizon bellman lookahead (LHBL)")

    # test args
    test_group = parser.add_argument_group('test')
    test_group.add_argument('--t_file', type=str, default=None, help="File to use when testing.")
    test_group.add_argument('--t_search_itrs', type=int, default=100, help="Number of search iterations when testing.")
    test_group.add_argument('--t_up_freq', type=int, default=10, help="Test every t_up_freq updates.")
    test_group.add_argument('--t_pathfinds', type=str, default="bwas", help="Comma separated list of pathfinding algorithms to use when testing.")

    # other
    parser.add_argument('--debug', action='store_true', default=False, help="Set for debug mode.")
    parser.set_defaults(func=train_cli)


def train_cli(args: argparse.Namespace) -> None:
    # parse domain and heur_nnet
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet_par: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]
    pathfind_name, pathfind_kwargs = get_pathfind_name_kwargs(args.pathfind)
    get_pathfind_from_arg(domain, args.heur_type, args.pathfind)  # check heur type

    # update args
    up_args: UpArgs = UpArgs(args.procs, args.up_itrs, args.step_max, args.search_itrs,
                             up_batch_size=args.up_batch_size, nnet_batch_size=args.up_nnet_batch_size,
                             sync_main=args.sync_main, v=args.up_v)
    up_heur_args: UpHeurArgs = UpHeurArgs(False, args.backup)

    # updater
    updater: UpdateHeur = get_updater(domain, heur_nnet_par, pathfind_name, pathfind_kwargs, up_args, up_heur_args, args.her)

    # train args
    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.bal,
                                      display=args.display)

    # test args
    test_args: Optional[TestArgs]
    if args.t_file is not None:
        data = pickle.load(open(args.t_file, "rb"))
        states: List[State] = data['states']
        goals: List[Goal] = data['goals']
        test_args = TestArgs(states, goals, args.t_search_itrs, args.t_pathfinds.split(","), args.up_nnet_batch_size, args.t_up_freq, False)
    else:
        test_args = None

    # test args
    train(heur_nnet_par, args.heur_type, updater, args.dir, train_args, test_args=test_args, debug=args.debug)
