from typing import Optional
import argparse
from argparse import ArgumentParser

from deepxube.factories.updater_factory import get_updater

from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.updater.updaters import UpGraphSearchArgs, UpGreedyPolicyArgs
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train, TestArgs
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg

import os
import pickle


def parser_train(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="")

    parser.add_argument('--heur', type=str, required=True, help="")
    parser.add_argument('--heur_type', type=str, required=True, help="V, QFix, QIn")

    parser.add_argument('--search', type=str, required=True, help="graph, greedy, sup")

    parser.add_argument('--dir', type=str, required=True, help="")

    # train args
    parser.add_argument('--batch_size', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=0.001, help="")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="")
    parser.add_argument('--max_itrs', type=int, default=100000, help="")
    parser.add_argument('--display', type=int, default=-1, help="")
    parser.add_argument('--no_bal', action='store_true', default=False, help="Set for no balancing")

    # updater args
    parser.add_argument('--procs', type=int, default=1, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--up_itrs', type=int, default=100, help="")
    parser.add_argument('--search_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_batch_size', type=int, default=100, help="")
    parser.add_argument('--up_nnet_batch_size', type=int, default=20000, help="")
    parser.add_argument('--sync_main', action='store_true', default=False, help="")
    parser.add_argument('--up_v', action='store_true', default=False, help="")

    # update heur args
    parser.add_argument('--backup', type=int, default=-1,
                        help="1 for Bellman backup, -1 for limited horizon bellman lookahead (LHBL)")

    # update graph search args
    parser.add_argument('--search_weight', type=int, default=1, help="")
    parser.add_argument('--search_eps', type=float, default=0.0, help="")

    # update greedy policy args
    parser.add_argument('--search_temp', type=float, default=1, help="")

    # test args
    parser.add_argument('--t_search_itrs', type=int, default=1000, help="")
    parser.add_argument('--t_up_freq', type=int, default=10, help="")

    # other
    parser.add_argument('--rb', type=int, default=1, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    parser.set_defaults(func=train_cli)


def train_cli(args: argparse.Namespace) -> None:
    # parse domain and heur_nnet
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]

    # update args
    up_args: UpArgs = UpArgs(args.procs, args.up_itrs, args.step_max, args.search_itrs,
                             up_batch_size=args.up_batch_size, nnet_batch_size=args.up_nnet_batch_size,
                             sync_main=args.sync_main, v=args.up_v)
    up_heur_args: UpHeurArgs = UpHeurArgs(False, args.backup)
    up_graphsch_args: UpGraphSearchArgs = UpGraphSearchArgs(args.search_weight, args.search_eps)
    up_greedy_args: UpGreedyPolicyArgs = UpGreedyPolicyArgs(args.search_eps, args.search_temp)

    # updater
    updater: UpdateHeur = get_updater(domain, heur_nnet, args.search, up_args, up_heur_args, up_graphsch_args,
                                      up_greedy_args)

    # train args
    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, not args.no_bal,
                                      rb=args.rb,
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