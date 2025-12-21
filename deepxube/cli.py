from typing import List, Optional, Tuple
import argparse
from argparse import ArgumentParser

from deepxube.train_cli import train_cli
from deepxube.base.domain import DomainParser
from deepxube.base.heuristic import HeurNNetParser
from deepxube.factories.domain_factory import get_all_domain_names, get_domain_parser
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys
from deepxube.factories.heuristic_factory import get_all_heur_nnet_names, get_heur_nnet_parser

import textwrap


def domain_info(args: argparse.Namespace) -> None:
    domain_names: List[str] = get_all_domain_names()
    for domain_name in domain_names:
        print(f"Domain: {domain_name}")
        parser: Optional[DomainParser] = get_domain_parser(domain_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))

        nnet_input_t_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
        if len(nnet_input_t_keys) > 0:
            print(f"\tNNet Inputs: {', '.join(nnet_input_t_key[1] for nnet_input_t_key in nnet_input_t_keys)}")


def heur_info(args: argparse.Namespace) -> None:
    heur_nnet_names: List[str] = get_all_heur_nnet_names()
    for heur_nnet_name in heur_nnet_names:
        print(f"Heur NNet: {heur_nnet_name}")
        parser: Optional[HeurNNetParser] = get_heur_nnet_parser(heur_nnet_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))


def main() -> None:
    parser = ArgumentParser(prog="deepxube", description="Solve pathfinding problems with deep reinforcement learning "
                                                         "and heuristic search.")
    subparsers = parser.add_subparsers(help="")

    parser_tr: ArgumentParser = subparsers.add_parser('train', help="")
    _parser_train(parser_tr)

    parser_domain_info: ArgumentParser = subparsers.add_parser('domain_info', help="")
    _parser_domain_info(parser_domain_info)

    parser_heur_info: ArgumentParser = subparsers.add_parser('heur_info', help="")
    _parser_heur_info(parser_heur_info)

    args = parser.parse_args()

    args.func(args)


def _parser_train(parser: ArgumentParser) -> None:
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
    parser.add_argument('--search_eps', type=float, default=0.1, help="")

    # update greedy policy args
    parser.add_argument('--search_temp', type=float, default=1, help="")

    # test args
    parser.add_argument('--t_search_itrs', type=int, default=1000, help="")
    parser.add_argument('--t_up_freq', type=int, default=10, help="")

    # other
    parser.add_argument('--rb', type=int, default=1, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    parser.set_defaults(func=train_cli)


def _parser_domain_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=domain_info)


def _parser_heur_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=heur_info)
