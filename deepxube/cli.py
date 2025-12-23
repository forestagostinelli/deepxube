from typing import List, Optional, Tuple
import argparse
from argparse import ArgumentParser

from deepxube.train_cli import parser_train
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
                                                         "and heuristic search.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="")

    parser_tr: ArgumentParser = subparsers.add_parser('train', help="Train a heuristic function.",
                                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train(parser_tr)

    parser_domain_info: ArgumentParser = subparsers.add_parser('domain_info', help="Print information on domains that "
                                                                                   "deepxube has registered. "
                                                                                   "Put user-defined definitions of "
                                                                                   "domains in './domains/'")
    _parser_domain_info(parser_domain_info)

    parser_heur_info: ArgumentParser = subparsers.add_parser('heur_info', help="Print information on neural network "
                                                                               "representations of heuristic functions "
                                                                               "that deepxube has registered. "
                                                                               "Put user-defined definitions of "
                                                                               "heuristic neural networks in "
                                                                               "'./heuristics/'")
    _parser_heur_info(parser_heur_info)

    args = parser.parse_args()

    args.func(args)


def _parser_domain_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=domain_info)


def _parser_heur_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=heur_info)
