from typing import List, Optional, Tuple, cast, Type
import argparse
from argparse import ArgumentParser

from deepxube.train_cli import parser_train
from deepxube.base.domain import DomainParser, StateGoalVizable, StringToAct, State, Action, Goal
from deepxube.base.heuristic import HeurNNet, HeurNNetPar, HeurNNetParser
from deepxube.factories.domain_factory import get_all_domain_names, get_domain_parser
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.heuristic_factory import get_all_heur_nnet_names, get_heur_nnet_type, get_heur_nnet_parser
from deepxube.tests.time_tests import time_test
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg

import matplotlib.pyplot as plt

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
            print(textwrap.indent("NNet Inputs:", '\t'))
            for nnet_input_t_key in nnet_input_t_keys:
                print(textwrap.indent(f"Name: {nnet_input_t_key[1]}, Type: {get_nnet_input_t(nnet_input_t_key)}", '\t\t'))


def heur_info(args: argparse.Namespace) -> None:
    heur_nnet_names: List[str] = get_all_heur_nnet_names()
    for heur_nnet_name in heur_nnet_names:
        print(f"Heur NNet: {heur_nnet_name}")
        heur_nnet_t: Type[HeurNNet] = get_heur_nnet_type(heur_nnet_name)
        print(textwrap.indent(f"NNet_Input type expected: {heur_nnet_t.nnet_input_type()}", '\t'))
        parser: Optional[HeurNNetParser] = get_heur_nnet_parser(heur_nnet_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))


def viz(args: argparse.Namespace) -> None:
    domain, domain_name = get_domain_from_arg(args.domain)
    assert isinstance(domain, StateGoalVizable)
    states, goals = domain.get_start_goal_pairs([args.steps])
    state: State = states[0]
    goal: Goal = goals[0]
    fig = plt.figure(figsize=(5, 5))
    domain.visualize_state_goal(state, goal, fig)
    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")

    if isinstance(domain, StringToAct):
        plt.show(block=False)
        while True:
            act_str = input("Write action (make blank to quit): ")
            if len(act_str) == 0:
                break
            action: Optional[Action] = domain.string_to_action(act_str)
            if action is None:
                print(f"No action {act_str}")
            else:
                states_next, tcs = domain.next_state([state], [action])
                state = states_next[0]
                tc: float = tcs[0]
                fig.clear()
                cast(StateGoalVizable, domain).visualize_state_goal(state, goal, fig)
                print(f"Transition cost: {tc}")
                fig.canvas.draw()

                print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
    else:
        plt.show(block=True)


def time_test_args(args: argparse.Namespace) -> None:
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet_par: Optional[HeurNNetPar] = None
    if args.heur is not None:
        heur_nnet_par = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]
    time_test(domain, heur_nnet_par, args.num_insts, args.step_max)


def main() -> None:
    parser = ArgumentParser(prog="deepxube", description="Solve pathfinding problems with deep reinforcement learning "
                                                         "and heuristic search.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="")

    # domain info
    parser_domain_info: ArgumentParser = subparsers.add_parser('domain_info', help="Print information on domains that "
                                                                                   "deepxube has registered. "
                                                                                   "Put user-defined definitions of "
                                                                                   "domains in './domains/'")
    _parser_domain_info(parser_domain_info)

    # visualization
    parser_viz: ArgumentParser = subparsers.add_parser('viz', help="Visualize states/goals")
    _parse_viz_info(parser_viz)

    # heuristic info
    parser_heur_info: ArgumentParser = subparsers.add_parser('heuristic_info', help="Print information on neural network "
                                                                                    "representations of heuristic functions "
                                                                                    "that deepxube has registered. "
                                                                                    "Put user-defined definitions of "
                                                                                    "heuristic neural networks in "
                                                                                    "'./heuristics/'")
    _parser_heur_info(parser_heur_info)

    # test_domain_heur
    parser_time: ArgumentParser = subparsers.add_parser('time', help="Time basic functionality.",
                                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parse_time(parser_time)

    # train
    parser_tr: ArgumentParser = subparsers.add_parser('train', help="Train a heuristic function.",
                                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train(parser_tr)

    args = parser.parse_args()

    args.func(args)


def _parser_domain_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=domain_info)


def _parser_heur_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=heur_info)


def _parse_viz_info(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to take to generate problem instnace.")
    parser.set_defaults(func=viz)


def _parse_time(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--heur', type=str, default=None, help="Heuristic name and arguments.")
    parser.add_argument('--heur_type', type=str, default="V", help="V, QFix, QIn.")
    parser.add_argument('--num_insts', type=int, default=10, help="Number of problem instances to generate.")
    parser.add_argument('--step_max', type=int, default=10, help="Randomly generates problem instances with between 0 and step_max steps.")
    parser.set_defaults(func=time_test_args)
