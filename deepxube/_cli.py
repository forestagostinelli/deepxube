from typing import List, Optional, Tuple, cast, Type
import argparse
from argparse import ArgumentParser

from deepxube._train_cli import parser_train
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, StateGoalVizable, StringToAct, State, Action, Goal
from deepxube.base.heuristic import HeurNNet, HeurNNetPar
from deepxube.base.pathfinding import PathFind, PathFindHeur
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.tests.time_tests import time_test
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg

import matplotlib.pyplot as plt

import textwrap


def get_mixins(cls: Type[object], mixin_base: Type) -> List[Type[object]]:
    return [base for base in cls.__mro__[1:] if issubclass(base, mixin_base) and base is not mixin_base]


def domain_info(args: argparse.Namespace) -> None:
    domain_names: List[str] = domain_factory.get_all_class_names()
    for domain_name in domain_names:
        domain_t: Type[Domain] = domain_factory.get_type(domain_name)
        print(f"Domain: {domain_name}, {domain_t}")
        parser: Optional[Parser] = domain_factory.get_parser(domain_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))

        mixin_str: str = ','.join([f"{x}" for x in get_mixins(domain_t, Domain)])
        print(textwrap.indent(f"Mixins: {mixin_str}", '\t'))

        nnet_input_t_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
        if len(nnet_input_t_keys) > 0:
            print(textwrap.indent("NNet Inputs:", '\t'))
            for nnet_input_t_key in nnet_input_t_keys:
                print(textwrap.indent(f"Name: {nnet_input_t_key[1]}, Type: {get_nnet_input_t(nnet_input_t_key)}", '\t\t'))


def heur_info(args: argparse.Namespace) -> None:
    heur_nnet_names: List[str] = heuristic_factory.get_all_class_names()
    for heur_nnet_name in heur_nnet_names:
        heur_nnet_t: Type[HeurNNet] = heuristic_factory.get_type(heur_nnet_name)
        print(f"Heur NNet: {heur_nnet_name}, {heur_nnet_t}")
        print(textwrap.indent(f"NNet_Input type expected: {heur_nnet_t.nnet_input_type()}", '\t'))
        parser: Optional[Parser] = heuristic_factory.get_parser(heur_nnet_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))


def pathfinding_info(args: argparse.Namespace) -> None:
    names: List[str] = pathfinding_factory.get_all_class_names()
    for name in names:
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(name)
        print(f"PathFind: {name}, {pathfind_t}")
        mixin_str: str = ','.join([f"{x}" for x in get_mixins(pathfind_t, PathFind)])
        print(textwrap.indent(f"Mixins: {mixin_str}", '\t'))

        print(textwrap.indent(f"Domain type expected: {pathfind_t.domain_type()}", '\t'))
        if issubclass(pathfind_t, PathFindHeur):
            print(textwrap.indent(f"Heuristic type expected: {pathfind_t.heur_fn_type()}", '\t'))

        parser: Optional[Parser] = pathfinding_factory.get_parser(name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))
        print("")


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


def problem_inst_gen(args: argparse.Namespace) -> None:
    pass


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

    # pathfinding info
    parser_pathfind_info: ArgumentParser = subparsers.add_parser('pathfinding_info', help="Print information on pathfinding algorithms that deepxube has "
                                                                                          "registered.")
    _parser_pathfind_info(parser_pathfind_info)

    # time functionality
    parser_time: ArgumentParser = subparsers.add_parser('time', help="Time basic functionality.",
                                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parse_time(parser_time)

    # problem instance generation
    parser_problem_instance: ArgumentParser = subparsers.add_parser('problem_inst', help="Generate problem instances (state/goal pairs) and save to a "
                                                                                         "pickle file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parse_problem_instance(parser_problem_instance)

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


def _parser_pathfind_info(parser: ArgumentParser) -> None:
    parser.set_defaults(func=pathfinding_info)


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


def _parse_problem_instance(parser: ArgumentParser) -> None:
    parser.add_argument('--step_max', type=int, required=True, help="Randomly generates problem instances with between 0 and step_max steps.")
    parser.add_argument('--num', type=int, required=True, help="Number of problem instances to generate.")
    parser.add_argument('--file', type=str, required=True, help="File to which problem instances are stored.")
    parser.add_argument('--redo', action='store_true', default=False, help="If true, generate problem instances even if file already exists.")
    parser.set_defaults(func=problem_inst_gen)
