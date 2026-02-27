from typing import List, Optional, Tuple, cast, Type, Dict, Any
import argparse
from argparse import ArgumentParser

from deepxube._train_cli import parser_train
from deepxube._solve import parse_solve
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, StateGoalVizable, StringToAct, State, Action, Goal
from deepxube.base.heuristic import HeurNNet, HeurNNetPar
from deepxube.base.pathfinding import PathFind, PathFindHasHeur
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.pathfinding_factory import pathfinding_factory, get_domain_compat_pathfind_names
from deepxube.pathfinding.utils.performance import PathFindPerf
from deepxube.base.trainer import Status
from deepxube.tests.time_tests import time_test
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
import pickle
import textwrap
import numpy as np
from numpy.typing import NDArray
import os
import time


def plot_scatter(ax: Axes, x: Any, y: Any, x_label: str, y_label: str, xy_line: bool, alpha: float = 1.0, title: str = "") -> None:
    ax.scatter(x, y, s=10, alpha=alpha)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xy_line:
        np.linspace(0, max(x.max(), y.max()), 100)
        ax.plot(x, x, color='k', ls="--")
    ax.set_title(title)
    ax.grid(True)


def get_immediate_mixins(cls: Type[object], mixin_base: Type) -> List[Type]:
    return [base for base in cls.__bases__ if issubclass(base, mixin_base) and base is not mixin_base]


def domain_info(args: argparse.Namespace) -> None:
    domain_names: List[str]
    if args.names is None:
        domain_names = domain_factory.get_all_class_names()
    else:
        domain_names = args.names.split(",")

    for domain_name in domain_names:
        domain_t: Type[Domain] = domain_factory.get_type(domain_name)
        print(f"Domain: {domain_name}, {domain_t}")
        parser: Optional[Parser] = domain_factory.get_parser(domain_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))

        # mixins
        mixin_str: str = ', '.join([f"{x}" for x in get_immediate_mixins(domain_t, Domain)])
        print(textwrap.indent(f"Mixins: {mixin_str}", '\t'))

        # nnet inputs
        nnet_input_t_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
        print(textwrap.indent("NNet Inputs:", '\t'))
        for nnet_input_t_key in nnet_input_t_keys:
            print(textwrap.indent(f"Name: {nnet_input_t_key[1]}, Type: {get_nnet_input_t(nnet_input_t_key)}", '\t\t'))

        # pathfinding
        pathfind_names: List[str] = get_domain_compat_pathfind_names(domain_t)
        print(textwrap.indent("Pathfinding:", '\t'))
        for pathfind_name in pathfind_names:
            print(textwrap.indent(f"Name: {pathfind_name}, Type: {pathfinding_factory.get_type(pathfind_name)}", '\t\t'))
        print("")


def heur_info(args: argparse.Namespace) -> None:
    heur_nnet_names: List[str]
    if args.names is None:
        heur_nnet_names = heuristic_factory.get_all_class_names()
    else:
        heur_nnet_names = args.names.split(",")

    for heur_nnet_name in heur_nnet_names:
        heur_nnet_t: Type[HeurNNet] = heuristic_factory.get_type(heur_nnet_name)
        print(f"Heur NNet: {heur_nnet_name}, {heur_nnet_t}")
        print(textwrap.indent(f"NNet_Input type expected: {heur_nnet_t.nnet_input_type()}", '\t'))
        parser: Optional[Parser] = heuristic_factory.get_parser(heur_nnet_name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))
        print("")


def pathfinding_info(args: argparse.Namespace) -> None:
    names: List[str]
    if args.names is None:
        names = pathfinding_factory.get_all_class_names()
    else:
        names = args.names.split(",")

    for name in names:
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(name)
        print(f"PathFind: {name}, {pathfind_t}")
        mixin_str: str = ', '.join([f"{x}" for x in get_immediate_mixins(pathfind_t, PathFind)])
        print(textwrap.indent(f"Mixins: {mixin_str}", '\t'))

        print(textwrap.indent(f"Domain type expected: {pathfind_t.domain_type()}", '\t'))
        if issubclass(pathfind_t, PathFindHasHeur):
            print(textwrap.indent(f"Heuristic type expected: {pathfind_t.heur_fn_type()}", '\t'))

        parser: Optional[Parser] = pathfinding_factory.get_parser(name)
        if parser is not None:
            print(textwrap.indent("Parser: " + parser.help(), '\t'))
        print("")


def viz(args: argparse.Namespace) -> None:
    # domain
    domain, domain_name = get_domain_from_arg(args.domain)

    # state and goal
    state: State
    goal: Goal
    data: Dict = dict()
    if args.file is not None:
        data = pickle.load(open(args.file, "rb"))
        state = data['states'][args.idx]
        goal = data['goals'][args.idx]
    else:
        states, goals = domain.sample_start_goal_pairs([args.steps])
        state = states[0]
        goal = goals[0]

    fig: Figure = plt.figure(figsize=(5, 5))
    assert isinstance(domain, StateGoalVizable)
    domain.visualize_state_goal(state, goal, fig)
    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")

    if args.soln:
        solved: bool = data['solved'][args.idx]
        if solved:
            states_on_path: List[State] = data['states_on_path'][args.idx]
            state_idx: int = 0
            state_idx_max: int = len(states_on_path) - 1
            plt.show(block=False)
            while True:
                act_str = input(f"State idx {state_idx} of {state_idx_max} on solution path. Next state (n), Previous state (p), or state idx: ")
                if len(act_str) == 0:
                    break
                if act_str.upper() == "N":
                    if state_idx < state_idx_max:
                        action: Action = data['actions'][args.idx][state_idx]
                        print(f"Action: {action}")
                        state_next_l, tcs = domain.next_state([state], [action])
                        state_next: State = state_next_l[0]
                        print(f"Transition cost: {tcs[0]}")
                        state_idx += 1
                        assert state_next == states_on_path[state_idx]
                        state = state_next

                        _viz_state_goal_update(domain, state, goal, fig)

                        print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
                        if state_idx == state_idx_max:
                            assert domain.is_solved([state], [goal])[0]
                elif act_str.upper() == "P":
                    if state_idx > 0:
                        state_idx -= 1
                        state = states_on_path[state_idx]
                        _viz_state_goal_update(domain, state, goal, fig)

                        print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
                else:
                    state_idx = int(act_str)
                    assert state_idx >= 0
                    state = states_on_path[state_idx]
                    _viz_state_goal_update(domain, state, goal, fig)
                    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
        else:
            input("Not solved (press enter to quit): ")
    else:
        if isinstance(domain, StringToAct):
            print(domain.string_to_action_help())
            plt.show(block=False)
            while True:
                act_str = input("Write action (press enter to quit): ")
                if len(act_str) == 0:
                    break
                action_op: Optional[Action] = domain.string_to_action(act_str)
                if action_op is None:
                    print(f"No action {act_str}")
                else:
                    states_next, tcs = domain.next_state([state], [action_op])
                    state = states_next[0]
                    print(f"Transition cost: {tcs[0]}")
                    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
                    _viz_state_goal_update(cast(StateGoalVizable, domain), state, goal, fig)
        else:
            plt.show(block=True)


def _viz_state_goal_update(domain: StateGoalVizable, state: State, goal: Goal, fig: Figure) -> None:
    fig.clear()
    domain.visualize_state_goal(state, goal, fig)
    fig.canvas.draw()


def time_test_args(args: argparse.Namespace) -> None:
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet_par: Optional[HeurNNetPar] = None
    if args.heur is not None:
        heur_nnet_par = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]
    time_test(domain, heur_nnet_par, args.num_insts, args.step_max)


def plot_itr_data(axs: List[Axes], step_slider: Slider, itr: int, itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]],
                  itr_to_steps_to_pathfindperf: Dict[int, Dict[int, PathFindPerf]]) -> None:
    steps_to_pathfindperf: Dict[int, PathFindPerf] = itr_to_steps_to_pathfindperf[itr]
    steps_at_itr: List[int] = sorted(steps_to_pathfindperf.keys())
    per_solved: List[float] = [steps_to_pathfindperf[step].per_solved() for step in steps_at_itr]
    path_costs: List[float] = [steps_to_pathfindperf[step].stats()[1] for step in steps_at_itr]
    search_itrs: List[float] = [steps_to_pathfindperf[step].stats()[2] for step in steps_at_itr]
    targets: List[float] = [float(np.mean(steps_to_pathfindperf[step].ctgs_bkup)) for step in steps_at_itr]
    num_instances: List[int] = [len(steps_to_pathfindperf[step].ctgs_bkup) for step in steps_at_itr]
    plot_scatter(axs[0], steps_at_itr, per_solved, "Step", "Percent Solved", False)
    plot_scatter(axs[1], steps_at_itr, path_costs, "Step", "Path Costs", False)
    plot_scatter(axs[2], steps_at_itr, search_itrs, "Step", "Search Iterations", False)
    plot_scatter(axs[3], steps_at_itr, targets, "Step", "Cost-to-Go Targets", False)
    plot_scatter(axs[4], steps_at_itr, num_instances, "Step", "# Instances", False)
    plot_scatter(axs[5], itr_to_in_out[itr][0], itr_to_in_out[itr][1], "Target", "Prediction", True, alpha=0.2)
    step_slider.valtext.set_text(f"Iteration {itr}")


def train_summary(args: argparse.Namespace) -> None:
    status_file: str = f"{args.dir}/status.pkl"
    status: Status = pickle.load(open(status_file, "rb"))
    itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]] = status.itr_to_in_out
    itr_to_steps_to_pathfindperf: Dict[int, Dict[int, PathFindPerf]] = status.itr_to_steps_to_pathfindperf
    itrs: List[int] = sorted(itr_to_in_out.keys())
    fig, axs_np = plt.subplots(3, 2)
    axs: List[Axes] = axs_np.flatten().tolist()
    plt.subplots_adjust(bottom=0.2)
    axstep = fig.add_axes((0.2, 0.01, 0.65, 0.03))
    step_slider: Slider = Slider(
        ax=axstep,
        label='',
        valmin=0,
        valmax=len(itrs) - 1,
        valinit=0,
        valstep=1,
    )

    itr_init: int = min(itrs)
    plot_itr_data(axs, step_slider, itr_init, itr_to_in_out, itr_to_steps_to_pathfindperf)

    def update(idx: float) -> None:
        itr: int = itrs[int(idx)]
        for ax in axs:
            ax.cla()
        plot_itr_data(axs, step_slider, itr, itr_to_in_out, itr_to_steps_to_pathfindperf)
        fig.canvas.draw()

    step_slider.on_changed(update)
    fig.tight_layout()
    plt.show()


def problem_inst_gen(args: argparse.Namespace) -> None:
    if os.path.isfile(args.file) and (not args.redo):
        print(f"File {args.file} already exists and redo not set. Not generating data.")
        return

    domain, _ = get_domain_from_arg(args.domain)
    num_steps_l: List[int] = list(np.random.randint(args.step_max + 1, size=args.num))
    print(f"Generating {args.num} states")
    start_time = time.time()
    states, goals = domain.sample_start_goal_pairs(num_steps_l)
    print(f"Time: {time.time() - start_time}")

    print(f"Saving data to {args.file}")
    start_time = time.time()
    data: Dict = dict()
    data['states'] = states
    data['goals'] = goals
    # noinspection PyTypeChecker
    pickle.dump(data, open(args.file, "wb"), protocol=-1)
    print(f"Time: {time.time() - start_time}")


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

    # train
    parser_tr: ArgumentParser = subparsers.add_parser('train', help="Train a heuristic function.",
                                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train(parser_tr)

    # train summary
    parser_tr_summ: ArgumentParser = subparsers.add_parser('train_summary', help="Visualize training information not shown in tensorboard.",
                                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parse_train_summary(parser_tr_summ)

    # problem instance generation
    parser_problem_instance: ArgumentParser = subparsers.add_parser('problem_inst', help="Generate problem instances (state/goal pairs) and save to a "
                                                                                         "pickle file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parse_problem_instance(parser_problem_instance)

    # solve
    parser_solve: ArgumentParser = subparsers.add_parser('solve', help="Solve problem instnaces.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse_solve(parser_solve)

    args = parser.parse_args()

    args.func(args)


def _parser_domain_info(parser: ArgumentParser) -> None:
    parser.add_argument('--names', type=str, default=None, help="Comma separated value for only specific names. List all if None.")
    parser.set_defaults(func=domain_info)


def _parser_heur_info(parser: ArgumentParser) -> None:
    parser.add_argument('--names', type=str, default=None, help="Comma separated value for only specific names. List all if None.")
    parser.set_defaults(func=heur_info)


def _parser_pathfind_info(parser: ArgumentParser) -> None:
    parser.add_argument('--names', type=str, default=None, help="Comma separated value for only specific names. List all if None.")
    parser.set_defaults(func=pathfinding_info)


def _parse_viz_info(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to take to generate problem instnace.")
    parser.add_argument('--file', type=str, default=None, help="If given, visualize results from file.")
    parser.add_argument('--idx', type=int, default=0, help="Index of problem instance in file.")
    parser.add_argument('--soln', action='store_true', default=False, help="If true, then assumes file contains solutions for problem instances and will "
                                                                           "visualize them.")
    parser.set_defaults(func=viz)


def _parse_time(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--heur', type=str, default=None, help="Heuristic name and arguments.")
    parser.add_argument('--heur_type', type=str, default="V", help="V, QFix, QIn.")
    parser.add_argument('--num_insts', type=int, default=10, help="Number of problem instances to generate.")
    parser.add_argument('--step_max', type=int, default=10, help="Randomly generates problem instances with between 0 and step_max steps.")
    parser.set_defaults(func=time_test_args)


def _parse_problem_instance(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--step_max', type=int, required=True, help="Randomly generates problem instances with between 0 and step_max steps.")
    parser.add_argument('--num', type=int, required=True, help="Number of problem instances to generate.")
    parser.add_argument('--file', type=str, required=True, help="File to which problem instances are stored.")
    parser.add_argument('--redo', action='store_true', default=False, help="If true, generate problem instances even if file already exists.")
    parser.set_defaults(func=problem_inst_gen)


def _parse_train_summary(parser: ArgumentParser) -> None:
    parser.add_argument('--dir', type=str, required=True, help="Training directory.")
    parser.set_defaults(func=train_summary)
