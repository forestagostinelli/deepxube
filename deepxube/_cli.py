from typing import List, Optional, Tuple, Type, Dict, Any, cast
import argparse
from argparse import ArgumentParser
from dataclasses import fields

from deepxube._train_cli import parser_train
from deepxube._solve import parse_solve
from deepxube.pytorch.nnet_utils import get_device
from deepxube.base.factory import Factory, Parser
from deepxube.base.domain import Domain, StateGoalVizable, StringToAct, State, Action, Goal
from deepxube.base.nnet_input import NNetInput
from deepxube.base.nnet import DeepXubeNNet
from deepxube.base.pathfind_fns import PFNs, DeepXubeNNetPar, UFNs
from deepxube.base.pathfinding import PathFind
from deepxube.base.updater import Update
from deepxube.base.trainer import Train
from deepxube.factories.domain_factory import domain_factory, get_domain_from_arg
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.nnet_factory import deepxube_nnet_factory
from deepxube.factories.pathfind_fns_factory import pathfind_fns_factory, deepxube_nnet_par_factory, updater_fns_factory, get_path_up_fns
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.factories.updater_factory import updater_factory
from deepxube.factories.trainer_factory import trainer_factory
from deepxube.base.trainer import TrainSummary
from deepxube.tests.time_tests import time_test

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from PIL import Image
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


def get_names_match_type(req_type: Type, factory: Factory) -> List[str]:
    names: List[str] = []
    for class_name in factory.get_all_class_names():
        class_t: Type = factory.get_type(class_name)
        if issubclass(class_t, req_type):
            names.append(class_name)

    return names


def domain_info(args: argparse.Namespace) -> None:
    domain_name: str
    domain_t: Type[Domain]
    if args.domain is None:
        domain_names: List[str] = domain_factory.get_all_class_names()
        for domain_name in domain_names:
            domain_t = domain_factory.get_type(domain_name)
            print(f"Domain (Name, Module, Class): {domain_name}, {domain_t.__module__}, {domain_t.__qualname__}")
    else:
        domain_name = args.domain

        domain_t = domain_factory.get_type(domain_name)
        print(f"Domain (Name, Module, Class): {domain_name}, {domain_t.__module__}, {domain_t.__qualname__}")

        # mixins
        mixin_str: str = textwrap.indent(', '.join([f"{x.__qualname__}" for x in get_immediate_mixins(domain_t, Domain)]), '\t')
        print("Mixins:\n" + mixin_str, '\t')

        # nnet inputs
        nnet_input_t_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
        print("NNet Inputs (Name, Module, Class):")
        for nnet_input_t_key in nnet_input_t_keys:
            nnet_input_t: Type[NNetInput] = get_nnet_input_t(nnet_input_t_key)
            print(textwrap.indent(f"{nnet_input_t_key[1]}, {nnet_input_t.__module__}, {nnet_input_t.__qualname__}", '\t'))

        parser: Optional[Parser] = domain_factory.get_parser(domain_name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'), '\t')


def nnet_info(args: argparse.Namespace) -> None:
    nnet_name: str
    nnet_t: Type[DeepXubeNNet]
    if args.name is None:
        nnet_names: List[str] = deepxube_nnet_factory.get_all_class_names()
        for nnet_name in nnet_names:
            nnet_t = deepxube_nnet_factory.get_type(nnet_name)
            print(f"DeepXubeNNet (Name, Module, Class): {nnet_name}, {nnet_t.__module__}, {nnet_t.__qualname__}")
    else:
        nnet_name = args.name
        nnet_t = deepxube_nnet_factory.get_type(nnet_name)
        print(f"DeepXubeNNet (Name, Module, Class): {nnet_name}, {nnet_t.__module__}, {nnet_t.__qualname__}")

        nnet_input_t: Type[NNetInput] = nnet_t.nnet_input_type()
        print(f"Expected NNet_Input (Module, Class): {nnet_input_t.__module__}, {nnet_input_t.__qualname__}")

        parser: Optional[Parser] = deepxube_nnet_factory.get_parser(nnet_name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'))


def fn_info(args: argparse.Namespace) -> None:
    name: str
    nn_par_t: Type[DeepXubeNNetPar]
    if args.name is None:
        names: List[str] = deepxube_nnet_par_factory.get_all_class_names()
        for name in names:
            nn_par_t = deepxube_nnet_par_factory.get_type(name)
            print(f"DeepXubeNNetPar (Name, Module, Class): {name}, {nn_par_t.__module__}, {nn_par_t.__qualname__}")
    else:
        name = args.name
        nn_par_t = deepxube_nnet_par_factory.get_type(name)
        print(f"DeepXubeNNetPar (Name, Module, Class): {name}, {nn_par_t.__module__}, {nn_par_t.__qualname__}")

        domain_t: Type[Domain] = nn_par_t.domain_type()
        print(f"Expected Domain (Module, Class): {domain_t.__module__}, {domain_t.__qualname__}")

        nn_t: Type[DeepXubeNNet] = nn_par_t.nnet_type()
        print(f"Expected DeepXubeNNet (Module, Class): {nn_t.__module__}, {nn_t.__qualname__}")

        nnet_input_t: Type[NNetInput] = nn_par_t.nnet_input_type()
        print(f"Expected NNet_Input (Module, Class): {nnet_input_t.__module__}, {nnet_input_t.__qualname__}")

        parser: Optional[Parser] = deepxube_nnet_factory.get_parser(name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'))


def pathfind_fns_info(args: argparse.Namespace) -> None:
    name: Tuple[Tuple[str, Type], ...]
    path_fns_t: Type[PFNs]
    if args.name is None:
        names: List[Tuple[Tuple[str, Type], ...]] = pathfind_fns_factory.get_all_class_names()
        for name in names:
            path_fns_t = pathfind_fns_factory.get_type(name)
            print(f"Pathfind Functions (Name, Module, Class): {name}, {path_fns_t.__module__}, {path_fns_t.__qualname__}")


def updater_fns_info(args: argparse.Namespace) -> None:
    name: Tuple[Tuple[str, Type], ...]
    up_fns_t: Type[UFNs]
    if args.name is None:
        names: List[Tuple[Tuple[str, Type], ...]] = updater_fns_factory.get_all_class_names()
        for name in names:
            up_fns_t = updater_fns_factory.get_type(name)
            print(f"Updater Functions (Name, Module, Class): {name}, {up_fns_t.__module__}, {up_fns_t.__qualname__}")


def pathfind_info(args: argparse.Namespace) -> None:
    name: str
    pathfind_t: Type[PathFind]
    if args.name is None:
        names: List[str] = pathfinding_factory.get_all_class_names()
        for name in names:
            pathfind_t = pathfinding_factory.get_type(name)
            print(f"PathFind (Name, Module, Class): {name}, {pathfind_t.__module__}, {pathfind_t.__qualname__}")
    else:
        name = args.name
        pathfind_t = pathfinding_factory.get_type(name)
        print(f"PathFind (Name, Module, Class): {name}, {pathfind_t.__module__}, {pathfind_t.__qualname__}")
        print(f"Description: {pathfind_t.description()}")
        mixin_str: str = ', '.join([f"{x.__qualname__}" for x in get_immediate_mixins(pathfind_t, PathFind)])
        print(f"Mixins: {mixin_str}", '\t')
        print(f"Expected Domain type: {pathfind_t.domain_type().__qualname__}", '\t')
        print(f"Expected Functions type: {pathfind_t.pathfind_functions_type().__qualname__}", '\t')

        parser: Optional[Parser] = pathfinding_factory.get_parser(name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'))


def updater_info(args: argparse.Namespace) -> None:
    name: str
    up_t: Type[Update]
    if args.name is None:
        names: List[str] = updater_factory.get_all_class_names()
        for name in names:
            up_t = updater_factory.get_type(name)
            print(f"Updater (Name, Module, Class): {name}, {up_t.__module__}, {up_t.__qualname__}")
    else:
        name = args.name
        up_t = updater_factory.get_type(name)
        print(f"Updater (Name, Module, Class): {name}, {up_t.__module__}, {up_t.__qualname__}")
        mixin_str: str = ', '.join([f"{x.__qualname__}" for x in get_immediate_mixins(up_t, Update)])
        print(f"Mixins: {mixin_str}", '\t')
        print(f"Expected Domain type: {up_t.domain_type().__qualname__}", '\t')
        print(f"Expected PathFind type: {up_t.pathfind_type().__qualname__} with functions {up_t.pathfind_functions_type().__qualname__}", '\t')
        print(f"Expected Updater functions type: {up_t.updater_functions_type()}", '\t')

        parser: Optional[Parser] = updater_factory.get_parser(name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'))


def trainer_info(args: argparse.Namespace) -> None:
    name: str
    tr_t: Type[Train]
    if args.name is None:
        names: List[str] = trainer_factory.get_all_class_names()
        for name in names:
            tr_t = trainer_factory.get_type(name)
            print(f"Trainer (Name, Module, Class): {name}, {tr_t.__module__}, {tr_t.__qualname__}")
    else:
        name = args.name
        tr_t = trainer_factory.get_type(name)
        print(f"Trainer (Name, Module, Class): {name}, {tr_t.__module__}, {tr_t.__qualname__}")
        print(f"Expected DeepXubeNNet type: {tr_t.nnet_type().__qualname__}", '\t')
        print(f"Expected Updater type: {tr_t.updater_type().__qualname__}", '\t')

        parser: Optional[Parser] = trainer_factory.get_parser(name)
        if parser is not None:
            print("Parser help:\n" + textwrap.indent(parser.help(), '\t'))


def fig_to_rgba(fig: Figure) -> NDArray:
    fig.canvas.draw()
    rgba: NDArray = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    return rgba


def viz_step(domain: StateGoalVizable, data: Dict, idx: int, state_idx: int, state_idx_max: int, states_on_path: List[State], state: State, goal: Goal,
             no_act: bool, fig: Figure) -> Tuple[State, int]:
    solved: bool = data['solved'][idx]

    action: Action = data['actions'][idx][state_idx]
    print(f"Action: {action}")

    state_idx += 1
    if no_act:
        state = states_on_path[state_idx]
    else:
        state_next_l, tcs = domain.next_state([state], [action])
        state_next: State = state_next_l[0]
        print(f"Transition cost: {tcs[0]}")
        assert state_next == states_on_path[state_idx]
        state = state_next

    _viz_state_goal_update(domain, state, goal, fig)

    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
    if (state_idx == state_idx_max) and solved:
        assert domain.is_solved([state], [goal])[0]

    return state, state_idx


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
        states, goals = domain.sample_problem_instances([args.steps])
        state = states[0]
        goal = goals[0]

    fig: Figure = plt.figure(figsize=(5, 5))
    assert isinstance(domain, StateGoalVizable)
    domain.visualize_state_goal(state, goal, fig)
    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")

    if args.soln:
        states_on_path: Optional[List[State]] = data['states_on_path'][args.idx]
        if states_on_path is not None:
            state_idx: int = 0
            state_idx_max: int = len(states_on_path) - 1
            if args.o is not None:
                rgba_l: List = []
                while state_idx < state_idx_max:
                    rgba_l.append(fig_to_rgba(fig).copy())
                    state, state_idx = viz_step(domain, data, args.idx, state_idx, state_idx_max, states_on_path, state, goal, args.no_act, fig)
                rgba_l.append(fig_to_rgba(fig).copy())

                frames: List[Image.Image] = [Image.fromarray(rgba, mode="RGBA") for rgba in rgba_l]
                frames[0].save(args.o, save_all=True, append_images=frames[1:], duration=1000 * args.v_time, loop=0)
            else:
                plt.show(block=False)
                while True:
                    act_str = input(f"State idx {state_idx} of {state_idx_max} on path. Next state (n), Previous state (p), Video (v), state idx, "
                                    f"'!' to quit: ")
                    if act_str == "!":
                        break
                    if act_str.upper() == "N":
                        if state_idx < state_idx_max:
                            state, state_idx = viz_step(domain, data, args.idx, state_idx, state_idx_max, states_on_path, state, goal, args.no_act, fig)
                    elif act_str.upper() == "V":
                        while state_idx < state_idx_max:
                            state, state_idx = viz_step(domain, data, args.idx, state_idx, state_idx_max, states_on_path, state, goal, args.no_act, fig)
                            plt.pause(float(args.v_time))
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
            input("No path (press enter to quit): ")
    else:
        if isinstance(domain, StringToAct):
            print(domain.string_to_action_help())
        if args.o is not None:
            rgba: NDArray = fig_to_rgba(fig)
            img = Image.fromarray(rgba, mode="RGBA")
            img.save(args.o)
        else:
            plt.show(block=False)
            while True:
                # get input
                input_options: List[str] = ["nothing for random action"]
                if isinstance(domain, StringToAct):
                    input_options.append("action string")
                input_options.append("'!' to quit")
                input_str = f"Enter {'; or '.join(input_options)}: "

                act_str = input(input_str)
                if act_str == "!":
                    break

                # get action
                action_op: Optional[Action] = None
                if len(act_str) == 0:
                    action_op = domain.sample_state_action([state])[0]
                elif isinstance(domain, StringToAct):
                    action_op = domain.string_to_action(act_str)

                # take action
                if action_op is None:
                    print(f"No action '{act_str}'")
                else:
                    print(action_op)
                    states_next, tcs = domain.next_state([state], [action_op])
                    state = states_next[0]
                    print(f"Transition cost: {tcs[0]}")
                    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}")
                    _viz_state_goal_update(domain, state, goal, fig)


def _viz_state_goal_update(domain: StateGoalVizable, state: State, goal: Goal, fig: Figure) -> None:
    fig.clear()
    domain.visualize_state_goal(state, goal, fig)
    fig.canvas.draw()


def time_test_args(args: argparse.Namespace) -> None:
    domain, domain_name = get_domain_from_arg(args.domain)
    dx_nnet_par_l: List[DeepXubeNNetPar] = []
    if args.fn is not None:
        device = get_device()[0]
        updater_fns: UFNs = get_path_up_fns(domain, domain_name, args.fn, device)[1]
        for field in fields(updater_fns):
            dx_nnet_par: DeepXubeNNetPar = cast(DeepXubeNNetPar, getattr(updater_fns, field.name))
            dx_nnet_par_l.append(dx_nnet_par)

    time_test(domain, dx_nnet_par_l, args.num_insts, args.step_min, args.step_max)


def plot_itr_data(axs: List[Axes], step_slider: Slider, itr: int, itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]],
                  itr_to_steps_to_pathfindstats: Dict[int, Dict[int, Dict]]) -> None:
    steps_to_pathfindperf: Dict[int, Dict] = itr_to_steps_to_pathfindstats[itr]
    steps_at_itr: List[int] = sorted(steps_to_pathfindperf.keys())
    per_solved: List[float] = [steps_to_pathfindperf[step]["per_solved"] for step in steps_at_itr]
    path_costs: List[float] = [steps_to_pathfindperf[step]["path_costs"] for step in steps_at_itr]
    search_itrs: List[float] = [steps_to_pathfindperf[step]["search_itrs"] for step in steps_at_itr]
    targets: List[float] = [np.mean(steps_to_pathfindperf[step]["ctgs_backup"]) for step in steps_at_itr]
    num_instances: List[int] = [steps_to_pathfindperf[step]["num_instances"] for step in steps_at_itr]
    plot_scatter(axs[0], steps_at_itr, per_solved, "Step", "Percent Solved", False)
    plot_scatter(axs[1], steps_at_itr, path_costs, "Step", "Path Costs", False)
    plot_scatter(axs[2], steps_at_itr, search_itrs, "Step", "Search Iterations", False)
    plot_scatter(axs[3], steps_at_itr, targets, "Step", "Cost-to-Go Targets", False)
    plot_scatter(axs[4], steps_at_itr, num_instances, "Step", "# Instances", False)
    plot_scatter(axs[5], itr_to_in_out[itr][0], itr_to_in_out[itr][1], "Target", "Prediction", True, alpha=0.2)
    step_slider.valtext.set_text(f"Iteration {itr}")


def train_summary(args: argparse.Namespace) -> None:
    status_file: str = f"{args.dir}/{args.type}_train_summary.pkl"
    train_summ: TrainSummary = pickle.load(open(status_file, "rb"))
    itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]] = train_summ.itr_to_in_out
    itr_to_steps_to_pathfindperf: Dict[int, Dict[int, Dict]] = train_summ.itr_to_steps_to_pathfindstats
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

    fig.tight_layout()

    if args.o is not None:
        rgba_l: List = []
        for idx_gif in range(len(itrs)):
            step_slider.set_val(idx_gif)
            update(idx_gif)
            rgba_l.append(fig_to_rgba(fig).copy())

        frames: List[Image.Image] = [Image.fromarray(rgba, mode="RGBA") for rgba in rgba_l]
        frames[0].save(args.o, save_all=True, append_images=frames[1:], duration=1000 * args.v_time, loop=0)
    else:
        step_slider.on_changed(update)
        plt.show()


def problem_inst_gen(args: argparse.Namespace) -> None:
    if os.path.isfile(args.file) and (not args.redo):
        print(f"File {args.file} already exists and redo not set. Not generating data.")
        return

    domain, _ = get_domain_from_arg(args.domain)
    num_steps_l: List[int] = list(np.random.randint(args.step_min, args.step_max + 1, size=args.num))
    print(f"Generating {args.num} states")
    start_time = time.time()
    states, goals = domain.sample_problem_instances(num_steps_l)
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
    parser_domain_info.add_argument('--domain', '--name', type=str, default=None, help="Name of domain.")
    parser_domain_info.set_defaults(func=domain_info)

    # nnet info
    parser_nnet_info: ArgumentParser = subparsers.add_parser('nnet_info', help="Print information on neural network architectures that deepxube has "
                                                                               "registered. Put user-defined definitions of neural networks in './nnets/'")
    parser_nnet_info.add_argument('--name', type=str, default=None, help="Name of nnet.")
    parser_nnet_info.set_defaults(func=nnet_info)

    # functions
    parser_fn_info: ArgumentParser = subparsers.add_parser('fn_info', help="Print information on parallel nnet functions")
    parser_fn_info.add_argument('--name', type=str, default=None, help="Name of nnet par function.")
    parser_fn_info.set_defaults(func=fn_info)

    parser_pathfind_fns_info: ArgumentParser = subparsers.add_parser('pathfind_fns_info', help="Print information on pathfinding functions")
    parser_pathfind_fns_info.add_argument('--name', type=str, default=None, help="Name of pathfinding functions object.")
    parser_pathfind_fns_info.set_defaults(func=pathfind_fns_info)

    parser_up_fns_info: ArgumentParser = subparsers.add_parser('updater_fns_info', help="Print information on updater functions")
    parser_up_fns_info.add_argument('--name', type=str, default=None, help="Name of updater functions object.")
    parser_up_fns_info.set_defaults(func=updater_fns_info)

    # pathfinding info
    parser_pathfind_info: ArgumentParser = subparsers.add_parser('pathfind_info', help="Print information on pathfinding algorithms that deepxube has "
                                                                                       "registered.")
    parser_pathfind_info.add_argument('--name', type=str, default=None, help="Name of pathfinding method.")
    parser_pathfind_info.set_defaults(func=pathfind_info)

    # updater info
    parser_up_info: ArgumentParser = subparsers.add_parser('updater_info', help="Print information on update algorithms that deepxube has registered.")
    parser_up_info.add_argument('--name', type=str, default=None, help="Name of update method.")
    parser_up_info.set_defaults(func=updater_info)

    # train info
    parser_tr_info: ArgumentParser = subparsers.add_parser('trainer_info', help="Print information on training algorithms that deepxube has registered.")
    parser_tr_info.add_argument('--name', type=str, default=None, help="Name of train method.")
    parser_tr_info.set_defaults(func=trainer_info)

    # visualization
    parser_viz: ArgumentParser = subparsers.add_parser('viz', help="Visualize states/goals")
    _parse_viz_info(parser_viz)

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


def _parse_viz_info(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to take to generate problem instnace.")
    parser.add_argument('--file', type=str, default=None, help="If given, visualize results from file.")
    parser.add_argument('--idx', type=int, default=0, help="Index of problem instance in file.")
    parser.add_argument('--v_time', type=float, default=0.5, help="Pause time for each step when showing video or gif (in seconds).")
    parser.add_argument('--soln', action='store_true', default=False, help="If true, then assumes file contains solutions for problem instances and will "
                                                                           "visualize them.")
    parser.add_argument('--no_act', action='store_true', default=False, help="If true, then will not take action in domain when stepping through solution to "
                                                                             "verify states match and will just use states on solution path.")
    parser.add_argument('--o', type=str, default=None, help="Output file. Extension should be .png for single image and .gif for solution.")
    parser.set_defaults(func=viz)


def _parse_time(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--fn', type=str, nargs='*', help="Function and neural network arguments separated by a comma.")
    parser.add_argument('--num_insts', type=int, default=10, help="Number of problem instances to generate.")
    parser.add_argument('--step_min', type=int, default=0, help="Min number of steps for problem instance generation.")
    parser.add_argument('--step_max', type=int, default=10, help="Max number of steps for problem instance generation.")
    parser.set_defaults(func=time_test_args)


def _parse_problem_instance(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--step_min', type=int, default=0, help="Minimum number of steps to take")
    parser.add_argument('--step_max', type=int, required=True, help="Maximum number of steps to take (inclusive)")
    parser.add_argument('--num', type=int, required=True, help="Number of problem instances to generate.")
    parser.add_argument('--file', type=str, required=True, help="File to which problem instances are stored.")
    parser.add_argument('--redo', action='store_true', default=False, help="If true, generate problem instances even if file already exists.")
    parser.set_defaults(func=problem_inst_gen)


def _parse_train_summary(parser: ArgumentParser) -> None:
    parser.add_argument('--dir', type=str, required=True, help="Training directory.")
    parser.add_argument('--type', type=str, default="heur", help="heur or policy")
    parser.add_argument('--v_time', type=float, default=0.5, help="Pause time for each step when making gif (in seconds).")
    parser.add_argument('--o', type=str, default=None, help="Output file. Extensrion should be .gif.")
    parser.set_defaults(func=train_summary)
