from typing import List, Optional, Tuple, Dict
from deepxube.base.domain import StateGoalVizable, StringToAct, State, Action, Goal
from matplotlib.figure import Figure
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.base.pathfinding import Instance, Node
import argparse
from argparse import ArgumentParser
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import pickle


def parse_viz(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to take to generate problem instnace.")
    parser.add_argument('--file', type=str, default=None, help="If given, visualize results from file.")
    parser.add_argument('--idx', type=int, default=0, help="Index of problem instance in file.")
    parser.add_argument('--v_time', type=float, default=0.5, help="Pause time for each step when showing video or gif (in seconds).")
    parser.add_argument('--soln', action='store_true', default=False, help="If true, then assumes file contains solutions for problem instances and will "
                                                                           "visualize them.")
    parser.add_argument('--inst', action='store_true', default=False, help="If true, then assumes file contains Instance objects seen during search.")
    parser.add_argument('--no_act', action='store_true', default=False, help="If true, then will not take action in domain when stepping through solution to "
                                                                             "verify states match and will just use states on solution path.")
    parser.add_argument('--o', type=str, default=None, help="Output file. Extension should be .png for single image and .gif for solution.")
    parser.set_defaults(func=viz)


def fig_to_rgba(fig: Figure) -> NDArray:
    fig.canvas.draw()
    rgba: NDArray = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    return rgba


def _viz_state_goal_update(domain: StateGoalVizable, state: State, goal: Goal, fig: Figure) -> None:
    fig.clear()
    domain.visualize_state_goal(state, goal, fig)
    fig.canvas.draw()


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


def viz_instance(domain: StateGoalVizable, fig: Figure, file: str, idx: int, v_time: float) -> None:
    data = pickle.load(open(file, "rb"))
    instances: List[Optional[Instance]] = data["instances"]
    instance: Optional[Instance] = instances[idx]
    if instance is not None:
        goal: Goal = instance.root_node.goal

        node: Node = instance.root_node
        state: State = node.state
        domain.visualize_state_goal(state, goal, fig)
        print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}, Heur: {node.heuristic}, PathCost: {node.path_cost}")
        plt.show(block=False)
        nodes_popped: List[Node] = instance.get_nodes_popped()
        node_idx: int = 0
        node_idx_max: int = len(nodes_popped) - 1

        while True:
            act_str = input(f"Nodes popped idx {node_idx} of {node_idx_max} on path. Next node (n), Previous node (p), node idx, "
                            f"'!' to quit: ")
            if act_str == "!":
                break
            elif act_str.upper() == "V":
                while node_idx < node_idx_max:
                    node_idx = node_idx + 1
                    node = nodes_popped[node_idx]
                    _viz_state_goal_update(domain, node.state, node.goal, fig)
                    print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}, Heur: {node.heuristic}, PathCost: {node.path_cost}")
                    plt.pause(v_time)
            else:
                if act_str.upper() == "N":
                    node_idx = min(node_idx + 1, node_idx_max)
                elif act_str.upper() == "P":
                    node_idx = max(node_idx - 1, 0)
                else:
                    node_idx = int(act_str)

                node = nodes_popped[node_idx]

                _viz_state_goal_update(domain, node.state, node.goal, fig)
                print(f"Goal Reached: {domain.is_solved([state], [goal])[0]}, Heur: {node.heuristic}, PathCost: {node.path_cost}")
    else:
        input("Instance is None (press enter to quit): ")


def viz(args: argparse.Namespace) -> None:
    # domain
    domain, domain_name = get_domain_from_arg(args.domain)
    assert isinstance(domain, StateGoalVizable)
    fig: Figure = plt.figure(figsize=(5, 5))
    if args.inst:
        viz_instance(domain, fig, args.file, args.idx, args.v_time)
        return

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
