from typing import List, Dict, Optional, Any, Tuple
import argparse
from argparse import ArgumentParser

import torch

from deepxube.pytorch.nnet_utils import get_device
from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfinding import Node, Instance, get_path
from deepxube.factories.pathfind_fns_factory import get_path_fns_nnet_par_dict
from deepxube.factories.pathfinding_factory import get_pathfind_from_arg
from deepxube.pathfinding.beam_search import BeamSearch
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.utils import data_utils, misc_utils
from deepxube.utils.command_line_utils import print_command
from deepxube.pathfinding.utils.performance import is_valid_soln
import numpy as np

import pickle
import os
import time
import sys


def policy_fn_rand(domain: Domain, states: List[State], num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
    if num_rand == 0:
        return [[] for _ in states], [[] for _ in states]

    states_rep: List[List[State]] = []
    for state in states:
        states_rep.append([state] * num_rand)

    states_rep_flat, split_idxs = misc_utils.flatten(states_rep)

    actions_samp_flat: List[Action] = domain.sample_state_action(states_rep_flat)
    actions_samp_l: List[List[Action]] = misc_utils.unflatten(actions_samp_flat, split_idxs)

    probs_l: List[List[float]] = []
    for actions_samp_i in actions_samp_l:
        probs_l.append([1.0 / len(actions_samp_i)] * len(actions_samp_i))

    return actions_samp_l, probs_l


def parse_solve(parser: ArgumentParser) -> None:
    # domain
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    # functions and corresponding nnets
    parser.add_argument('--fn', type=str, nargs='*', help="Function, neural network arguments, and neural network file separated by a comma.")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Maximum number of inputs to give to any nnet at a time during search. "
                                                                          "Lower if running out of memory. None means no limit.")
    # pathfinding
    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments.")
    parser.add_argument('--time_limit', type=float, default=-1.0, help="A time limit (in seconds) for search. Default is -1, which means infinite.")
    parser.add_argument('--max_itrs', type=int, default=None, help="Maximum number of search iterations. None for infinite.")

    # data
    parser.add_argument('--file', type=str, required=True, help="File containing problem instances to solve")
    parser.add_argument('--results', type=str, required=True, help="Directory to save results. Saves results after every instance.")
    parser.add_argument('--start_idx', type=int, default=None, help="Index of instance at which to start. Useful for debugging.")

    parser.add_argument('--redo', action='store_true', default=False, help="Set to redo already completed instances")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging with breakpoints")
    parser.set_defaults(func=solve_cli)


def solve_cli(args: argparse.Namespace) -> None:
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    # get data
    data: Dict = pickle.load(open(args.file, "rb"))
    states: List[State] = data['states']
    goals: List[Goal] = data['goals']

    results_file: str = "%s/results.pkl" % args.results
    output_file: str = "%s/output.txt" % args.results

    has_results: bool = False
    if os.path.isfile(results_file):
        has_results = True

    results: Dict[str, Any]
    if has_results and (not args.redo):
        results = pickle.load(open(results_file, "rb"))
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "a")
    else:
        results = {"states": states, "goals": goals, "actions": [], "states_on_path": [], "path_costs": [], "iterations": [], "times": [], "itrs/sec": [],
                   "num_nodes_generated": [], "solved": []}
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "w")

    print_command()

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # domain
    domain, domain_name = get_domain_from_arg(args.domain)

    # parse nnet fn args
    fns: List[str] = []
    nnet_files: List[Optional[str]] = []
    for fn in args.fn:
        fns_split: List[str] = fn.split(",")
        if len(fns_split) == 1:
            fns.append(f"{fns_split[0]}")
            nnet_files.append(None)
        elif len(fns_split) == 3:
            fns.append(f"{fns_split[0]},{fns_split[1]}")
            nnet_files.append(fns_split[2])
        else:
            raise ValueError("--fn must be either --fn <fn> or --fn <fn>,<nnet>,<nnet_file>")

    pathfind_fns, nnet_par_dict = get_path_fns_nnet_par_dict(domain, domain_name, fns, device, nnet_files=nnet_files, nnet_batch_size=args.nnet_batch_size)
    for nnet_par in nnet_par_dict.values():
        print(nnet_par)
        print(f"(name: {nnet_par.get_field_name()}, nnet_input_name: {nnet_par.nnet_input_name})")

    # pathfind functions
    print(pathfind_fns)

    # pathfinding
    pathfind, pathfind_name, _ = get_pathfind_from_arg(domain, pathfind_fns, args.pathfind)
    print(pathfind, f"(name: {pathfind_name})")

    start_idx: int
    if args.start_idx is not None:
        start_idx = args.start_idx
    else:
        start_idx = len(results["actions"])
    for state_idx in range(start_idx, len(states)):
        # get problem instance
        state: State = states[state_idx]
        goal: Goal = goals[state_idx]

        # get pathfinding alg
        pathfind = get_pathfind_from_arg(domain, pathfind_fns, args.pathfind)[0]

        # do pathfinding
        start_time = time.time()
        num_itrs: int = 0
        instance: Instance = pathfind.make_instances([state], [goal], None, True)[0]
        pathfind.add_instances([instance])
        while not min(x.finished() for x in pathfind.instances):
            pathfind.step(verbose=args.verbose)
            num_itrs += 1
            if (args.time_limit >= 0) and ((time.time() - start_time) > args.time_limit):
                break
            if (args.max_itrs is not None) and (num_itrs == args.max_itrs):
                break
        solve_time = time.time() - start_time

        # record results
        solved: bool = False
        path_states: Optional[List[State]] = None
        path_actions: Optional[List[Action]] = None
        path_cost: float = np.inf
        itrs_per_sec: float = num_itrs / solve_time
        num_nodes_gen_idx: int = pathfind.instances[0].num_nodes_generated
        goal_node: Optional[Node] = pathfind.instances[0].goal_node

        is_rollout: bool = isinstance(pathfind, BeamSearch) and pathfind.rollout  # special case
        if goal_node is not None:
            path_states, path_actions, tcs, path_cost = get_path(goal_node)
            assert (path_states is not None) and (path_actions is not None)
            if is_rollout:
                # see if any state on path is solved, if so, modify path to end at solved state
                is_sovled_path: List[bool] = domain.is_solved(path_states, [goal] * len(path_states))
                if any(is_sovled_path):
                    solved = True
                    solved_idx: int = is_sovled_path.index(True)
                    path_states = path_states[:(solved_idx + 1)]
                    path_actions = path_actions[:solved_idx]
                    tcs = tcs[:solved_idx]
                    path_cost = sum(tcs)
            else:
                solved = True

            if solved:
                assert path_actions is not None
                assert is_valid_soln(state, goal, path_actions, domain)

        results["actions"].append(path_actions)
        results["states_on_path"].append(path_states)
        results["path_costs"].append(path_cost)
        results["iterations"].append(num_itrs)
        results["itrs/sec"].append(itrs_per_sec)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved"].append(solved)

        # print to screen
        print(f"State: %i, SolnCost: %.2f, # Nodes Gen: %s, Itrs: %i, Itrs/sec: %.2f, Solved: {solved}, "
              f"Time: %.2f" % (state_idx, path_cost, format(num_nodes_gen_idx, ","), num_itrs,
                               itrs_per_sec, solve_time))

        print("Times - %s, num_itrs: %i" % (pathfind.times.get_time_str(), num_itrs))
        print("Means - SolnCost: %.2f, # Nodes Gen: %.2f, Itrs: %.2f, Itrs/sec: %.2f, Solved: %.2f%%, "
              "Time: %.2f" % (_get_mean(results, "path_costs"), _get_mean(results, "num_nodes_generated"),
                              _get_mean(results, "iterations"), _get_mean(results, "itrs/sec"),
                              float(100.0 * np.mean(results["solved"])), _get_mean(results, "times")))
        print("")

        # noinspection PyTypeChecker
        pickle.dump(results, open(results_file, "wb"), protocol=-1)


def _get_mean(results: Dict[str, Any], key: str) -> float:
    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    if len(vals) == 0:
        return 0
    else:
        mean_val = np.mean([x for x, solved in zip(results[key], results["solved"]) if solved])
        return float(mean_val)
