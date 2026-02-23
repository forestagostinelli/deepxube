from typing import List, Dict, Optional, Any, Tuple
import argparse
from argparse import ArgumentParser

from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.heuristic import HeurNNetPar, HeurFn, HeurFnV, HeurFnQ, PolicyFn
from deepxube.base.pathfinding import Node, Instance, PathFind, PathFindHasHeur, PathFindHasPolicy, get_path
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg, get_pathfind_from_arg
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.pathfinding.utils.performance import is_valid_soln
import numpy as np
from torch import nn

import pickle
import os
import time
import sys


def parse_solve(parser: ArgumentParser) -> None:
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    parser.add_argument('--heur', type=str, default=None, help="Heuristic neural network and arguments. If None then a heuristic whose output is always zero "
                                                               "is used.")
    parser.add_argument('--heur_file', type=str, default=None, help="File that has heuristic nnet. Can be None if using all zeros heuristic.")
    parser.add_argument('--heur_type', type=str, default=None, help="V, QFix, QIn. V maps state/goal tuples to cost-to-go. "
                                                                    "QFix maps state/goal tuples to q_values for a fixed action space. "
                                                                    "QIn maps state/goal/action tuples to q_value (can be used in arbitrary action spaces).")

    parser.add_argument('--policy', type=str, default=None, help="Policy neural network and arguments. If None then a policy that randomly samples actions "
                                                                 "with equal probability is used.")
    parser.add_argument('--policy_file', type=str, default=None, help="File that has policy nnet. Can be None if using random policy.")
    parser.add_argument('--policy_samp', type=int, default=None, help="Number of actions to sample.")

    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments.")
    parser.add_argument('--file', type=str, required=True, help="File containing problem instances to solve")

    parser.add_argument('--time_limit', type=float, default=-1.0, help="A time limit for search. Default is -1, which means infinite.")

    parser.add_argument('--results', type=str, required=True, help="Directory to save results. Saves results after every instance.")
    parser.add_argument('--start_idx', type=int, default=0, help="Index of instance at which to start. Useful for debugging.")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Maximum number of inputs to give to any nnet at a time during search. "
                                                                          "Lower if running out of memory. None means no limit.")

    parser.add_argument('--redo', action='store_true', default=False, help="Set to redo already completed instances")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging with breakpoints")
    parser.set_defaults(func=solve_cli)


def get_heur_fn(domain: Domain, domain_name: str, heur_nnet_str: Optional[str], heur_file: Optional[str], heur_type: Optional[str],
                nnet_batch_size: Optional[int]) -> Optional[HeurFn]:
    heur_fn: Optional[HeurFn] = None
    if heur_nnet_str is not None:
        assert heur_file is not None
        assert heur_type is not None
        heur_nnet_par: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, heur_nnet_str, heur_type)[0]
        device, devices, on_gpu = nnet_utils.get_device()
        print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

        nnet: nn.Module = nnet_utils.load_nnet(heur_file, heur_nnet_par.get_nnet())
        nnet.eval()
        nnet.to(device)
        nnet = nn.DataParallel(nnet)
        heur_fn = heur_nnet_par.get_nnet_fn(nnet, nnet_batch_size, device, None)
    elif heur_type is not None:
        if heur_type.upper() == "V":
            class HeurFnZerosV(HeurFnV):
                def __call__(self, states_in: List[State], goals_in: List[Goal]) -> List[float]:
                    return [0.0] * len(states_in)

            heur_fn = HeurFnZerosV()
        elif heur_type.upper() in {"QFIX", "QIN"}:
            class HeurFnZerosQ(HeurFnQ):
                def __call__(self, states_in: List[State], goals_in: List[Goal], actions_l_in: List[List[Action]]) -> List[List[float]]:
                    heur_vals_l: List[List[float]] = []
                    for actions_in in actions_l_in:
                        heur_vals_l.append([0.0] * len(actions_in))
                    return heur_vals_l

            heur_fn = HeurFnZerosQ()
        else:
            raise ValueError(f"Unknown heur type {heur_type}")

    return heur_fn


def get_policy_fn(domain: Domain, domain_name: str, policy_nnet_str: Optional[str], policy_file: Optional[str], use_policy: bool,
                  nnet_batch_size: Optional[int]) -> Optional[PolicyFn]:
    policy_fn: Optional[PolicyFn] = None
    if policy_nnet_str is not None:
        assert policy_file is not None
        raise NotImplementedError
    elif use_policy:
        class PolicyFnRand(PolicyFn):
            def __call__(self, domain_in: Domain, states: List[State], goals: List[Goal], num_samp_in: int) -> Tuple[List[List[Action]], List[List[float]]]:
                # sample actions
                states_rep_flat: List[State] = []
                for state in states:
                    states_rep_flat.extend([state] * num_samp_in)

                actions_samp_flat: List[Action] = domain_in.sample_state_action(states_rep_flat)

                # unflatten
                actions_samp: List[List[Action]] = []
                probs_l: List[List[float]] = []
                for _ in states:
                    actions_samp_i: List[Action] = list(set(actions_samp_flat[:num_samp_in]))  # make unique
                    actions_samp.append(actions_samp_i)
                    probs_l.append([1.0/len(actions_samp_i)] * len(actions_samp_i))
                    actions_samp_flat = actions_samp_flat[num_samp_in:]

                return actions_samp, probs_l

        policy_fn = PolicyFnRand()

    return policy_fn


def solve_cli(args: argparse.Namespace) -> None:
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    # domain
    domain, domain_name = get_domain_from_arg(args.domain)

    # heur and policy fn
    pathfind: PathFind = get_pathfind_from_arg(domain, args.heur_type, args.pathfind)[0]
    heur_fn: Optional[HeurFn] = get_heur_fn(domain, domain_name, args.heur, args.heur_file, args.heur_type, args.nnet_batch_size)
    policy_fn: Optional[PolicyFn] = get_policy_fn(domain, domain_name, args.policy, args.policy_file, isinstance(pathfind, PathFindHasPolicy),
                                                  args.nnet_batch_size)
    print(domain)
    print(pathfind)

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

    start_idx = len(results["actions"])
    for state_idx in range(start_idx, len(states)):
        # get problem instance
        state: State = states[state_idx]
        goal: Goal = goals[state_idx]

        # get pathfinding alg
        pathfind = get_pathfind_from_arg(domain, args.heur_type, args.pathfind)[0]
        if isinstance(pathfind, PathFindHasHeur):
            assert heur_fn is not None
            pathfind.set_heur_fn(heur_fn)
        if isinstance(pathfind, PathFindHasPolicy):
            assert policy_fn is not None
            pathfind.set_policy_fn(policy_fn, args.policy_samp)

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
        solve_time = time.time() - start_time

        # record results
        solved: bool = False
        path_states: Optional[List[State]] = None
        path_actions: Optional[List[Action]] = None
        path_cost: float = np.inf
        itrs_per_sec: float = num_itrs / solve_time
        num_nodes_gen_idx: int = pathfind.instances[0].num_nodes_generated
        goal_node: Optional[Node] = pathfind.instances[0].goal_node
        if goal_node is not None:
            path_states, path_actions, path_cost = get_path(goal_node)
            assert is_valid_soln(state, goal, path_actions, domain)
            solved = True

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
                              100.0 * np.mean(results["solved"]), _get_mean(results, "times")))
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
