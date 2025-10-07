from typing import List, Dict, Optional, Any

from deepxube.base.env import Env, State, Action, Goal
from deepxube.utils import data_utils
from deepxube.nnet import nnet_utils
from deepxube.base.pathfinding import get_path, NodeV
from deepxube.pathfinding.pathfinding_utils import is_valid_soln
from deepxube.pathfinding.v.bwas import BWAS
from deepxube.environments.env_utils import get_environment
import numpy as np
from argparse import ArgumentParser
import torch

import pickle
import os
import time
import sys


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment name")
    parser.add_argument('--insts', type=str, required=True, help="File containing instances (states and goals) to "
                                                                 "solve")
    parser.add_argument('--heur', type=str, required=True, help="nnet model file")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for batch-weighted A* pathfinding")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight on path cost f(n) = w * g(n) + h(n)")
    parser.add_argument('--time_limit', type=float, default=-1.0, help="A time limit for pathfinding. Default is -1, "
                                                                       "which means infinite.")

    parser.add_argument('--results', type=str, required=True, help="Directory to save results. Saves results after "
                                                                   "every instance.")
    parser.add_argument('--start_idx', type=int, default=0, help="Index of instance at which to start. "
                                                                 "Useful for debugging.")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect path found, but will "
                                                                          "help if nnet is running out of memory.")

    parser.add_argument('--redo', action='store_true', default=False, help="Set to start from scratch")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging with breakpoints")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    # environment
    env: Env = get_environment(args.env)

    # get data
    # sys.path.insert(0, '../DeepXube/deepxube/')  # TODO updater states to not need this
    data: Dict = pickle.load(open(args.insts, "rb"))
    states: List[State] = data['states']
    goals: List[Goal] = data['goals']

    results_file: str = "%s/results.pkl" % args.results
    output_file: str = "%s/output.txt" % args.results

    has_results: bool = False
    if os.path.isfile(results_file):
        has_results = True

    if has_results and (not args.redo):
        results: Dict[str, Any] = pickle.load(open(results_file, "rb"))
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "a")
    else:
        results: Dict[str, Any] = {"states": states, "actions": [], "states_on_path": [], "path_costs": [],
                                   "iterations": [], "times": [], "itrs/sec": [], "num_nodes_generated": [],
                                   "solved": []}
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "w")

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.heur, device, on_gpu, env.get_v_nnet(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size)

    start_idx = len(results["actions"])
    for state_idx in range(start_idx, len(states)):
        state: State = states[state_idx]
        goal: Goal = goals[state_idx]

        start_time = time.time()
        num_itrs: int = 0
        astar = BWAS(env)
        astar.add_instances([state], [goal], heuristic_fn, weights=[args.weight])
        while not min(x.finished for x in astar.instances):
            astar.step(heuristic_fn, batch_size=args.batch_size, verbose=args.verbose)
            num_itrs += 1
            if (args.time_limit >= 0) and ((time.time() - start_time) > args.time_limit):
                break
        solve_time = time.time() - start_time

        solved: bool = False
        path_states: Optional[List[State]] = None
        path_actions: Optional[List[Action]] = None
        path_cost: float = np.inf
        itrs_per_sec: float = num_itrs / solve_time
        num_nodes_gen_idx: int = astar.instances[0].num_nodes_generated
        goal_node: Optional[NodeV] = astar.instances[0].goal_node
        if goal_node is not None:
            path_states, path_actions, path_cost = get_path(goal_node)
            assert is_valid_soln(state, goal, path_actions, env)
            solved = True

        # record solution information
        results["actions"].append(path_actions)
        results["states_on_path"].append(path_states)
        results["path_costs"].append(path_cost)
        results["iterations"].append(num_itrs)
        results["itrs/sec"].append(itrs_per_sec)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved"].append(solved)

        # print to screen
        print(f"Times - {astar.times.get_time_str()}, num_itrs: {num_itrs}")

        print(f"State: %i, SolnCost: %.2f, # Nodes Gen: %s, Itrs: %i, Itrs/sec: %.2f, Solved: {solved}, "
              f"Time: %.2f" % (state_idx, path_cost, format(num_nodes_gen_idx, ","), num_itrs,
                               itrs_per_sec, solve_time))

        print(f"Means, SolnCost: %.2f, # Nodes Gen: %.2f, Itrs: %.2f, Itrs/sec: %.2f, Solved: %.2f%%, "
              f"Time: %.2f" % (_get_mean(results, "path_costs"), _get_mean(results, "num_nodes_generated"),
                               _get_mean(results, "iterations"), _get_mean(results, "itrs/sec"),
                               100.0 * np.mean(results["solved"]), _get_mean(results, "times")))
        print("")

        pickle.dump(results, open(results_file, "wb"), protocol=-1)


def _get_mean(results: Dict[str, Any], key: str) -> float:
    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    if len(vals) == 0:
        return 0
    else:
        mean_val = np.mean([x for x, solved in zip(results[key], results["solved"]) if solved])
        return float(mean_val)


if __name__ == "__main__":
    main()
