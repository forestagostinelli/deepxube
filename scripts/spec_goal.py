from typing import List, cast

from deepxube.environments.environment_abstract import EnvGrndAtoms, State
from deepxube.utils import env_select, program_utils, nnet_utils, viz_utils
from deepxube.search_state.spec_goal_asp import path_to_spec_goal
from deepxube.logic.program import Clause
from argparse import ArgumentParser
import torch


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--bk_add', type=str, default="", help="File of additional background knowledge")

    parser.add_argument('--model_batch_size', type=int, default=1, help="Maximum number of models sampled at once "
                                                                        "for parallelized search")

    parser.add_argument('--heur', type=str, required=True, help="nnet model file")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for batch-weighted A* search")
    parser.add_argument('--weight', type=float, default=0.2, help="Weight on path cost f(n) = w * g(n) + h(n)")
    parser.add_argument('--max_search_itrs', type=float, default=100, help="Maximum number of iterations to search "
                                                                           "for a path to a given model.")

    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect path found, but will "
                                                                          "help if nnet is running out of memory.")

    parser.add_argument('--spec', type=str, required=True, help="Should have 'goal' in the head. "
                                                                "Separate multiple clauses by ';'")
    parser.add_argument('--spec_verbose', action='store_true', default=False, help="Set for verbose specification")
    parser.add_argument('--search_verbose', action='store_true', default=False, help="Set for verbose search")
    parser.add_argument('--viz_start', action='store_true', default=False, help="Set to visualize starting state")
    parser.add_argument('--viz_model', action='store_true', default=False, help="Set to visualize each model before "
                                                                                "search")

    args = parser.parse_args()

    # environment
    env: EnvGrndAtoms = cast(EnvGrndAtoms, env_select.get_environment(args.env))

    # start state
    state: State = env.get_start_states(1)[0]
    if args.viz_start:
        print("Starting state visualization:")
        viz_utils.visualize_examples(env, [state])

    # spec clauses
    spec_clauses_str = args.spec.split(";")
    clauses: List[Clause] = []
    for clause_str in spec_clauses_str:
        clause = program_utils.parse_clause(clause_str)[0]
        clauses.append(clause)
    print("Parsed input clauses:")
    print(clauses)

    # heuristic function
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    heuristic_fn = nnet_utils.load_heuristic_fn(args.heur, device, on_gpu, env.get_v_nnet(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size)

    # find path to goal
    found, state_path, action_path, path_cost, times = path_to_spec_goal(env, state, clauses, heuristic_fn,
                                                                         args.model_batch_size, args.batch_size,
                                                                         args.weight, args.max_search_itrs,
                                                                         bk_add=args.bk_add,
                                                                         spec_verbose=args.spec_verbose,
                                                                         search_verbose=args.search_verbose,
                                                                         viz_model=args.viz_model)
    if found:
        print("Found goal state")
        viz_utils.visualize_examples(env, [state_path[-1]])
    else:
        print("Did not find goal state")

    print(times.get_time_str())


if __name__ == "__main__":
    main()
