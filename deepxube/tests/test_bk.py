from typing import List, cast
from deepxube.environments.environment_abstract import EnvGrndAtoms
from deepxube.logic.asp import ASPSpec
from deepxube.logic.logic_objects import Clause, Model
from deepxube.utils import viz_utils
from deepxube.environments import env_utils
from deepxube.logic.logic_utils import parse_clause
import time
import argparse


def init_bk(env: EnvGrndAtoms, patterns_file: str) -> List[str]:
    bk_init: List[str] = env.get_bk()
    bk_init.append("")

    if len(patterns_file) > 0:
        pat_file = open(patterns_file, 'r')
        bk_init.extend(pat_file.read().split("\n"))
        pat_file.close()

    return bk_init


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bk_file', type=str, required=True, help="")
    parser.add_argument('--patterns', type=str, default="", help="")

    parser.add_argument('--env', type=str, required=True, help="")

    parser.add_argument('--num_models', type=int, default=1, help="")
    parser.add_argument('--spec', type=str, required=True, help="")

    args = parser.parse_args()

    # Initialize
    # sample_args: SampleArgs = SampleArgs(args.num_samp, args.samp_astar_weight, args.samp_astar_batch_size,
    #                                     args.samp_max_astar_itr, args.samp_astar_v)

    env: EnvGrndAtoms = cast(EnvGrndAtoms, env_utils.get_environment(args.env))
    # viz_utils.visualize_examples(env, env.get_start_states(4))

    # add to bk
    bk: List[str] = init_bk(env, args.patterns)

    asp_spec: ASPSpec = ASPSpec(env.get_ground_atoms(), bk)

    bk_file_name: str = args.bk_file
    with open(bk_file_name, "w") as bk_file:
        bk_file.write('\n'.join(bk))

    print("Getting models")
    start_time = time.time()
    spec_clauses_str = args.spec.split(";")
    print(spec_clauses_str)
    clauses: List[Clause] = []
    for clause_str in spec_clauses_str:
        clause = parse_clause(clause_str)[0]
        print(clause)
        clauses.append(clause)
    print(clauses)
    goals: List[Model] = asp_spec.get_models(clauses, env.on_model, minimal=True, num_models=args.num_models)
    print(f"{len(goals)} model(s) found, Time: {time.time() - start_time}")
    print(goals)

    viz_utils.visualize_examples(env, goals)


if __name__ == "__main__":
    main()
