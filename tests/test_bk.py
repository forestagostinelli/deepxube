from typing import List, cast, Optional
from deepxube.environments.environment_abstract import EnvGrndAtoms
from deepxube.logic.asp import ASPSpec
from deepxube.logic.logic_objects import Clause, Model
from deepxube.utils import viz_utils
from deepxube.logic import logic_utils
from deepxube.utils import env_select
import time
import argparse


def get_bk(env: EnvGrndAtoms, bk_add_file_name: Optional[str]) -> List[str]:
    bk_init: List[str] = env.get_bk()
    bk_init.append("")

    if bk_add_file_name is not None:
        bk_add_file = open(bk_add_file_name, 'r')
        bk_init.extend(bk_add_file.read().split("\n"))
        bk_add_file.close()

    return bk_init


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bk_add', type=str, default=None, help="If given will append to bk file")
    parser.add_argument('--bk_out', type=str, default=None, help="If given will output bk file")
    parser.add_argument('--env', type=str, required=True, help="")

    parser.add_argument('--num_models', type=int, default=1, help="")
    parser.add_argument('--spec', type=str, required=True, help="")

    args = parser.parse_args()

    # Initialize
    # sample_args: SampleArgs = SampleArgs(args.num_samp, args.samp_astar_weight, args.samp_astar_batch_size,
    #                                     args.samp_max_astar_itr, args.samp_astar_v)

    env: EnvGrndAtoms = cast(EnvGrndAtoms, env_select.get_environment(args.env))
    # viz_utils.visualize_examples(env, env.get_start_states(4))

    # add to bk
    bk: List[str] = get_bk(env, args.bk_add)
    asp_spec: ASPSpec = ASPSpec(env.get_ground_atoms(), bk)

    bk_file_name: Optional[str] = args.bk_out
    if bk_file_name is not None:
        with open(bk_file_name, "w") as bk_file:
            bk_file.write('\n'.join(bk))

    print("Getting models")
    start_time = time.time()
    spec_clauses_str = args.spec.split(";")
    clauses: List[Clause] = []
    for clause_str in spec_clauses_str:
        clause = program_utils.parse_clause(clause_str)[0]
        clauses.append(clause)
    print(clauses)
    goals: List[Model] = asp_spec.get_models(clauses, env.on_model, minimal=True, num_models=args.num_models)
    print(f"{len(goals)} model(s) found, Time: {time.time() - start_time}")
    print(goals)

    viz_utils.visualize_examples(env, goals)


if __name__ == "__main__":
    main()
