from deepxube.base.env import Env
from deepxube.nnet.nnet_utils import NNetPar
from deepxube.tests.test_env import test
from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--heur_type', type=str, required=True, help="")
    parser.add_argument('--num_states', type=int, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    args = parser.parse_args()

    if (args.env == "cube3") or (args.env == "cube3_fixed"):
        from deepxube.implementations.cube3 import Cube3
        env: Env = Cube3(args.env == "cube3_fixed")
        heur_nnet: NNetPar
        if args.heur_type.upper() == "V":
            from deepxube.implementations.cube3 import Cube3NNetParV
            heur_nnet = Cube3NNetParV()
        elif args.heur_type.upper() == "Q":
            from deepxube.implementations.cube3 import Cube3NNetParQFixOut
            heur_nnet = Cube3NNetParQFixOut()
        elif args.heur_type.upper() == "QIN":
            from deepxube.implementations.cube3 import Cube3NNetParQIn
            heur_nnet = Cube3NNetParQIn()
        else:
            raise ValueError(f"Unknown heur type {args.heur_type}")

        test(env, heur_nnet, args.num_states, args.step_max)
    else:
        raise ValueError(f"Unknown environment {args.env}")

if __name__ == "__main__":
    main()
