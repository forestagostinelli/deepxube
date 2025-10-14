from deepxube.tests.test_env import test_env
from deepxube.base.env import Env
from deepxube.implementations.env_utils import get_environment
from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--num_states', type=int, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    args = parser.parse_args()

    env: Env = get_environment(args.env)
    test_env(env, args.num_states, args.step_max)


if __name__ == "__main__":
    main()
