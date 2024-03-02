from argparse import ArgumentParser

from deepxube.environments.environment_abstract import Environment
from deepxube.tests.test_env import test_env
from deepxube.utils import env_select

import time


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--num_states', type=int, default=100, help="")
    parser.add_argument('--step_max', type=int, default=30, help="")

    args = parser.parse_args()

    # get environment
    start_time = time.time()
    env: Environment = env_select.get_environment(args.env)

    elapsed_time = time.time() - start_time
    print(f"Initialized environment {env.env_name} in %s seconds" % elapsed_time)

    test_env(env, args.num_states, args.step_max)


if __name__ == "__main__":
    main()
