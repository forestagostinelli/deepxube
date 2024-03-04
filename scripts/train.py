from argparse import ArgumentParser
from deepxube.environments.environment_abstract import Environment
from deepxube.training import avi
from deepxube.utils import env_select


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--nnet_dir', type=str, required=True, help="")
    parser.add_argument('--batch_size', type=int, default=1000, help="")
    parser.add_argument('--itrs_per_update', type=int, default=5000000, help="")
    parser.add_argument('--max_itrs', type=int, default=1000000, help="")
    parser.add_argument('--greedy_step_update_max', type=int, default=30, help="")
    parser.add_argument('--num_update_procs', type=int, default=48, help="")
    parser.add_argument('--display', type=int, default=100, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    args = parser.parse_args()

    env: Environment = env_select.get_environment(args.env)
    avi.train(env, args.step_max, args.nnet_dir, batch_size=args.batch_size, itrs_per_update=args.itrs_per_update,
              max_itrs=args.max_itrs, greedy_update_step_max=args.greedy_step_update_max,
              num_update_procs=args.num_update_procs, display=args.display, debug=args.debug)


if __name__ == "__main__":
    main()
