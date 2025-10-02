from argparse import ArgumentParser
from deepxube.environments.environment_abstract import Environment
from deepxube.training.train_utils import TrainArgs
from deepxube.training.davi import train
from deepxube.update.update_davi import UpdateArgs
from deepxube.environments.env_utils import get_environment


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--nnet_dir', type=str, required=True, help="")

    # train args
    parser.add_argument('--batch_size', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=0.001, help="")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="")
    parser.add_argument('--max_itrs', type=int, default=1000000, help="")
    parser.add_argument('--display', type=int, default=100, help="")

    # update args
    parser.add_argument('--up_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_gen_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_procs', type=int, default=1, help="")
    parser.add_argument('--up_search_itrs', type=int, default=200, help="")
    parser.add_argument('--up_batch_size', type=int, default=1000, help="")
    parser.add_argument('--up_nnet_batch_size', type=int, default=10000, help="")

    # other
    parser.add_argument('--rb', type=int, default=1, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    args = parser.parse_args()

    env: Environment = get_environment(args.env)
    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.display)
    up_args: UpdateArgs = UpdateArgs(args.up_itrs, args.up_gen_itrs, args.up_procs, args.up_search_itrs,
                                     args.up_batch_size, args.up_nnet_batch_size)
    train(env, args.step_max, args.nnet_dir, train_args, up_args, rb_past_up=args.rb, debug=args.debug)


if __name__ == "__main__":
    main()
