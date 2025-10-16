from argparse import ArgumentParser
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.base.updater import UpHeurArgs, UpdateHeur
from deepxube.updater.updaters import UpdateHeurBWASEnum, UpdateHeurBWQSEnum


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--heur_type', type=str, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--nnet_dir', type=str, required=True, help="")

    # train args
    parser.add_argument('--batch_size', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=0.001, help="")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="")
    parser.add_argument('--max_itrs', type=int, default=1000000, help="")
    parser.add_argument('--balance', action='store_true', default=False, help="")
    parser.add_argument('--display', type=int, default=100, help="")

    # updater args
    parser.add_argument('--up_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_gen_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_procs', type=int, default=1, help="")
    parser.add_argument('--up_search_itrs', type=int, default=200, help="")
    parser.add_argument('--up_batch_size', type=int, default=1000, help="")
    parser.add_argument('--up_nnet_batch_size', type=int, default=10000, help="")

    parser.add_argument('--temp', type=float, default=1.0, help="")

    # other
    parser.add_argument('--rb', type=int, default=1, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    args = parser.parse_args()

    up_args: UpHeurArgs = UpHeurArgs(args.up_itrs, args.up_gen_itrs, args.up_procs, args.up_search_itrs,
                                     args.up_batch_size, args.up_nnet_batch_size)
    updater: UpdateHeur
    if (args.env == "cube3") or (args.env == "cube3_fixed"):
        from deepxube.implementations.cube3 import Cube3
        env = Cube3(args.env == "cube3_fixed")
        if args.heur_type.upper() == "V":
            from deepxube.implementations.cube3 import Cube3NNetParV
            updater = UpdateHeurBWASEnum(env, up_args, Cube3NNetParV())
        elif args.heur_type.upper() == "Q":
            from deepxube.implementations.cube3 import Cube3NNetParQFixOut
            """
            from deepxube.nnet.nnet_utils import get_device, to_pytorch_input
            device = get_device()[0]
            heur_nnet = Cube3NNetParQFix()
            nnet = heur_nnet.get_nnet()
            nnet.train()
            states, goals = env.get_start_goal_pairs([0, 10, 3, 4, 5])
            actions = env.get_state_actions(states)
            inputs_nnet = heur_nnet.to_np(states, goals, actions)
            inputs_nnet_t = to_pytorch_input(inputs_nnet, device)
            out = nnet(inputs_nnet_t)
            breakpoint()
            """
            updater = UpdateHeurBWQSEnum(env, up_args, Cube3NNetParQFixOut())
        elif args.heur_type.upper() == "QIN":
            from deepxube.implementations.cube3 import Cube3NNetParQIn
            """
            from deepxube.nnet.nnet_utils import get_device, to_pytorch_input
            device = get_device()[0]
            heur_nnet = Cube3NNetParQIn()
            nnet = heur_nnet.get_nnet()
            nnet.train()
            states, goals = env.get_start_goal_pairs([0, 10, 3, 4, 5])
            inputs_nnet = heur_nnet.to_np(states, goals, [[x] for x in env.get_state_action_rand(states)])
            inputs_nnet_t = to_pytorch_input(inputs_nnet, device)
            out = nnet(inputs_nnet_t)
            breakpoint()
            """
            updater = UpdateHeurBWQSEnum(env, up_args, Cube3NNetParQIn())
        else:
            raise ValueError(f"Unknown heur type {args.heur_type}")
    else:
        raise ValueError(f"Unknown environment {args.env}")

    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.balance, args.display)
    train(updater, args.step_max, args.nnet_dir, train_args, rb_past_up=args.rb, debug=args.debug)


if __name__ == "__main__":
    main()
