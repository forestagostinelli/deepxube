### USAGE:
# python train_heur.py --heur_type V --step_max 100 --nnet_dir models/cube3_v/ --batch_size 10000 --up_itrs 1000 --up_gen_itrs 1000 --up_procs 48 --up_search_itrs 200
# python train_heur.py --heur_type Q --step_max 100 --nnet_dir models/cube3_q/ --batch_size 10000 --up_itrs 1000 --up_gen_itrs 1000 --up_procs 48 --up_search_itrs 200
# python train_heur.py --heur_type QIn --step_max 100 --nnet_dir models/cube3_qin/ --batch_size 10000 --up_itrs 1000 --up_gen_itrs 1000 --up_procs 48 --up_search_itrs 200

# python train_heur.py --heur_type V --step_max 100 --nnet_dir models/cube3_v_sup/ --batch_size 10000 --up_itrs 1000 --up_gen_itrs 1000 --up_procs 48 --up_search_itrs 200 --sup
from argparse import ArgumentParser

from deepxube.base.heuristic import HeurNNetQ
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.base.updater import UpArgs, UpdateHeur
from deepxube.updater.updaters import UpdateHeurBWASEnum, UpdateHeurBWQSEnum, UpdateHeurGrPolQEnum, UpdateHeurStepLenSup
from deepxube.implementations.cube3 import Cube3


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--heur_type', type=str, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--nnet_dir', type=str, required=True, help="")
    parser.add_argument('--sup', action='store_true', default=False, help="")
    parser.add_argument('--greedy', action='store_true', default=False, help="")

    # train args
    parser.add_argument('--batch_size', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=0.001, help="")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="")
    parser.add_argument('--max_itrs', type=int, default=1000000, help="")
    parser.add_argument('--balance', action='store_true', default=False, help="")
    parser.add_argument('--targ_up_searches', type=int, default=0, help="")
    parser.add_argument('--display', type=int, default=100, help="")

    # updater args
    parser.add_argument('--up_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_gen_itrs', type=int, default=1000, help="")
    parser.add_argument('--up_procs', type=int, default=1, help="")
    parser.add_argument('--up_search_itrs', type=int, default=200, help="")
    parser.add_argument('--up_batch_size', type=int, default=100, help="")
    parser.add_argument('--up_nnet_batch_size', type=int, default=10000, help="")

    parser.add_argument('--backup', type=int, default=1, help="")

    parser.add_argument('--temp', type=float, default=1, help="")
    parser.add_argument('--eps', type=float, default=0.0, help="")

    # other
    parser.add_argument('--rb', type=int, default=1, help="")
    parser.add_argument('--debug', action='store_true', default=False, help="")
    args = parser.parse_args()

    up_args: UpArgs = UpArgs(args.up_itrs, args.up_gen_itrs, args.up_procs, args.up_search_itrs,
                             args.up_batch_size, args.up_nnet_batch_size)
    updater: UpdateHeur
    env = Cube3(True)
    if args.heur_type.upper() == "V":
        from deepxube.implementations.cube3 import Cube3NNetParV
        if args.sup:
            updater = UpdateHeurStepLenSup(env, up_args, Cube3NNetParV())
        else:
            updater = UpdateHeurBWASEnum(env, up_args, False, args.backup, Cube3NNetParV(), eps=args.eps)
    elif (args.heur_type.upper() == "Q") or (args.heur_type.upper() == "QIN"):
        nnet_par: HeurNNetQ
        if args.heur_type.upper() == "Q":
            from deepxube.implementations.cube3 import Cube3NNetParQFixOut
            nnet_par = Cube3NNetParQFixOut()
        elif args.heur_type.upper() == "QIN":
            from deepxube.implementations.cube3 import Cube3NNetParQIn
            nnet_par = Cube3NNetParQIn()
        else:
            raise ValueError("")

        if args.greedy:
            updater = UpdateHeurGrPolQEnum(env, up_args, True, nnet_par, args.temp, args.eps)
        else:
            updater = UpdateHeurBWQSEnum(env, up_args, True, nnet_par, args.eps)
    else:
        raise ValueError(f"Unknown heur type {args.heur_type}")

    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.balance,
                                      args.targ_up_searches, args.display)
    train(updater, args.step_max, args.nnet_dir, train_args, rb_past_up=args.rb, debug=args.debug)


if __name__ == "__main__":
    main()
