from typing import Optional
import argparse

from deepxube.factories.updater_factory import get_updater

from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.updater.updaters import UpGraphSearchArgs, UpGreedyPolicyArgs
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train, TestArgs
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg

import os
import pickle


def train_cli(args: argparse.Namespace) -> None:
    # parse domain and heur_nnet
    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, args.heur_type)[0]

    # update args
    up_args: UpArgs = UpArgs(args.procs, args.up_itrs, args.step_max, args.search_itrs,
                             up_batch_size=args.up_batch_size, nnet_batch_size=args.up_nnet_batch_size,
                             sync_main=args.sync_main, v=args.up_v)
    up_heur_args: UpHeurArgs = UpHeurArgs(False, args.backup)
    up_graphsch_args: UpGraphSearchArgs = UpGraphSearchArgs(args.search_weight, args.search_eps)
    up_greedy_args: UpGreedyPolicyArgs = UpGreedyPolicyArgs(args.search_eps, args.search_temp)

    # updater
    updater: UpdateHeur = get_updater(domain, heur_nnet, args.search, up_args, up_heur_args, up_graphsch_args,
                                      up_greedy_args)

    # train args
    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, not args.no_bal,
                                      rb=args.rb,
                                      display=args.display)

    # test args
    valid_file: str = f"data/{args.domain}/valid.pkl"
    test_args: Optional[TestArgs]
    if os.path.isfile(valid_file):
        states, goals = pickle.load(open(valid_file, "rb"))
        test_args = TestArgs(states, goals, args.t_search_itrs, [0.0], args.up_nnet_batch_size, args.t_up_freq,
                             False)
    else:
        test_args = None

    # test args
    train(updater, args.dir, train_args, test_args=test_args, debug=args.debug)