from typing import Dict, List
import argparse
from argparse import ArgumentParser

import torch

from deepxube.utils.command_line_utils import get_name_args, print_command
from deepxube.nnet.nnet_utils import NNetCallable, NNetPar, get_device
from deepxube.base.pathfind_fns import PFNs, DeepXubeNNetPar
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.factories.pathfind_fns_factory import get_dx_nnet_par, pathfind_fns_factory
from deepxube.factories.pathfinding_factory import get_pathfind_from_arg
from deepxube.factories.updater_factory import get_updater_from_args
from deepxube.factories.trainer_factory import get_trainer_from_args
from deepxube.utils.data_utils import Logger
from torch.utils.tensorboard import SummaryWriter

import os
import sys


def parser_train(parser: ArgumentParser) -> None:
    # domain
    parser.add_argument('--domain', type=str, required=True, help="Domain name and arguments.")

    # functions and corresponding nnets
    parser.add_argument('--fn', type=str, nargs='*', help="Function and neural network arguments separated by a comma.")

    # pathfinding
    parser.add_argument('--pathfind', type=str, required=True, help="Pathfinding algorithm and arguments. Batch size of any pathfinding algorithm should be 1 "
                                                                    "since updater assumes 1 instance is generated per iteration.")

    # updater
    parser.add_argument('--up', type=str, required=True, help="Updater algorithm and arguments.")

    # train args
    parser.add_argument('--tr', type=str, required=True, help="Trainer algorithm and arguments.")
    parser.add_argument('--dir', type=str, required=True, help="Directory to save neural networks.")

    # test args
    test_group = parser.add_argument_group('test')
    test_group.add_argument('--t_file', type=str, default=None, help="File to use when testing.")
    test_group.add_argument('--t_search_itrs', type=int, default=100, help="Number of search iterations when testing.")
    test_group.add_argument('--t_up_freq', type=int, default=10, help="Test every t_up_freq updates.")
    test_group.add_argument('--t_pathfinds', type=str, default="bwas", help="Comma separated list of pathfinding algorithms to use when testing.")
    test_group.add_argument('--t_init', action='store_true', default=False, help="Set for testing before training begins.")

    # other
    parser.add_argument('--debug', action='store_true', default=False, help="Set for debug mode.")
    parser.set_defaults(func=train_cli)


def train_cli(args: argparse.Namespace) -> None:
    # logging
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if not args.debug:
        output_save_loc = f"{args.dir}/output.txt"
        sys.stdout = Logger(output_save_loc, "a")

    print_command()

    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM JOB ID: %s" % os.environ['SLURM_JOB_ID'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # parse domain
    domain, domain_name = get_domain_from_arg(args.domain)

    # parse nnet fn args
    nnet_fn_dict: Dict[str, NNetCallable] = dict()
    nnet_par_dict: Dict[str, NNetPar] = dict()
    for fn_arg in args.fn:
        fn_arg_split: List[str] = fn_arg.split(",")
        assert len(fn_arg_split) == 2
        nnet_par_name_args, nnet_name_args = fn_arg_split[0], fn_arg_split[1]
        nnet_par, nnet_par_name = get_dx_nnet_par(domain, domain_name, nnet_par_name_args, nnet_name_args)

        print(nnet_par)
        print(f"(name: {nnet_par_name}, nnet_input_name: {nnet_par.nnet_input_name})")

        field_name: str = nnet_par.get_field_name()
        nnet_fn_dict[field_name] = nnet_par.get_nnet_fn(nnet_par.get_nnet(), None, device, None)
        nnet_par_dict[field_name] = nnet_par

    print(domain, f"(name: {domain_name})")

    # pathfind functions
    pathfind_fns: PFNs = pathfind_fns_factory.build_class(nnet_fn_dict)
    print(pathfind_fns)

    # pathfinding
    pathfind, pathfind_name = get_pathfind_from_arg(domain, pathfind_fns, args.pathfind)
    print(pathfind, f"(name: {pathfind_name})")

    _, pathfind_args_str = get_name_args(args.pathfind)
    pathfind_name_args: str = f"{pathfind_name}"
    if pathfind_args_str is not None:
        pathfind_name_args = f"{pathfind_name_args}.{pathfind_args_str}"

    # updater
    updater, updater_name = get_updater_from_args(domain, pathfind, pathfind_name_args, nnet_par_dict, args.up)
    print(updater, f"(name: {updater_name})")

    """
    # test args
    test_args: Optional[TestArgs] = None
    if args.t_file is not None:
        data = pickle.load(open(args.t_file, "rb"))
        states: List[State] = data['states']
        goals: List[Goal] = data['goals']
        test_args = TestArgs(states, goals, args.t_search_itrs, args.t_pathfinds.split(","), args.up_nnet_batch_size, args.t_up_freq, args.t_init)
        print(f"{test_args}")
    """

    writer: SummaryWriter = SummaryWriter(args.dir)

    nnet_par_train: DeepXubeNNetPar = updater.get_train_nnet_par()
    trainer, trainer_name = get_trainer_from_args(args.dir, nnet_par_train, updater, device, on_gpu, writer, args.tr)
    print(trainer, f"(name: {trainer_name})")

    trainer.train_loop()
