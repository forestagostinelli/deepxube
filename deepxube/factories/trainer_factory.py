from typing import Type, List, Tuple, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

from deepxube.nnet.nnet_utils import NNetPar
from deepxube.utils.command_line_utils import get_name_args
from deepxube.base.domain import Domain
from deepxube.base.pathfind_fns import PFNs
from deepxube.base.pathfinding import PathFind
from deepxube.base.updater import Update
from deepxube.base.trainer import Train
from deepxube.base.factory import Factory

trainer_factory: Factory[Train] = Factory[Train]("Train")


def get_trainer_from_args(nnet_dir: str, updater: Update, device: torch.device, on_gpu: bool, writer: SummaryWriter,
                          trainer_name_args: str) -> Tuple[Train, str]:
    trainer_name, args_str = get_name_args(trainer_name_args)

    trainer_kwargs: Dict[str, Any] = trainer_factory.get_kwargs(trainer_name, args_str)
    trainer_kwargs["nnet_dir"] = nnet_dir
    trainer_kwargs["updater"] = updater
    trainer_kwargs["device"] = device
    trainer_kwargs["on_gpu"] = on_gpu
    trainer_kwargs["writer"] = writer
    return trainer_factory.build_class(trainer_name, trainer_kwargs), trainer_name
