from typing import Tuple, Dict, Any, List, Optional

import torch

from deepxube.utils.command_line_utils import get_name_args
from deepxube.base.updater import Update
from deepxube.base.trainer import Train
from deepxube.base.factory import Factory

trainer_factory: Factory[Train] = Factory[Train]("Train")


def get_trainer_from_args(nnet_dir: str, updater: Update, device: torch.device, on_gpu: bool, trainer_name_args: str) -> Tuple[Train, str]:
    trainer_name_pre, args_str = get_name_args(trainer_name_args)

    names: List[str] = trainer_factory.get_all_class_names()
    compat_names: List[str] = []
    incompat_reasons: List[str] = []
    for name in names:
        if not name.startswith(trainer_name_pre):
            continue

        incompat_reason: Optional[str] = trainer_factory.get_type(name).get_incompat_reason(updater)
        if incompat_reason is not None:
            incompat_reasons.append(incompat_reason + f" (Trainer name: {name})")
        else:
            compat_names.append(name)

    if len(compat_names) == 0:
        incompat_reasons_str: str = '\n'.join(incompat_reasons)
        raise ValueError(f"Could not find any compatable Trainer for Trainer "
                         f"name: {trainer_name_pre}.\nIncompatibility reasons:\n{incompat_reasons_str}")

    if len(compat_names) > 1:
        raise ValueError(f"More then 1 compatable Trainer for Trainer name: {trainer_name_pre}: {compat_names}.")

    assert len(compat_names) == 1
    trainer_name: str = compat_names[0]

    trainer_kwargs: Dict[str, Any] = trainer_factory.get_kwargs(trainer_name, args_str)
    trainer_kwargs["nnet_dir"] = nnet_dir
    trainer_kwargs["updater"] = updater
    trainer_kwargs["device"] = device
    trainer_kwargs["on_gpu"] = on_gpu
    return trainer_factory.build_class(trainer_name, trainer_kwargs), trainer_name
