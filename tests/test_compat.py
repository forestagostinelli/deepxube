import pytest  # type: ignore

from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.factories.pathfind_fns_factory import get_path_up_fns
from deepxube.factories.pathfinding_factory import get_pathfind_from_arg
from deepxube.factories.updater_factory import get_updater_from_args
from deepxube.factories.trainer_factory import get_trainer_from_args
from deepxube.pytorch import nnet_utils
from itertools import product
import shutil
import os


cases_compat = (
        [pytest.param(a, b, c, d, id="rl_her") for a, b, c, d in
         product(["heurv", "heurq_fixout", "heurq_in"], ["graph", "beam"], ["up_rl", "up_her"], ["tr_h"])]

        + [pytest.param(a, b, c, d, id="sup_v") for a, b, c, d in product(["heurv"], ["sup_v"], ["up_sup"], ["tr_h"])]
        + [pytest.param(a, b, c, d, id="sup_q") for a, b, c, d in product(["heurq_fixout", "heurq_in"], ["sup_q"], ["up_sup"], ["tr_h"])]
)


@pytest.mark.parametrize("fn_str,pathfind_str,up_str,tr_str", cases_compat)
def test_train_compat(fn_str: str, pathfind_str: str, up_str: str, tr_str: str) -> None:
    domain_str: str = "grid.7d"
    nnet_name_args: str = "resnet_fc.100H_1B_bn"
    device, devices, on_gpu = nnet_utils.get_device()
    save_dir: str = "tests/dummy/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    domain, domain_name = get_domain_from_arg(domain_str)
    pathfind_fns, updater_fns = get_path_up_fns(domain, domain_name, [f"{fn_str},{nnet_name_args}"], device)
    pathfind, pathfind_name, pathfind_name_args_full = get_pathfind_from_arg(domain, pathfind_fns, pathfind_str)
    updater, updater_name = get_updater_from_args(domain, pathfind, pathfind_name_args_full, updater_fns, up_str)
    trainer, trainer_name = get_trainer_from_args(save_dir, updater, device, on_gpu, tr_str)

    assert (trainer is not None) and (trainer_name is not None)
