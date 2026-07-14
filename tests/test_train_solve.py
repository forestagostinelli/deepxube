from typing import List, Optional, Any
import pytest  # type: ignore

from torch.utils.tensorboard import SummaryWriter

from deepxube.base.pathfinding import Node, Instance, get_path
from deepxube.pathfinding.utils.performance import is_valid_soln, PathFindPerf
from deepxube.base.pathfinding import PathFind
from deepxube.base.pathfind_fns import DeepXubeNNetPar, PFNs, PFNsHeurV, PFNsHeurQ
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.factories.pathfind_fns_factory import get_fn_dicts, pathfind_fns_factory
from deepxube.factories.pathfinding_factory import get_pathfind_from_arg
from deepxube.factories.updater_factory import get_updater_from_args
from deepxube.factories.trainer_factory import get_trainer_from_args
from deepxube.base.updater import UpArgs, Update, UpdateHeur
from deepxube.base.trainer import TrainArgs
from deepxube.pytorch import nnet_utils
from itertools import product
import shutil
from torch import nn
import os


cases_compat = (
    [pytest.param(a, b, c, d, id="rl_her") for a, b, c, d in
     product(["heurv", "heurq_fixout", "heurq_in"], ["graph", "beam"], ["up_rl", "up_her"], ["tr_h"])]

    + [pytest.param(a, b, c, d, id="rl_her") for a, b, c, d in product(["heurv"], ["sup_v"], ["up_sup"], ["tr_h"])]
    + [pytest.param(a, b, c, d, id="rl_her") for a, b, c, d in product(["heurq_fixout", "heurq_in"], ["sup_q"], ["up_sup"], ["tr_h"])]
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
    nnet_fn_dict, nnet_par_dict = get_fn_dicts(domain, domain_name, [f"{fn_str},{nnet_name_args}"], device)
    pathfind_fns: PFNs = pathfind_fns_factory.build_class(nnet_fn_dict)
    pathfind, pathfind_name, pathfind_name_args_full = get_pathfind_from_arg(domain, pathfind_fns, pathfind_str)
    updater, updater_name = get_updater_from_args(domain, pathfind, pathfind_name_args_full, nnet_par_dict, up_str)

    writer: SummaryWriter = SummaryWriter(save_dir)
    nnet_par_train: DeepXubeNNetPar = updater.get_train_nnet_par()

    trainer, trainer_name = get_trainer_from_args(save_dir, nnet_par_train, updater, device, on_gpu, writer, tr_str)

    assert (trainer is not None) and (trainer_name is not None)


cases = (
    [pytest.param(a, b, c, d, e, f, g, 85.0, id="graph_v") for a, b, c, d, e, f, g in
     product(["graph_v"], ["graph_v"], ["V"], [True, False], [False], [1, -1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 85.0, id="beam_v") for a, b, c, d, e, f, g in
       product(["beam_v.1T"], ["beam_v"], ["V"], [True, False], [True, False], [1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 85.0, id="sup_v") for a, b, c, d, e, f, g in
       product(["sup_v"], ["graph_v"], ["V"], [False], [False], [1], [False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="graph_q") for a, b, c, d, e, f, g in
       product(["graph_q"], ["graph_q"], ["QFix", "QIn"], [True, False], [False], [1, -1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="beam_q") for a, b, c, d, e, f, g in
       product(["beam_q.1T"], ["beam_q"], ["QFix", "QIn"], [True, False], [True, False], [1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="sup_q") for a, b, c, d, e, f, g in
       product(["sup_q"], ["graph_q"], ["QFix", "QIn"], [False], [False], [1], [False])]
)


@pytest.mark.parametrize("pathfind_tr_str,pathfind_solve_str,heur_type,bal,ub_heur_solns,backup,sync_main,soln_thresh", cases)
def test_train_solve_heur(pathfind_tr_str: str, pathfind_solve_str: str, heur_type: str, bal: bool, ub_heur_solns: bool, backup: int, sync_main: bool,
                          soln_thresh: float) -> None:
    domain_str: str = "grid.7d"
    heur_str: str = "resnet_fc.100H_1B_bn"
    search_itrs: int = 20
    domain, domain_name = get_domain_from_arg(domain_str)
    heur_nnet_par: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, heur_str, heur_type)[0]

    # update args
    up_args: UpArgs = UpArgs(1, 100, 100, search_itrs, ub_heur_solns=ub_heur_solns, backup=backup, sync_main=sync_main)

    # updater
    updater_ret: Update = get_updater(domain, pathfind_tr_str, up_args, False, "heur")
    assert isinstance(updater_ret, UpdateHeur)
    updater: UpdateHeur = updater_ret

    # train args
    rb: int = 0
    if sync_main:
        rb = 1
    train_args: TrainArgs = TrainArgs(50, 2000, bal, rb=rb, display=0)

    # train
    save_dir: str = "tests/dummy/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    train(domain, heur_nnet_par, updater, None, None, 10, save_dir, train_args)

    # solve
    heur_file: str = f"{save_dir}/heur.pt"
    device, devices, on_gpu = nnet_utils.get_device()

    nnet: nn.Module = nnet_utils.load_nnet(heur_file, heur_nnet_par.get_nnet())
    nnet.eval()
    nnet.to(device)
    nnet = nn.DataParallel(nnet)
    heur_fn = heur_nnet_par.get_nnet_fn(nnet, None, device, None)

    # do pathfinding
    functions: Any
    if heur_type == "V":
        assert isinstance(heur_fn, HeurFnV)
        functions = PFNsHeurV(heur_fn)
    else:
        assert isinstance(heur_fn, HeurFnQ)
        functions = PFNsHeurQ(heur_fn)

    pathfind: PathFind = get_pathfind_from_arg(domain, functions, pathfind_solve_str)[0]
    states, goals = domain.sample_problem_instances(list(range(0, 100)))
    instances: List[Instance] = pathfind.make_instances(states, goals, None, True)
    pathfind.add_instances(instances)
    for _ in range(search_itrs):
        pathfind.step()

    # get pathfind perf
    pathfind_perf: PathFindPerf = PathFindPerf()
    instance: Instance
    for instance in pathfind.instances:
        pathfind_perf.update_perf(instance)

        # check soln
        goal_node: Optional[Node] = instance.goal_node
        if goal_node is not None:
            path_states, path_actions, _, path_cost = get_path(goal_node)
            assert is_valid_soln(instance.root_node.state, instance.root_node.goal, path_actions, domain)

    print(pathfind_perf.to_string())
    per_solved: float = pathfind_perf.per_solved()
    assert per_solved >= soln_thresh, f"Should solve at least 90%, but solved {per_solved}"
