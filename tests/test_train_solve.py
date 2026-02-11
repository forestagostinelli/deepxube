from typing import List, Optional
import pytest  # type: ignore

from deepxube.factories.updater_factory import get_updater

from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.pathfinding import Node, Instance, get_path
from deepxube.pathfinding.utils.performance import is_valid_soln, PathFindPerf
from deepxube.base.pathfinding import PathFind, PathFindHeur
from deepxube.utils.command_line_utils import get_domain_from_arg, get_heur_nnet_par_from_arg, get_pathfind_name_kwargs, get_pathfind_from_arg
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.nnet import nnet_utils
from itertools import product
import shutil
from torch import nn
import os


pathfind_v_l: List[str] = ["bwas", "greedy_v"]
cases = (
    [pytest.param(a, b, c, d, e, f, g, 85.0, id="bwas") for a, b, c, d, e, f, g in
     product(["bwas"], ["bwas"], ["V"], [True, False], [False], [1, -1], [False])]

    + [pytest.param(a, b, c, d, e, f, g, 85.0, id="greedy_v") for a, b, c, d, e, f, g in
       product(["greedy_v"], ["greedy_v"], ["V"], [True, False], [True, False], [1], [False])]

    + [pytest.param(a, b, c, d, e, f, g, 85.0, id="sup_v_rw") for a, b, c, d, e, f, g in
       product(["sup_v_rw"], ["bwas"], ["V"], [False], [False], [1], [False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="bwqs") for a, b, c, d, e, f, g in
       product(["bwqs"], ["bwqs"], ["QFix", "QIn"], [True, False], [False], [1, -1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="greedy_q") for a, b, c, d, e, f, g in
       product(["greedy_q"], ["greedy_q"], ["QFix", "QIn"], [True, False], [True, False], [1], [True, False])]

    + [pytest.param(a, b, c, d, e, f, g, 80.0, id="sup_q_rw") for a, b, c, d, e, f, g in
       product(["sup_q_rw"], ["bwqs"], ["QFix", "QIn"], [False], [False], [1], [False])]
)


@pytest.mark.parametrize("pathfind_tr_str,pathfind_solve_str,heur_type,bal,ub_heur_solns,backup,sync_main,soln_thresh", cases)
def test_train_solve_heur(pathfind_tr_str: str, pathfind_solve_str: str, heur_type: str, bal: bool, ub_heur_solns: bool, backup: int, sync_main: bool,
                          soln_thresh: float) -> None:
    domain_str: str = "grid.7"
    heur_str: str = "resnet_fc.100H_1B_bn"
    search_itrs: int = 20
    domain, domain_name = get_domain_from_arg(domain_str)
    heur_nnet_par: HeurNNetPar = get_heur_nnet_par_from_arg(domain, domain_name, heur_str, heur_type)[0]
    pathfind_name, pathfind_kwargs = get_pathfind_name_kwargs(pathfind_tr_str)

    # update args
    up_args: UpArgs = UpArgs(1, 100, 100, search_itrs, sync_main=sync_main)
    up_heur_args: UpHeurArgs = UpHeurArgs(ub_heur_solns, backup)

    # updater
    updater: UpdateHeur = get_updater(domain, heur_nnet_par, pathfind_name, pathfind_kwargs, up_args, up_heur_args, False)

    # train args
    train_args: TrainArgs = TrainArgs(50, 0.001, 0.9999993, 2000, bal, display=0)

    # train
    save_dir: str = "tests/dummy/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    train(heur_nnet_par, heur_type, updater, save_dir, train_args)

    # solve
    heur_file: str = f"{save_dir}/heur.pt"
    pathfind: PathFind = get_pathfind_from_arg(domain, heur_type, pathfind_solve_str)[0]
    assert isinstance(pathfind, PathFindHeur), f"Current implementation only uses {PathFindHeur}"
    device, devices, on_gpu = nnet_utils.get_device()

    nnet: nn.Module = nnet_utils.load_nnet(heur_file, heur_nnet_par.get_nnet())
    nnet.eval()
    nnet.to(device)
    nnet = nn.DataParallel(nnet)
    heur_fn = heur_nnet_par.get_nnet_fn(nnet, None, device, None)

    pathfind.set_heur_fn(heur_fn)

    # do pathfinding
    states, goals = domain.sample_start_goal_pairs(list(range(0, 100)))
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
            path_states, path_actions, path_cost = get_path(goal_node)
            assert is_valid_soln(instance.root_node.state, instance.root_node.goal, path_actions, domain)

    print(pathfind_perf.to_string())
    per_solved: float = pathfind_perf.per_solved()
    assert per_solved >= soln_thresh, f"Should solve at least 90%, but solved {per_solved}"
