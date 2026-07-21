from typing import List, Optional
import pytest  # type: ignore

from deepxube.base.pathfinding import Node, Instance, get_path
from deepxube.pathfinding.utils.performance import is_valid_soln, PathFindPerf
from deepxube.factories.domain_factory import get_domain_from_arg
from deepxube.factories.pathfind_fns_factory import get_path_up_fns, get_path_fns_nnet_par_dict
from deepxube.factories.pathfinding_factory import get_pathfind_from_arg
from deepxube.factories.updater_factory import get_updater_from_args
from deepxube.factories.trainer_factory import get_trainer_from_args
from deepxube.pytorch import nnet_utils
from itertools import product
import shutil
import os


cases = (
    [pytest.param(a, b, c, d, id="rl") for a, b, c, d in
     product(["heurv", "heurq_fixout", "heurq_in"], ["graph", "beam.1T"],
             ["up_rl.100up_100sm_20sitrs_4p_lhbl", "up_rl.100up_100sm_20sitrs_4p_sync_1rb", "up_her.100up_100sm_20sitrs_4p_1rb"],
             ["tr_h.50bs_2000maxit", "tr_h.50bs_2000maxit_bal"])]

    + [pytest.param(a, b, c, d, id="sup_v") for a, b, c, d in product(["heurv"], ["sup_v"],  ["up_sup.100up_100sm_1sitrs_4p"], ["tr_h.50bs_2000maxit"])]
    + [pytest.param(a, b, c, d, id="sup_q") for a, b, c, d in product(["heurq_fixout", "heurq_in"], ["sup_q"], ["up_sup.100up_100sm_1sitrs_4p"],
                                                                      ["tr_h.50bs_2000maxit"])]
)


@pytest.mark.parametrize("fn_str,pathfind_tr_str,up_str,tr_str", cases)
def test_train_solve_heur(fn_str: str, pathfind_tr_str: str, up_str: str, tr_str: str) -> None:
    device, devices, on_gpu = nnet_utils.get_device()

    save_dir: str = "tests/dummy/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    domain_str: str = "grid.7d"
    nnet_name_args: str = "resnet_fc.100H_1B_bn"
    fn_l: List[str] = [f"{fn_str},{nnet_name_args}"]

    # train
    domain, domain_name = get_domain_from_arg(domain_str)
    pathfind_fns, updater_fns = get_path_up_fns(domain, domain_name, fn_l, device)
    pathfind, pathfind_name, pathfind_name_args_full = get_pathfind_from_arg(domain, pathfind_fns, pathfind_tr_str)
    updater, updater_name = get_updater_from_args(domain, pathfind, pathfind_name_args_full, updater_fns, up_str)
    trainer, trainer_name = get_trainer_from_args(save_dir, updater, device, on_gpu, tr_str)

    trainer.train_loop()

    # solve
    search_itrs: int = 20
    for pathfind_solve_str, soln_thresh in [("graph", 85), ("beam.10B", 80)]:
        heur_file: str = f"{save_dir}/heur.pt"
        pathfind_fns = get_path_fns_nnet_par_dict(domain, domain_name, fn_l, device, nnet_files=[heur_file])[0]
        pathfind = get_pathfind_from_arg(domain, pathfind_fns, pathfind_solve_str)[0]
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
        assert per_solved >= soln_thresh, f"Should solve at least {soln_thresh}%, but solved {per_solved}"
