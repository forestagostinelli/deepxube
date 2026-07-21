from typing import List, Optional, Tuple
import pytest  # type: ignore

from deepxube.base.domain import Domain, State, Goal
from deepxube.domains.cube3 import Cube3
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


@pytest.fixture
def prob_insts() -> Tuple[List[State], List[Goal]]:
    states, goals = Cube3().sample_problem_instances(([0] * 3) + ([1] * 3) + ([2] * 3))
    return states, goals


cases = (
    [pytest.param(a, b, id="static") for a, b in
     product(["heurv", "heurq_fixout", "heurq_in"], ["graph", "graph.10B", "beam.1000B"])]
)


@pytest.mark.parametrize("fn_str,pathfind_solve_str", cases)
def test_bruteforce(fn_str: str, pathfind_solve_str: str, prob_insts: Tuple[List[State], List[Goal]]) -> None:
    device, devices, on_gpu = nnet_utils.get_device()

    domain: Domain = Cube3()
    domain_name: str = "cube3"
    fn_l: List[str] = [f"{fn_str}"]

    pathfind_fns = get_path_fns_nnet_par_dict(domain, domain_name, fn_l, device)[0]
    pathfind = get_pathfind_from_arg(domain, pathfind_fns, pathfind_solve_str)[0]

    states: List[State] = prob_insts[0]
    goals: List[Goal] = prob_insts[1]
    instances: List[Instance] = pathfind.make_instances(states, goals, None, True)
    pathfind.add_instances(instances)
    for _ in range(200):
        pathfind.step()

    # get pathfind perf
    pathfind_perf: PathFindPerf = PathFindPerf()
    instance: Instance
    for instance in pathfind.instances:
        pathfind_perf.update_perf(instance)

        # check soln
        goal_node: Optional[Node] = instance.goal_node
        assert goal_node is not None
        path_states, path_actions, _, path_cost = get_path(goal_node)
        assert is_valid_soln(instance.root_node.state, instance.root_node.goal, path_actions, domain)

    print(pathfind_perf.to_string())
    per_solved: float = pathfind_perf.per_solved()
    assert per_solved == 100.0, f"Should solve at least {100}%, but solved {per_solved}"
