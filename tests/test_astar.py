from typing import List
import pytest
from deepxube.environments.environment_abstract import Environment
from deepxube.search.astar import AStar, get_path
from deepxube.environments.env_utils import get_environment
from deepxube.nnet.nnet_utils import get_heuristic_fn, get_device, HeurFN_T
from deepxube.search.search_utils import is_valid_soln

import numpy as np

env_names: List[str] = ["cube3", "puzzle15"]
num_steps_l_l: List[List[int]] = [[0], [1], [2], [0, 1, 2]]
batch_sizes: List[int] = [1, 10]


@pytest.mark.parametrize("env_name", env_names)
@pytest.mark.parametrize("num_steps_l", num_steps_l_l)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_search(env_name: str, num_steps_l: List[int], batch_size: int):
    # get instances
    env: Environment = get_environment(env_name)
    states, goals = env.get_start_goal_pairs(num_steps_l)
    print(env.env_name, len(states))

    # get random init heur fn
    device, _, _ = get_device()
    nnet = env.get_v_nnet()
    heur_fn: HeurFN_T = get_heuristic_fn(nnet, device, env, is_v=True)

    # add astar instances
    weight: float = 1.0
    astar: AStar = AStar(env)
    astar.add_instances(states, goals, [weight] * len(states), heur_fn)

    # do search
    while min(x.finished for x in astar.instances) is False:
        astar.step(heur_fn, batch_size)

    for instance, state, goal in zip(astar.instances, states, goals):
        assert instance.goal_node is not None
        _, actions, _ = get_path(instance.goal_node)
        assert is_valid_soln(state, goal, actions, env)

        for node in instance.popped_nodes:
            assert node.bellman_backup() is not None

        assert instance.root_node.tree_backup() < np.inf
