from typing import List
import pytest
from deepxube.environments.environment_abstract import Environment, State
from deepxube.environments.env_utils import get_environment

env_names: List[str] = ["cube3", "puzzle8", "puzzle15", "puzzle24"]


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_states(env_name: str):
    env: Environment = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        assert len(states) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_goal_pairs(env_name: str):
    env: Environment = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs(list(range(0, num_states)))
        assert len(states) == num_states
        assert len(goals) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_is_solved(env_name: str):
    env: Environment = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs([0] * num_states)
        assert all(env.is_solved(states, goals))
