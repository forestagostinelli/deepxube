from typing import List, Any
import pytest
from deepxube.environments.environment_abstract import Environment, State
from deepxube.environments.env_utils import get_environment

env_names: List[str] = ["cube3", "puzzle8", "puzzle15", "puzzle24"]


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_states(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        assert len(states) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_goal_pairs(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs(list(range(0, num_states)))
        assert len(states) == num_states
        assert len(goals) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_is_solved(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs([0] * num_states)
        assert all(env.is_solved(states, goals))


@pytest.mark.parametrize("env_name", env_names)
def test_expand(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    states, goals = env.get_start_goal_pairs([1] * 10)
    states_next_l, tcs_l = env.expand(states)

    for states_next, tcs, goal in zip(states_next_l, tcs_l, goals):
        assert len(states_next) == len(tcs)
        assert any(env.is_solved(states_next, [goal] * len(states_next)))
