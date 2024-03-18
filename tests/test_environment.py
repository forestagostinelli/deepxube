from typing import List, Any
import pytest
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.environments.env_utils import get_environment
from numpy.typing import NDArray

env_names: List[str] = ["cube3", "puzzle8", "puzzle15", "puzzle24"]


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_states(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        assert len(states) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_get_state_actions(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        actions_l: List[List[Any]] = env.get_state_actions(states)
        assert all(len(x) > 0 for x in actions_l)


@pytest.mark.parametrize("env_name", env_names)
def test_get_start_goal_pairs(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs(list(range(0, num_states)))
        assert len(states) == num_states
        assert len(goals) == num_states


@pytest.mark.parametrize("env_name", env_names)
def test_state_to_nnet_input(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        states_nnet: List[NDArray[Any]] = env.states_to_nnet_input(states)
        assert all(x.shape[0] == num_states for x in states_nnet)


@pytest.mark.parametrize("env_name", env_names)
def test_goal_to_nnet_input(env_name: str):
    env: Environment[Any, Any] = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        goals: List[Goal] = env.sample_goal(states)
        goals_nnet: List[NDArray[Any]] = env.goals_to_nnet_input(goals)
        assert all(x.shape[0] == num_states for x in goals_nnet)


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
