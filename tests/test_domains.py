from typing import List, cast
import pytest  # type: ignore

from deepxube.factories.domain_factory import domain_factory
from deepxube.base.domain import Domain, State, Action, Goal, GoalSampleableFromState, GoalSampleable, ActsRev


DOMAIN_NAMES: List[str] = domain_factory.get_all_class_names()


def build_domain_from_name(domain_id: str) -> Domain:
    return domain_factory.build_class(domain_id, {})


@pytest.fixture(params=DOMAIN_NAMES, ids=lambda dom_name: dom_name)  # type: ignore
def domain_name(request) -> str:  # type: ignore
    return cast(str, request.param)


@pytest.fixture  # type: ignore
def domain(domain_name: str) -> Domain:
    return build_domain_from_name(domain_name)


@pytest.fixture(
    params=[dom_id for dom_id in DOMAIN_NAMES if issubclass(domain_factory.get_type(dom_id), GoalSampleable)],
    ids=lambda dom_id: dom_id,
)  # type: ignore
def domain_goalsamp(request) -> Domain:  # type: ignore
    return build_domain_from_name(request.param)


@pytest.fixture(
    params=[dom_id for dom_id in DOMAIN_NAMES if issubclass(domain_factory.get_type(dom_id), GoalSampleableFromState)],
    ids=lambda dom_id: dom_id,
)  # type: ignore
def domain_goalsamp_fromstate(request) -> Domain:  # type: ignore
    return build_domain_from_name(request.param)


@pytest.fixture(
    params=[dom_id for dom_id in DOMAIN_NAMES if issubclass(domain_factory.get_type(dom_id), GoalSampleable)],
    ids=lambda dom_id: dom_id,
)  # type: ignore
def domain_actsrev(request) -> Domain:  # type: ignore
    return build_domain_from_name(request.param)


@pytest.mark.parametrize("num_states", [1, 5, 10])  # type: ignore
def test_get_start_goal_pairs(domain: Domain, num_states: int) -> None:
    states, goals = domain.sample_start_goal_pairs(list(range(0, num_states)))
    assert len(states) == num_states
    assert len(goals) == num_states


@pytest.mark.parametrize("num_states", [1, 5, 10])  # type: ignore
def test_get_start_goal_pairs_0steps(domain: Domain, num_states: int) -> None:
    states, goals = domain.sample_start_goal_pairs([0] * num_states)
    assert all(domain.is_solved(states, goals))


@pytest.mark.parametrize("num_states", [1, 5, 10])  # type: ignore
def test_goalsamp(domain_goalsamp_fromstate: GoalSampleableFromState, num_states: int) -> None:
    states, _ = domain_goalsamp_fromstate.sample_start_goal_pairs(list(range(0, num_states)))
    goals_samp: List[Goal] = domain_goalsamp_fromstate.sample_goal_from_state(None, states)
    assert all(domain_goalsamp_fromstate.is_solved(states, goals_samp))


@pytest.mark.parametrize("num_states", [1, 5, 10])  # type: ignore
def test_actsrev(domain_actsrev: ActsRev, num_states: int) -> None:
    states, _ = domain_actsrev.sample_start_goal_pairs(list(range(0, num_states)))
    actions: List[Action] = domain_actsrev.sample_state_action(states)
    states_next: List[State] = domain_actsrev.next_state(states, actions)[0]
    actions_rev: List[Action] = domain_actsrev.rev_action(states_next, actions)
    states_rev: List[State] = domain_actsrev.next_state(states_next, actions_rev)[0]
    assert all(state == state_rev for state, state_rev in zip(states, states_rev))


"""
def test_get_start_states(domain_id: str):
    breakpoint()
    env: Env = build_domain(domain_data[domain_id]["domain_name"], **domain_data[domain_id]["args"])
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        assert len(states) == num_states

@pytest.mark.parametrize("domain_name", domain_ids)
def test_get_state_actions(domain_name):
    env: Env = get_environment(domain_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        actions_l: List[List[Action]] = env.get_state_actions(states)
        assert all(len(x) > 0 for x in actions_l)


@pytest.mark.parametrize("env_name", domain_ids)
def test_states_goals_to_nnet_input(env_name: str):
    env: Env = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states: List[State] = env.get_start_states(num_states)
        states_goal: List[State] = env.next_state_rand(states)[0]
        goals: List[Goal] = env.sample_goal(states, states_goal)
        goals_nnet: List[NDArray[Any]] = env.states_goals_to_nnet_input(states, goals)
        assert all(x.shape[0] == num_states for x in goals_nnet)


@pytest.mark.parametrize("env_name", domain_ids)
def test_heurfn_v(env_name: str):
    env: Env = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs(list(np.random.randint(0, num_states, size=num_states)))
        device, _, _ = get_device()
        nnet = env.get_v_nnet()
        heur_fn = get_heuristic_fn(nnet, device, env, is_v=True)
        heur_vals = heur_fn(states, goals)

        assert len(heur_vals.shape) == 1
        assert heur_vals.shape[0] == num_states


@pytest.mark.parametrize("env_name", domain_ids)
def test_is_solved(env_name: str):
    env: Env = get_environment(env_name)
    for num_states in [1, 5, 10]:
        states, goals = env.get_start_goal_pairs([0] * num_states)
        assert all(env.is_solved(states, goals))


@pytest.mark.parametrize("env_name", domain_ids)
def test_expand(env_name: str):
    env: Env = get_environment(env_name)
    states, goals = env.get_start_goal_pairs([1] * 10)
    states_next_l, _, tcs_l = env.expand(states)

    for states_next, tcs, goal in zip(states_next_l, tcs_l, goals):
        assert len(states_next) == len(tcs)
        assert any(env.is_solved(states_next, [goal] * len(states_next)))
"""
