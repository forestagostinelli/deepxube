from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, cast

import numpy as np

from deepxube.base.domain import StartGoalWalkable, GoalStartRevWalkableActsRev, ActsOptim, State, Action, Goal
from deepxube.base.heuristic import HeurFnQ, HeurNNetParQIn
from deepxube.base.updater import UpArgs
from deepxube.domains.cont_opt import ContOptDomain, ContState, ContGoal, ContAction
from deepxube.factories.heuristic_factory import build_heur_nnet_par
from deepxube.pathfinding.bwqs_optim import BWQSActsOptim
from deepxube.pathfinding.supervised_q import PathFindQSupRW, PathFindQSupRWRev
from deepxube.updaters.updater_q_sup import UpdateHeurQSup
from deepxube.utils.timing_utils import Times


# Simple 1D deterministic domain to make supervised-Q and BWQS-optim tests repeatable
@dataclass(frozen=True)
class LineState(State):
    pos: int

    def __hash__(self) -> int:
        return hash(self.pos)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LineState):
            return self.pos == other.pos
        return NotImplemented


@dataclass(frozen=True)
class LineGoal(Goal):
    target: int


@dataclass(frozen=True)
class LineAction(Action):
    delta: int

    def __hash__(self) -> int:
        return hash(self.delta)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LineAction):
            return self.delta == other.delta
        return NotImplemented


class LineDomain(
    StartGoalWalkable[LineState, LineAction, LineGoal],
    GoalStartRevWalkableActsRev[LineState, LineAction, LineGoal],
    ActsOptim[LineState, LineAction, LineGoal],
):
    """1D line world with deterministic unit actions and reversible dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.heurs_eval_count: int = 0

    # Sampling utilities
    def sample_start_states(self, num_states: int) -> List[LineState]:
        return [LineState(0) for _ in range(num_states)]

    def sample_state_action(self, states: List[LineState]) -> List[LineAction]:
        # Always move +1 to keep path-costs predictable
        return [LineAction(1) for _ in states]

    def next_state(self, states: List[LineState], actions: List[LineAction]) -> Tuple[List[LineState], List[float]]:
        states_next: List[LineState] = [LineState(state.pos + act.delta) for state, act in zip(states, actions, strict=True)]
        tcs: List[float] = [float(abs(act.delta)) for act in actions]
        return states_next, tcs

    def is_solved(self, states: List[LineState], goals: List[LineGoal]) -> List[bool]:
        return [state.pos == goal.target for state, goal in zip(states, goals, strict=True)]

    def sample_goal_from_state(self, states_start: Optional[List[LineState]], states_goal: List[LineState]) -> List[LineGoal]:
        return [LineGoal(st.pos) for st in states_goal]

    def sample_goal_state_goal_pairs(self, num: int) -> Tuple[List[LineState], List[LineGoal]]:
        states_goal: List[LineState] = [LineState(idx) for idx in range(num)]
        goals: List[LineGoal] = [LineGoal(st.pos) for st in states_goal]
        return states_goal, goals

    # Reversibility
    def rev_action(self, states: List[LineState], actions: List[LineAction]) -> List[LineAction]:
        return [LineAction(-act.delta) for act in actions]

    # ActsOptim
    def get_state_actions_opt(
        self,
        states: List[LineState],
        goals: List[LineGoal],
        heur_fn: Callable[[List[LineState], List[LineGoal], List[List[LineAction]]], List[List[float]]],
        num_actions: Optional[int] = None,
    ) -> List[List[LineAction]]:
        cand_actions: List[LineAction] = [LineAction(delta) for delta in (-2, -1, 0, 1, 2)]
        num_ret: int = len(cand_actions) if num_actions is None else num_actions

        actions_out: List[List[LineAction]] = []
        for state, goal in zip(states, goals, strict=True):
            states_rep: List[LineState] = [state] * len(cand_actions)
            goals_rep: List[LineGoal] = [goal] * len(cand_actions)
            q_vals = heur_fn(states_rep, goals_rep, [[act] for act in cand_actions])
            scores: List[float] = [vals[0] for vals in q_vals]
            self.heurs_eval_count += len(scores)
            best_idxs = np.argsort(scores)[:num_ret]
            actions_out.append([cand_actions[int(idx)] for idx in best_idxs])

        return actions_out


def test_sup_q_rw_path_cost_matches_steps() -> None:
    domain = LineDomain()
    steps = [0, 2, 3]
    pathfind = PathFindQSupRW(domain)
    instances = pathfind.make_instances_rw(steps, None)
    pathfind.add_instances(instances)

    edges = pathfind.step()

    assert [edge.q_val for edge in edges] == steps
    assert all(edge.action == LineAction(1) for edge in edges)


def test_sup_q_rw_rev_path_cost_matches_steps() -> None:
    domain = LineDomain()
    steps = [0, 1, 4]
    pathfind = PathFindQSupRWRev(domain)
    instances = pathfind.make_instances_rw(steps, None)
    pathfind.add_instances(instances)

    edges = pathfind.step()

    assert [edge.q_val for edge in edges] == steps
    assert all(edge.action == LineAction(-1) for edge in edges)  # reverse of +1


def test_update_q_sup_returns_inputs_and_targets() -> None:
    np.random.seed(0)
    domain = ContOptDomain(dim=2, action_scale=0.1, goal_tol=0.01, action_candidates=4)
    pathfind = PathFindQSupRW(domain)
    instances = pathfind.make_instances_rw([1], None)
    pathfind.add_instances(instances)

    up_args = UpArgs(procs=1, up_itrs=1, step_max=1, search_itrs=1, up_batch_size=1, nnet_batch_size=8, sync_main=True, v=False)
    updater = UpdateHeurQSup(domain, "sup_q_rw", {"domain": domain}, up_args)
    data = updater._step(pathfind, Times())
    assert len(data) == 2

    features, targets = data
    assert features.shape[0] == 1
    assert targets.shape == (1,)
    # Targets should mirror the backup value placed on the root node during supervised Q generation
    assert np.isclose(targets[0], pathfind.instances[0].root_node.backup_val)


def test_bwqs_acts_optim_finds_solution_and_uses_heur() -> None:
    domain = LineDomain()

    def heur_fn(states: List[LineState], goals: List[LineGoal], actions_l: List[List[LineAction]]) -> List[List[float]]:
        # True one-step cost-to-go for each candidate action
        return [[float(abs(state.pos + act.delta - goal.target)) for act in actions] for state, goal, actions in zip(states, goals, actions_l, strict=True)]

    pathfind = BWQSActsOptim(domain, batch_size=1, weight=1.0, eps=0.0, num_actions=2)
    pathfind.set_heur_fn(cast(HeurFnQ, heur_fn))

    instances = pathfind.make_instances([LineState(0)], [LineGoal(2)], None, compute_root_heur=True)
    pathfind.add_instances(instances)

    for _ in range(8):
        pathfind.step()
        if all(inst.finished() for inst in pathfind.instances):
            break

    inst = pathfind.instances[0]
    assert inst.has_soln()
    assert inst.goal_node is not None
    assert inst.goal_node.path_cost <= 2.0
    assert domain.heurs_eval_count > 0


def test_acts_optim_get_state_actions_opt_limits_and_uses_heur() -> None:
    domain = LineDomain()

    def heur_fn(states: List[LineState], goals: List[LineGoal], actions_l: List[List[LineAction]]) -> List[List[float]]:
        # Penalize positive deltas to force a predictable ordering
        return [[float(idx)] for idx, _ in enumerate(actions_l)]

    actions_l = domain.get_state_actions_opt([LineState(0)], [LineGoal(5)], heur_fn, num_actions=3)
    assert len(actions_l) == 1
    assert len(actions_l[0]) == 3
    assert domain.heurs_eval_count >= 3


def test_heur_nnet_par_qin_to_np_shapes_cont_opt() -> None:
    domain = ContOptDomain(dim=2, action_scale=0.1, goal_tol=0.01, action_candidates=4)
    heur_par = build_heur_nnet_par(domain, "cont_opt", "cont_mlp", {"hidden": 8}, "QIN")

    states = [ContState(np.array([0.0, 0.0])), ContState(np.array([0.5, -0.5]))]
    goals = [ContGoal(np.array([1.0, 1.0]), 0.1), ContGoal(np.array([-0.5, 0.5]), 0.1)]
    actions = [ContAction(np.array([0.1, 0.2])), ContAction(np.array([-0.2, -0.1]))]

    assert isinstance(heur_par, HeurNNetParQIn)
    np_inputs = heur_par.to_np(
        cast(List[State], states),
        cast(List[Goal], goals),
        cast(List[List[Action]], [[action] for action in actions]),
    )
    assert len(np_inputs) == 1
    arr = np_inputs[0]
    assert arr.shape == (2, 6)  # concat s(2) + g(2) + a(2)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr[0], np.array([0.0, 0.0, 1.0, 1.0, 0.1, 0.2], dtype=np.float32))
