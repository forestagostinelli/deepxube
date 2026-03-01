from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Action, Goal, ActsOptim, StartGoalWalkable
from deepxube.base.nnet_input import HasFlatSGAIn
from deepxube.base.factory import Parser
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.heuristic_factory import build_heur_nnet_par


@dataclass(frozen=True)
class ContState(State):
    pos: NDArray[np.float64]

    def __hash__(self) -> int:
        return hash(tuple(np.round(self.pos, 6)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContState):
            return np.allclose(self.pos, other.pos)
        return NotImplemented


@dataclass(frozen=True)
class ContGoal(Goal):
    target: NDArray[np.float64]
    tol: float


@dataclass(frozen=True)
class ContAction(Action):
    delta: NDArray[np.float64]

    def __hash__(self) -> int:
        return hash(tuple(np.round(self.delta, 6)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContAction):
            return np.allclose(self.delta, other.delta)
        return NotImplemented


@domain_factory.register_class("cont_opt")
class ContOptDomain(ActsOptim[ContState, ContAction, ContGoal],
                    StartGoalWalkable[ContState, ContAction, ContGoal],
                    HasFlatSGAIn[ContState, ContAction, ContGoal]):
    """Simple continuous point-mass domain for exercising ActsOptim and action-input DQNs."""

    def __init__(self, dim: int = 2, action_scale: float = 0.5, goal_tol: float = 0.05,
                 action_candidates: int = 8):
        super().__init__()
        self.dim = dim
        self.action_scale = action_scale
        self.goal_tol = goal_tol
        self.action_candidates = action_candidates

        # Wire action-input DQN (HeurNNetParQIn) by default
        heur_par = build_heur_nnet_par(self, "cont_opt", "cont_mlp", {"hidden": 64}, "QIN")
        self.nnet_pars.append(("heur", "cont_opt_heur.pt", heur_par))

    # Domain API
    def sample_start_states(self, num_states: int) -> List[ContState]:
        return [ContState(np.random.uniform(-1.0, 1.0, size=self.dim)) for _ in range(num_states)]

    def sample_goal_from_state(self, states_start: Optional[List[ContState]], states_goal: List[ContState]) -> List[ContGoal]:
        return [ContGoal(state_goal.pos.copy(), self.goal_tol) for state_goal in states_goal]

    def sample_state_action(self, states: List[ContState]) -> List[ContAction]:
        return [ContAction(np.random.normal(0.0, self.action_scale, size=self.dim)) for _ in states]

    def next_state(self, states: List[ContState], actions: List[ContAction]) -> Tuple[List[ContState], List[float]]:
        states_next: List[ContState] = []
        tcs: List[float] = []
        for state, action in zip(states, actions, strict=True):
            pos_next = state.pos + action.delta
            states_next.append(ContState(pos_next))
            tcs.append(float(np.linalg.norm(action.delta, ord=2)))
        return states_next, tcs

    def is_solved(self, states: List[ContState], goals: List[ContGoal]) -> List[bool]:
        return [bool(np.linalg.norm(state.pos - goal.target) <= goal.tol) for state, goal in zip(states, goals, strict=True)]

    # ActsOptim
    def get_state_actions_opt(self, states: List[ContState], goals: List[ContGoal],
                              heur_fn: Callable[[List[ContState], List[ContGoal], List[List[ContAction]]], List[List[float]]],
                              num_actions: Optional[int] = None) -> List[List[ContAction]]:
        num_out = num_actions if num_actions is not None else self.action_candidates
        actions_l: List[List[ContAction]] = []
        for state, goal in zip(states, goals, strict=True):
            candidates: List[ContAction] = self._sample_candidates(state, goal, num_out * 4)
            # Evaluate h(s,g,a) for each candidate independently
            states_rep = [state] * len(candidates)
            goals_rep = [goal] * len(candidates)
            acts_nested: List[List[ContAction]] = [[act] for act in candidates]
            q_vals: List[List[float]] = heur_fn(states_rep, goals_rep, acts_nested)
            scores: List[float] = [qv[0] for qv in q_vals]
            idxs_sorted = np.argsort(scores)[:num_out]
            actions_l.append([candidates[int(idx)] for idx in idxs_sorted])
        return actions_l

    def _sample_candidates(self, state: ContState, goal: ContGoal, num: int) -> List[ContAction]:
        """Directional noise toward goal with exploration."""
        candidates: List[ContAction] = []
        direction = goal.target - state.pos
        direction_norm = np.linalg.norm(direction) + 1e-8
        direction_unit = direction / direction_norm

        # Half candidates along goal direction with magnitude clipped, half pure exploration
        num_dir = num // 2
        num_explore = num - num_dir
        dir_steps = np.random.normal(self.action_scale, self.action_scale * 0.25, size=num_dir)
        for step in dir_steps:
            delta = direction_unit * step
            candidates.append(ContAction(delta))

        noise = np.random.normal(0.0, self.action_scale, size=(num_explore, self.dim))
        candidates.extend(ContAction(noise_i) for noise_i in noise)
        return candidates

    # NNet input (Flat S/G/A)
    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        # three numeric vectors concatenated (no one-hot)
        return [3 * self.dim], [0]

    def to_np_flat_sga(self, states: List[ContState], goals: List[ContGoal], actions: List[ContAction]) -> List[NDArray]:
        data = np.stack([np.concatenate([s.pos, g.target, a.delta]) for s, g, a in zip(states, goals, actions, strict=True)])
        return [data.astype(np.float32)]

    def __repr__(self) -> str:
        return f"ContOptDomain(dim={self.dim}, action_scale={self.action_scale}, goal_tol={self.goal_tol})"


@domain_factory.register_parser("cont_opt")
class ContOptDomainParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args = args_str.split("_") if args_str else []
        kwargs: Dict[str, Any] = {}
        for arg in args:
            if arg.startswith("d"):
                kwargs["dim"] = int(arg[1:])
            elif arg.startswith("a"):
                kwargs["action_scale"] = float(arg[1:])
            elif arg.startswith("t"):
                kwargs["goal_tol"] = float(arg[1:])
            elif arg.startswith("k"):
                kwargs["action_candidates"] = int(arg[1:])
            else:
                raise ValueError(f"Unrecognized cont_opt arg {arg!r}")
        return kwargs

    def help(self) -> str:
        return "Args: d<int> (dim), a<float> (action scale), t<float> (goal tol), k<int> (candidate actions). Example: 'cont_opt.d3_a0.3_t0.05_k8'"
