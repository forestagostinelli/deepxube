from typing import List, Tuple, Optional
import numpy as np

from deepxube.base.env import (State, Action, Goal, EnvStartGoalRW, EnvEnumerableActs)
from deepxube.base.heuristic import HeurNNetModule, HeurNNetV
from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel
from deepxube.utils.timing_utils import Times
import torch
from torch import nn, Tensor
from numpy.typing import NDArray
import time


class LOState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: NDArray[np.uint8]):
        self.tiles: NDArray[np.uint8] = tiles
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(self.tiles.tobytes())

        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LOState):
            return np.array_equal(self.tiles, other.tiles)
        return NotImplemented


class LOGoal(Goal):
    def __init__(self, tiles: NDArray[np.uint8]):
        self.tiles: NDArray[np.uint8] = tiles


class LOAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LOAction):
            return self.action == other.action
        return NotImplemented


class LightsOut(EnvStartGoalRW[LOState, LOAction, LOGoal], EnvEnumerableActs[LOState, LOAction, LOGoal]):
    def __init__(self, dim: int, fixed: bool):
        super().__init__()
        self.dim = dim
        self.num_tiles = self.dim ** 2

        self.move_matrix: NDArray = np.zeros((self.num_tiles, 5), dtype=np.int64)
        for move in range(self.num_tiles):
            x_pos = int(np.floor(move / self.dim))
            y_pos = move % self.dim

            right = move + self.dim if x_pos < (self.dim-1) else move
            left = move - self.dim if x_pos > 0 else move
            up = move + 1 if y_pos < (self.dim - 1) else move
            down = move - 1 if y_pos > 0 else move

            self.move_matrix[move] = [move, right, left, up, down]

        self.fixed_goal: bool = fixed

    def next_state(self, states: List[LOState], actions: List[LOAction]) -> Tuple[List[LOState], List[float]]:
        # initialize
        states_np: NDArray[np.uint8] = np.stack([x.tiles for x in states], axis=0)
        states_next_np: NDArray[np.uint8] = states_np.copy()

        tcs_np: NDArray[np.float64] = np.zeros(len(states))
        for action in set(actions):
            action_idxs: NDArray[np.int_] = np.array([idx for idx in range(len(actions)) if actions[idx] == action])
            states_np_act = states_np[action_idxs]

            states_next_np_act, tcs_act = self._move_np(states_np_act, [action.action] * states_np_act.shape[0])

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        # make states
        states_next: List[LOState] = [LOState(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs

    def get_state_actions(self, states: List[LOState]) -> List[List[LOAction]]:
        return [[LOAction(x) for x in range(self.num_tiles)] for _ in range(len(states))]

    def is_solved(self, states: List[LOState], goals: List[LOGoal]) -> List[bool]:
        states_np = np.stack([state.tiles for state in states], axis=0)
        goals_np = np.stack([goal.tiles for goal in goals], axis=0)

        is_solved_l: List[bool] = np.all(states_np == goals_np, axis=1).tolist()

        return is_solved_l

    def get_goal_states(self, num_states: int) -> List[LOState]:
        states_goal: List[LOState] = [LOState(np.zeros(self.num_tiles, dtype=np.uint8)) for _ in range(num_states)]

        return states_goal

    def get_start_states(self, num_states: int) -> List[LOState]:
        assert (num_states > 0)

        # Get goal states
        states_goal: List[LOState] = self.get_goal_states(num_states)

        # random walk
        scrambs: List[int] = list(range(100, max(200, self.num_tiles * 4) + 1))
        num_steps_l: List[int] = np.random.choice(scrambs, num_states).tolist()
        states_start: List[LOState] = self.random_walk(states_goal, num_steps_l)

        return states_start

    def get_start_goal_pairs(self, num_steps_l: List[int],
                             times: Optional[Times] = None) -> Tuple[List[LOState], List[LOGoal]]:
        if not self.fixed_goal:
            return super().get_start_goal_pairs(num_steps_l, times=times)
        else:
            if times is None:
                times = Times()
            start_time = time.time()
            states_goal: List[LOState] = self.get_goal_states(len(num_steps_l))
            times.record_time("state_init", time.time() - start_time)

            start_time = time.time()
            states_start: List[LOState] = self.random_walk(states_goal, num_steps_l)
            times.record_time("rand_walk", time.time() - start_time)

            start_time = time.time()
            goals: List[LOGoal] = self.sample_goal(states_start, states_goal)
            times.record_time("samp_goal", time.time() - start_time)

            return states_start, goals

    def sample_goal(self, states_start: List[LOState], states_goal: List[LOState]) -> List[LOGoal]:
        goals: List[LOGoal] = [LOGoal(state.tiles) for state in states_goal]
        return goals

    def _move_np(self, states_np: NDArray, actions: List[int]) -> Tuple[NDArray, List[float]]:
        states_next_np: NDArray = states_np.copy()

        state_idxs: NDArray = np.arange(0, states_next_np.shape[0])
        state_idxs = np.expand_dims(state_idxs, 1)

        move_matrix = self.move_matrix[actions]
        states_next_np[state_idxs, move_matrix] = (states_next_np[state_idxs, move_matrix] + 1) % 2

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs


class NNet(HeurNNetModule):
    def __init__(self, state_dim: int, res_dim: int, num_res_blocks: int, out_dim: int, batch_norm: bool,
                 weight_norm: bool):
        super().__init__()
        input_dim: int = state_dim + state_dim

        act_fn: str = "RELU"

        def res_block_init() -> nn.Module:
            return FullyConnectedModel(res_dim, [res_dim] * 2, [act_fn, "LINEAR"],
                                       batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2,
                                       group_norms=[-1] * 2)

        self.heur = nn.Sequential(
            nn.Linear(input_dim, res_dim),
            ResnetModel(res_block_init, num_res_blocks, act_fn),
            nn.Linear(res_dim, out_dim)
        )

    def forward(self, states_goals_l: List[Tensor]) -> Tensor:
        x: Tensor = self.heur(torch.cat(states_goals_l, dim=1).float())

        return x


class LONNetParV(HeurNNetV[LOState, LOGoal]):
    def __init__(self, env: LightsOut):
        self.env: LightsOut = env

    def get_nnet(self) -> HeurNNetModule:
        return NNet(self.env.num_tiles, 1000, 4, 1, True, False)

    def to_np(self, states: List[LOState], goals: List[LOGoal]) -> List[NDArray[np.uint8]]:
        states_np: NDArray[np.uint8] = np.stack([x.tiles for x in states], axis=0).astype(np.uint8)
        goals_np: NDArray[np.uint8] = np.stack([x.tiles for x in goals], axis=0).astype(np.uint8)

        return [states_np, goals_np]
