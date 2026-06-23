from typing import List, Tuple, Dict, Optional, Any, cast
import numpy as np

from deepxube.base.factory import Parser
from deepxube.base.nnet_input import HasFlatSGAIn, HasFlatSGActsEnumFixedIn, HasTwoDSGActsEnumFixedIn
from deepxube.base.domain import State, Action, Goal, GoalStartRevWalkableActsRev, NextStateNPActsEnumFixed, StateGoalVizable, StringToAct
from deepxube.factories.domain_factory import domain_factory

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes

from numpy.typing import NDArray


class LOState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: NDArray[np.uint8]):
        self.tiles: NDArray[np.uint8] = tiles
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(self.tiles.tobytes())

        assert self.hash is not None
        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LOState):
            return np.array_equal(self.tiles, other.tiles)
        return NotImplemented


class LOGoal(Goal):
    def __init__(self, tiles: NDArray[np.uint8]):
        self.tiles: NDArray[np.uint8] = tiles


class LOAction(Action):
    def __init__(self, action: int, dim: int):
        self.action: int = action
        self.dim: int = dim

    def __hash__(self) -> int:
        return hash((self.action, self.dim))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LOAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        x: int = self.action % self.dim
        y: int = self.action // self.dim
        return f"{x},{y}"


@domain_factory.register_class("lightsout")
class LightsOut(NextStateNPActsEnumFixed[LOState, LOAction, LOGoal], GoalStartRevWalkableActsRev[LOState, LOAction, LOGoal],
                StateGoalVizable[LOState, LOAction, LOGoal], StringToAct[LOState, LOAction, LOGoal],
                HasFlatSGActsEnumFixedIn[LOState, LOAction, LOGoal], HasFlatSGAIn[LOState, LOAction, LOGoal],
                HasTwoDSGActsEnumFixedIn[LOState, LOAction, LOGoal]):
    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim: int = dim
        self.num_tiles: int = self.dim ** 2

        self.move_matrix: NDArray = np.zeros((self.num_tiles, 5), dtype=np.int64)
        for move in range(self.num_tiles):
            x_pos = int(np.floor(move / self.dim))
            y_pos = move % self.dim

            right = move + self.dim if x_pos < (self.dim-1) else move
            left = move - self.dim if x_pos > 0 else move
            up = move + 1 if y_pos < (self.dim - 1) else move
            down = move - 1 if y_pos > 0 else move

            self.move_matrix[move] = [move, right, left, up, down]

        self.actions_fixed: List[LOAction] = [LOAction(x, self.dim) for x in range(self.num_tiles)]
        self.goal_np: NDArray = np.zeros(self.num_tiles, dtype=np.uint8)

    def is_solved(self, states: List[LOState], goals: List[LOGoal]) -> List[bool]:
        states_np = np.stack([state.tiles for state in states], axis=0)
        goals_np = np.stack([goal.tiles for goal in goals], axis=0)

        return cast(List[bool], np.all(states_np == goals_np, axis=1).tolist())

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[LOState], List[LOGoal]]:
        states_goal: List[LOState] = [LOState(self.goal_np.copy())] * num
        goals: List[LOGoal] = [LOGoal(self.goal_np.copy())] * num

        return states_goal, goals

    def sample_rev_state(self, states: List[LOState]) -> Tuple[List[LOState], List[LOAction], List[float]]:
        actions: List[LOAction] = self.sample_state_action(states)
        states_rev: List[LOState] = self.next_state(states, actions)[0]

        return states_rev, actions, [1.0] * len(states)

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [self.num_tiles], [1]

    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        return [self.num_tiles, 1], [1, self.get_num_acts()]

    def to_np_flat_sg(self, states: List[LOState], goals: List[LOGoal]) -> List[NDArray]:
        return [np.stack([x.tiles for x in states], axis=0).astype(np.uint8)]

    def to_np_flat_sga(self, states: List[LOState], goals: List[LOGoal], actions: List[LOAction]) -> List[NDArray]:
        return self.to_np_flat_sg(states, goals) + [np.expand_dims(np.array(self.actions_to_indices(actions)), 1)]

    def get_input_info_2d_sg(self) -> Tuple[List[int], Tuple[int, int], List[int], Optional[int]]:
        return [1], (self.dim, self.dim), [1], 1

    def to_np_2d_sg(self, states: List[LOState], goals: List[LOGoal]) -> List[NDArray]:
        tiles_flat: NDArray[np.uint8] = np.stack([x.tiles for x in states], axis=0).astype(np.uint8)
        return [tiles_flat.reshape((-1, 1, self.dim, self.dim))]

    def actions_to_indices(self, actions: List[LOAction]) -> List[int]:
        return [action_lo.action for action_lo in actions]

    def get_actions_fixed(self) -> List[LOAction]:
        return self.actions_fixed.copy()

    def visualize_state_goal(self, state: LOState, goal: LOGoal, fig: Figure) -> None:
        state_grid: NDArray = state.tiles.reshape((self.dim, self.dim))

        ax: Axes = fig.subplots(1, 1)

        self._make_ax(state_grid, ax)
        fig.tight_layout()

    def string_to_action_help(self) -> str:
        return "Comma separated X,Y locations. For example: 3,5"

    def string_to_action(self, act_str: str) -> Optional[LOAction]:
        x_str, y_str = act_str.split(",")
        action: LOAction = LOAction(int(y_str) * self.dim + int(x_str), self.dim)
        return action

    def _make_ax(self, grid: NDArray, ax: Axes) -> None:
        cmap = ListedColormap([
            "#003b00",  # dark green
            "#00ff00",  # bright green
        ])

        ax.imshow(grid.astype(int), cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(np.arange(self.dim))
        ax.set_yticks(np.arange(self.dim))

        ax.set_xticklabels([str(i) for i in range(self.dim)])
        ax.set_yticklabels([str(i) for i in range(self.dim)])

        ax.set_xticks(np.arange(-0.5, self.dim, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.dim, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlim(-0.5, self.dim - 0.5)
        ax.set_ylim(self.dim - 0.5, -0.5)

        ax.set_aspect("equal")

    def _states_to_np(self, states: List[LOState]) -> List[NDArray[np.uint8]]:
        return [np.stack([x.tiles for x in states], axis=0)]

    def _np_to_states(self, states_np: List[NDArray]) -> List[LOState]:
        assert len(states_np) == 1
        return [LOState(x) for x in states_np[0]]

    def _next_state_np(self, states_np_l: List[NDArray], actions: List[LOAction]) -> Tuple[List[NDArray], List[float]]:
        assert len(states_np_l) == 1
        tiles_next_np: NDArray = states_np_l[0].copy()

        state_idxs: NDArray = np.arange(0, tiles_next_np.shape[0])
        state_idxs = np.expand_dims(state_idxs, 1)

        actions_np: NDArray = np.array([action.action for action in actions])
        move_matrix = self.move_matrix[actions_np]
        tiles_next_np[state_idxs, move_matrix] = (tiles_next_np[state_idxs, move_matrix] + 1) % 2

        return [tiles_next_np], [1.0] * len(actions)

    def __repr__(self) -> str:
        return f"LightsOut(dim={self.dim})"


@domain_factory.register_parser("lightsout")
class LightsOutParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"dim": int(args_str)}

    def help(self) -> str:
        return "An integer for the dimension. E.g. '7'"
