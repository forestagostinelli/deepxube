from typing import List, Tuple, Optional, Type
import numpy as np
from torch import nn, Tensor

from deepxube.base.factory import DelimParser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StateGoalVizable, StringToAct
from deepxube.base.nnet_input import StateGoalIn, StateGoalActFixIn, StateGoalActIn, FlatIn
from deepxube.base.heuristic import HeurNNet

from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel

from numpy.typing import NDArray
import random

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes


# start sag
class GridState(State):
    def __init__(self, robot_x: int, robot_y: int):
        self.robot_x: int = robot_x
        self.robot_y: int = robot_y

    def __hash__(self) -> int:
        return hash(self.robot_x + self.robot_y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridState):
            return (self.robot_x == other.robot_x) and (self.robot_y == other.robot_y)
        return NotImplemented


class GridGoal(Goal):
    def __init__(self, robot_x: int, robot_y: int):
        self.robot_x: int = robot_x
        self.robot_y: int = robot_y


class GridAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return ["UP", "DOWN", "LEFT", "RIGHT"][self.action]
# end sag


# start def
@domain_factory.register_class("grid_tut")
class Grid(ActsEnumFixed[GridState, GridAction, GridGoal], StartGoalWalkable[GridState, GridAction, GridGoal],
           StateGoalVizable[GridState, GridAction, GridGoal], StringToAct[GridState, GridAction, GridGoal]):
    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim: int = dim
        self.actions_fixed: List[GridAction] = [GridAction(x) for x in [0, 1, 2, 3]]
    # end init

    # start domain methods
    def is_solved(self, states: List[GridState], goals: List[GridGoal]) -> List[bool]:
        return [(state.robot_x == goal.robot_x) and (state.robot_y == goal.robot_y) for state, goal in zip(states, goals)]

    def next_state(self, states: List[GridState], actions: List[GridAction]) -> Tuple[List[GridState], List[float]]:
        states_next: List[GridState] = []
        for state, action in zip(states, actions):
            if action.action == 1:  # up
                states_next.append(GridState(min(state.robot_x + 1, self.dim - 1), state.robot_y))
            elif action.action == 0:  # down
                states_next.append(GridState(max(state.robot_x - 1, 0), state.robot_y))
            elif action.action == 3:  # left
                states_next.append(GridState(state.robot_x, min(state.robot_y + 1, self.dim - 1)))
            elif action.action == 2:  # right
                states_next.append(GridState(state.robot_x, max(state.robot_y - 1, 0)))

        return states_next, [1.0] * len(states_next)
    # end domain methods

    # start actsenumfixed methods
    def get_actions_fixed(self) -> List[GridAction]:
        return self.actions_fixed.copy()
    # end actsenumfixed methods

    # start startgoalwalkable methods
    def sample_start_states(self, num_states: int) -> List[GridState]:
        return [GridState(random.randint(0, self.dim - 1), random.randint(0, self.dim - 1)) for _ in range(num_states)]

    def sample_goal_from_state(self, states_start: Optional[List[GridState]], states_goal: List[GridState]) -> List[GridGoal]:
        return [GridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]
    # end startgoalwalkable methods

    # start viz methods
    def visualize_state_goal(self, state: GridState, goal: GridGoal, fig: Figure) -> None:
        ax: Axes = fig.subplots(1, 1)
        grid: NDArray = np.zeros((self.dim, self.dim))
        grid[goal.robot_x, goal.robot_y] = 2
        grid[state.robot_x, state.robot_y] = 1
        ax.imshow(grid, cmap=ListedColormap(["white", "black", "green"]), origin="upper")

    def string_to_action(self, act_str: str) -> Optional[GridAction]:
        if act_str in {"w", "s", "a", "d"}:
            return GridAction(["w", "s", "a", "d"].index(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "w, s, a, or d for up, down, left, and right, respectively."
    # end viz methods

    # start repr methods
    def __repr__(self) -> str:
        return f"Grid(dim={self.dim})"
    # end repr methods


# start domain parser
@domain_factory.register_parser("grid_tut")
class GridParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("d", "dim", int, "dimensionality of grid")

    @property
    def delim(self) -> str:
        return "_"
# end domain parser


# start gridflatin definition
@register_nnet_input("grid_tut", "grid_flat_in")
class GridFlatIn(StateGoalIn[Grid, GridState, GridGoal], FlatIn[Grid]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return [4], [self.domain.dim]

    def to_np(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)]
# end gridflatin definition


# start gridflatinqfix definition
@register_nnet_input("grid_tut", "grid_flat_in_qfix")
class GridFlatInQFix(StateGoalActFixIn[Grid, GridState, GridGoal, GridAction], FlatIn[Grid]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return [4], [self.domain.dim]

    def to_np(self, states: List[GridState], goals: List[GridGoal], actions_l: List[List[GridAction]]) -> List[NDArray]:
        actions_np: NDArray = np.array([[action_i.action for action_i in actions] for actions in actions_l])
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)] + [actions_np]
# end gridflatinqfix definition


# start gridflatinactin definition
@register_nnet_input("grid_tut", "grid_flat_in_actin")
class GridFlatInActIn(StateGoalActIn[Grid, GridState, GridGoal, GridAction], FlatIn[Grid]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return [4, 1], [self.domain.dim, self.domain.get_num_acts()]

    def to_np(self, states: List[GridState], goals: List[GridGoal], actions: List[GridAction]) -> List[NDArray]:
        actions_np: NDArray = np.expand_dims(np.array([action_i.action for action_i in actions]), 1)
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)] + [actions_np]
# end gridflatinactin definition


# start grid nnet input definition
@register_nnet_input("grid_tut", "grid_nnet_input")
class GridNNetInput(StateGoalIn[Grid, GridState, GridGoal]):
    def get_input_info(self) -> int:
        return self.domain.dim

    def to_np(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        np_rep: NDArray = np.zeros((len(states), 2, self.domain.dim, self.domain.dim))
        for idx, (state, goal) in enumerate(zip(states, goals)):
            np_rep[idx, 0, state.robot_x, state.robot_y] = 1
            np_rep[idx, 1, goal.robot_x, goal.robot_y] = 1

        return [np_rep]
# end grid nnet input definition


# start grid nnet definition
@heuristic_factory.register_class("gridnet_tut")
class GridNet(HeurNNet[GridNNetInput]):
    @staticmethod
    def nnet_input_type() -> Type[GridNNetInput]:
        return GridNNetInput

    def __init__(self, nnet_input: GridNNetInput, out_dim: int, q_fix: bool, chan_size: int = 8, fc_size: int = 100):
        super().__init__(nnet_input, out_dim, q_fix)
        grid_dim: int = self.nnet_input.get_input_info()

        self.heur: nn.Module = nn.Sequential(
            Conv2dModel(2, [chan_size, chan_size], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(grid_dim * grid_dim * chan_size, [fc_size], ["RELU"], batch_norms=[True]),
            nn.Linear(fc_size, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        x: Tensor = self.heur(inputs[0])
        return x
# end grid nnet definition


# start grid nnet parser definition
@heuristic_factory.register_parser("gridnet_tut")
class GridNetParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("ch", "chan_size", int, "number of channels")
        self.add_argument("fc", "fc_size", int, "size of fully connected layer")

    @property
    def delim(self) -> str:
        return "_"
# end grid nnet parser definition
