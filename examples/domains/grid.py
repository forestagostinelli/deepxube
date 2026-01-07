from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from matplotlib.figure import Figure

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StateGoalVizable, StringToAct
from deepxube.base.nnet_input import StateGoalIn, HasFlatSGActsEnumFixedIn
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
from numpy.typing import NDArray


# Define states, goals, and actions
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


@domain_factory.register_class("grid_example")
class GridExample(ActsEnumFixed[GridState, GridAction, GridGoal], StartGoalWalkable[GridState, GridAction, GridGoal],
                  StateGoalVizable[GridState, GridAction, GridGoal], StringToAct[GridState, GridAction, GridGoal],
                  HasFlatSGActsEnumFixedIn[GridState, GridAction, GridGoal]):
    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim: int = dim
        self.actions_fixed: List[GridAction] = [GridAction(x) for x in [0, 1, 2, 3]]

    def is_solved(self, states: List[GridState], goals: List[GridGoal]) -> List[bool]:
        return [(state.robot_x == goal.robot_x) and (state.robot_y == goal.robot_y) for state, goal in zip(states, goals)]

    def get_start_states(self, num_states: int) -> List[GridState]:
        return [GridState(np.random.randint(self.dim), np.random.randint(self.dim)) for _ in range(num_states)]

    def next_state(self, states: List[GridState], actions: List[GridAction]) -> Tuple[List[GridState], List[float]]:
        states_next: List[GridState] = []
        for state, action in zip(states, actions):
            if action.action == 0:  # up
                states_next.append(GridState(min(state.robot_x + 1, self.dim - 1), state.robot_y))
            elif action.action == 1:  # down
                states_next.append(GridState(max(state.robot_x - 1, 0), state.robot_y))
            elif action.action == 2:  # left
                states_next.append(GridState(state.robot_x, min(state.robot_y + 1, self.dim - 1)))
            elif action.action == 3:  # right
                states_next.append(GridState(state.robot_x, max(state.robot_y - 1, 0)))

        return states_next, [1.0] * len(states_next)

    def sample_goal_from_state(self, states_start: List[GridState], states_goal: List[GridState]) -> List[GridGoal]:
        return [GridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [4], [self.dim]

    def to_np_flat_sg(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)]

    def actions_to_indices(self, actions: List[GridAction]) -> List[int]:
        return [action_i.action for action_i in actions]

    def visualize_state_goal(self, state: GridState, goal: GridGoal, fig: Figure) -> None:
        ax = plt.axes()
        grid: NDArray = np.zeros((self.dim, self.dim))
        grid[goal.robot_x, goal.robot_y] = 2
        grid[state.robot_x, state.robot_y] = 1
        ax.imshow(grid, cmap=ListedColormap(["white", "black", "green"]), origin="upper")
        fig.add_axes(ax)

    def string_to_action(self, act_str: str) -> Optional[GridAction]:
        if act_str in {"0", "1", "2", "3"}:
            return GridAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "0, 1, 2, or 3 for down, up, right, and left, respectively."

    def get_actions_fixed(self) -> List[GridAction]:
        return self.actions_fixed.copy()

    def __repr__(self) -> str:
        return f"Grid(dim={self.dim})"


@domain_factory.register_parser("grid_example")
class GridParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"dim": int(args_str)}

    def help(self) -> str:
        return "An integer for the dimension. E.g. 'grid_example.7'"


@register_nnet_input("grid_example", "grid_nnet_input")
class GridNNetInput(StateGoalIn[GridExample, GridState, GridGoal]):
    def get_input_info(self) -> int:
        return self.domain.dim

    def to_np(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        np_rep: NDArray = np.zeros((len(states), 2, self.domain.dim, self.domain.dim))
        for idx, (state, goal) in enumerate(zip(states, goals)):
            np_rep[idx, 0, state.robot_x, state.robot_y] = 1
            np_rep[idx, 1, goal.robot_x, goal.robot_y] = 1

        return [np_rep]