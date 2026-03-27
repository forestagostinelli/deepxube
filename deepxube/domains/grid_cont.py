from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from matplotlib.figure import Figure

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, StartGoalWalkable, StateGoalVizable, StringToAct
from deepxube.base.nnet_input import HasFlatSGIn, FlatInPolicy
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

import matplotlib.pyplot as plt
from numpy.typing import NDArray


# Define states, goals, and actions
class GridState(State):
    def __init__(self, robot_x: float, robot_y: float):
        self.robot_x: float = robot_x
        self.robot_y: float = robot_y

    def __hash__(self) -> int:
        return hash(self.robot_x + self.robot_y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridState):
            return max(abs(self.robot_x - other.robot_x), abs(self.robot_y - other.robot_y)) < 0.0001
        return NotImplemented


class GridGoal(Goal):
    def __init__(self, robot_x: float, robot_y: float):
        self.robot_x: float = robot_x
        self.robot_y: float = robot_y


class GridAction(Action):
    def __init__(self, action: Tuple[float, float]):
        self.action: Tuple[float, float] = action

    def __hash__(self) -> int:
        return hash(self.action)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridAction):
            return max(abs(self.action[0] - other.action[0]), abs(self.action[1] - other.action[1])) < 0.0001
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class("grid_cont")
class Grid(StartGoalWalkable[GridState, GridAction, GridGoal], StateGoalVizable[GridState, GridAction, GridGoal], StringToAct[GridState, GridAction, GridGoal],
           HasFlatSGIn[GridState, GridAction, GridGoal]):
    def __init__(self) -> None:
        super().__init__()
        self.dim: int = 1
        self.max_step: float = 0.1

    def is_solved(self, states: List[GridState], goals: List[GridGoal]) -> List[bool]:
        return [max(abs(state.robot_x - goal.robot_x), abs(state.robot_y - goal.robot_y)) < 0.02 for state, goal in zip(states, goals)]

    def sample_start_states(self, num_states: int) -> List[GridState]:
        return [GridState(np.random.uniform(0, self.dim), np.random.uniform(0, self.dim)) for _ in range(num_states)]

    def sample_state_action(self, states: List[GridState]) -> List[GridAction]:
        action_vals_l: List[List[float]] = np.random.uniform(-self.max_step, self.max_step, size=(len(states), 2)).tolist()
        actions: List[GridAction] = []
        for action_vals_i in action_vals_l:
            actions.append(GridAction((action_vals_i[0], action_vals_i[1])))

        return actions

    def next_state(self, states: List[GridState], actions: List[GridAction]) -> Tuple[List[GridState], List[float]]:
        states_next: List[GridState] = []
        for state, action in zip(states, actions):
            act_x: float = max(min(action.action[0], self.max_step), -self.max_step)
            act_y: float = max(min(action.action[1], self.max_step), -self.max_step)
            x_next: float = max(min(state.robot_x + act_x, self.dim), 0)
            y_next: float = max(min(state.robot_y + act_y, self.dim), 0)
            states_next.append(GridState(x_next, y_next))

        return states_next, [1.0] * len(states_next)

    def sample_goal_from_state(self, states_start: Optional[List[GridState]], states_goal: List[GridState]) -> List[GridGoal]:
        return [GridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [4], [1]

    def to_np_flat_sg(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)/self.dim]

    def visualize_state_goal(self, state: GridState, goal: GridGoal, fig: Figure) -> None:
        ax = plt.axes()
        ax.scatter([state.robot_x], [state.robot_y], s=120, color="black")
        ax.scatter([goal.robot_x], [goal.robot_y], s=120, color="green")
        ax.annotate(f"({state.robot_x:.3f}, {state.robot_y:.3f})", (state.robot_x, state.robot_y), textcoords="offset points", xytext=(8, 8))
        ax.annotate(f"({goal.robot_x:.3f}, {goal.robot_y:.3f})", (goal.robot_x, goal.robot_y), textcoords="offset points", xytext=(8, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")  # keep it a true square
        ax.grid(True, alpha=0.3)

        # res: int = 1
        # grid: NDArray = np.zeros((self.dim * res, self.dim * res))
        # grid[int(goal.robot_x), int(goal.robot_y)] = 2
        # grid[int(state.robot_x), int(state.robot_y)] = 1
        # ax.imshow(grid, cmap=ListedColormap(["white", "black", "green"]), origin="upper")
        fig.add_axes(ax)

    def string_to_action(self, act_str: str) -> Optional[GridAction]:
        x_str, y_str = act_str.split(",")
        x: float = float(x_str)
        y: float = float(y_str)
        return GridAction((x, y))

    def string_to_action_help(self) -> str:
        return "<x>,<y> relative force to apply in x and y directions."

    def __repr__(self) -> str:
        return "GridCont"


@domain_factory.register_parser("grid_cont")
class GridParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {}

    def help(self) -> str:
        return ""


@register_nnet_input("grid_cont", "grid_nnet_input_policy")
class GridNNetInputPolicy(FlatInPolicy[Grid, GridState, GridGoal, GridAction]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return [4, 2], [1, 1]

    def to_np(self, states: List[GridState], goals: List[GridGoal], actions: List[GridAction]) -> List[NDArray]:
        act_x: NDArray = np.array([action.action[0] for action in actions])
        act_y: NDArray = np.array([action.action[1] for action in actions])
        return self.domain.to_np_flat_sg(states, goals) + [np.column_stack((act_x, act_y))]

    def states_goals_actions_split_idx(self) -> int:
        return 1

    def to_np_fn(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        return self.domain.to_np_flat_sg(states, goals)

    def nnet_out_to_actions(self, nnet_out: List[NDArray[np.float64]]) -> List[GridAction]:
        nnet_out_l: List[List[float]] = nnet_out[0].tolist()
        actions: List[GridAction] = []
        for nnet_out_i in nnet_out_l:
            actions.append(GridAction((nnet_out_i[0], nnet_out_i[1])))

        return actions
