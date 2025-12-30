# <img src="./misc/images/scrambledCube.png" width="50"> DeepXube <img src="./misc/images/solvedCube.png" width="50">
![Tests](https://github.com/forestagostinelli/deepxube/actions/workflows/test.yml/badge.svg)

--------------------------------------------------------------------------------

DeepXube (pronounced "Deep Cube") aims to solve pathfinding using a combination of deep reinforcement learning and heuristic search.

1) Learn a heuristic function that maps states and goals to an estimate of the cost-to-go from the given state to the state given goal.
2) Use the learned heuristic function with heuristic search algorithms, such as batch weighted A* search or batch weighted Q* search, to solve problem instances.

DeepXube is a generalization of DeepCubeA ([code](https://github.com/forestagostinelli/DeepCubeA/),[paper](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf)).

For any issues, you can create a GitHub issue or contact Forest Agostinelli (foresta@cse.sc.edu).

**Outline**:

- [Installation](#installation)
- [Domains](#domains)
- [Neural Network Inputs](#Neural-Network-Inputs)
- [Examples](#examples)



## Installation

`pip install deepxube`

See [INSTALL.md](INSTALL.md) for more details

## Domains
User-defined domains should go in the `./domains/` folder.
deepxube will recursively search this directory and import all modules so that domains are registered.

For example, in `./domains/grid.py` you can put the following

```python
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from matplotlib.figure import Figure

from deepxube.base.nnet_input import HasFlatSGIn
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StateGoalVizable, StringToAct, DomainParser
from deepxube.factories.domain_factory import register_domain, register_domain_parser
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


@register_domain("grid_example")
class GridExample(ActsEnumFixed[GridState, GridAction, GridGoal], StartGoalWalkable[GridState, GridAction, GridGoal],
                  StateGoalVizable[GridState, GridAction, GridGoal], StringToAct[GridState, GridAction, GridGoal],
                  HasFlatSGIn[GridState, GridAction, GridGoal]):
    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim: int = dim
        self.num_tiles: int = self.dim ** 2
        self.actions_fixed: List[GridAction] = [GridAction(x) for x in [0, 1, 2, 3]]

    def is_solved(self, states: List[GridState], goals: List[GridGoal]) -> List[bool]:
        return [(goal.robot_x is None or state.robot_x == goal.robot_x) and
                (goal.robot_y is None or state.robot_y == goal.robot_y)
                for state, goal in zip(states, goals)]

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

    def sample_goal(self, states_start: List[GridState], states_goal: List[GridState]) -> List[GridGoal]:
        return [GridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [4], [self.dim]

    def to_np_flat_sg(self, states: List[GridState], goals: List[GridGoal]) -> List[NDArray]:
        return [np.stack([x.robot_x for x in states], axis=0)]

    def visualize_state_goal(self, state: GridState, goal: GridGoal, fig: Figure) -> None:
        ax = plt.axes()
        cmap = ListedColormap(["white", "black", "green"])
        grid: NDArray = np.zeros((self.dim, self.dim))
        grid[goal.robot_x, goal.robot_y] = 2
        grid[state.robot_x, state.robot_y] = 1
        ax.imshow(grid, cmap=cmap, origin="upper")
        fig.add_axes(ax)

    def string_to_action(self, act_str: str) -> Optional[GridAction]:
        if act_str in {"0", "1", "2", "3"}:
            return GridAction(int(act_str))
        else:
            return None

    def _get_actions_fixed(self) -> List[GridAction]:
        return self.actions_fixed.copy()

    def __repr__(self) -> str:
        return f"Grid(dim={self.dim})"

@register_domain_parser("grid_example")
class GridParser(DomainParser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"dim": int(args_str)}

    def help(self) -> str:
        return "An integer for the dimension. E.g. 'grid.7'"
```

`GridExample` inherits from Mixin classes from `deepxube.base.domain`, which give it additional functionality.
By inheriting from `ActsEnumFixed`, `GridExample` implements `_get_actions_fixed` and has methods automatically implemented 
to randomly sample actions and to expand states by applying every possible action to that state.
By inheriting from `StartGoalWalkable`, `GridExample` implements `get_start_states`
and has a method automatically implemented to get problem instances (start state and goal pairs) via a random walk.

By using registers from `deepxube.factories.domain_factory`, `GridExample` can be obtained from its name.
Furthermore, a parser can be implemetned and registered to allow one to specify arguments for the constructor via the command line.
By convention, everything after the '.' are considered arguments.

Now, by running `deepxube domain_info` in a directory with `domains/grid.py` should produce:
```terminaloutput
Domain: grid_example
        Parser: An integer for the dimension. E.g. 'grid.7'
        NNet Inputs: flat_sg_dynamic
```


## Neural Network Inputs
`deepxube` trains heuristic functions represented as neural networks. Different kinds of neural networks expect different kinds of inputs.
By inheriting from Mixins from `deepxube.base.nnet_input` a `NNetInput` class can be dynamically created for a domain.
`GridExample` inherits from `HasFlatSGIn` and implements `get_input_info_flat_sg` and `to_np_flat_sg`.
From this, if a neural network expects a flat (1D) input from a state/goal pair, then a `NNetInput` class that tells the neural network the 
dimension of the input, the number of inputs, and that converts state/goal pairs to numpy arrays is dynamically created. 


## Examples
### Using DeepXube to train a heuristic function for the Rubik's cube (this part is not yet pip installable, but will be soon)
```
from deepxube.base.env import EnvEnumerableActs
from deepxube.base.updater import UpHeurArgs, UpdateHeur
from deepxube.implementations.cube3 import Cube3NNetParV
from deepxube.updater.updaters import UpdateHeurBWAS

# get environment
env: EnvEnumerableActs = Cube3(True)

# update every 1000 iterations, generate 1000 iterations worth of data, use 48 CPUs, generate data by doing search for 200 iterations, limit to 1000 searches at a time, limit neural network to process 20,000 instances at a time
up_args: UpHeurArgs = UpHeurArgs(1000, 1000, 48, 200, 1000, 20000)

# Update using value iteration and A* search to generate states
updater: UpdateHeur = UpdateHeurBWAS(env, up_args, Cube3NNetParV())

# Batch size, learning rate, learning rate decay, max training itrs, balance steps based on % solved, display iterations 
train_args: TrainArgs = TrainArgs(10000, 0.001, 0.9999993, 1000000, False, 100)

# Take between 0 and 100 steps to generate start/goal pairs, save to models/cube3/
train(updater, 100, 'models/cube3/', train_args)
```

For Q-learning and using Q* search to generate states for learning and when the output of the DQN is fixed and represents
the cost-to-go for all actions 
```
from deepxube.updater.updaters import UpdateHeurBWQSEnum
from deepxube.implementations.cube3 import Cube3NNetParQFixOut
updater: UpdateHeur = UpdateHeurBWQSEnum(env, up_args, Cube3NNetParQFixOut())
```

When the DQN takes the action as an input (can do Q* search in dynamic action spaces)
```
from deepxube.implementations.cube3 import Cube3NNetParQIn
updater: UpdateHeur = UpdateHeurBWQSEnum(env, up_args, Cube3NNetParQIn())
```