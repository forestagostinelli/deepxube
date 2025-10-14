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
- [Examples](#examples)



## Installation

`pip install deepxube`

See [INSTALL.md](INSTALL.md) for more details

## Examples
### Using DeepXube to train a heuristic function for the Rubik's cube.
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