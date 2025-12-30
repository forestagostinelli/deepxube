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

The following information is not yet pip installable, but will be soon.

## Domains
User-defined domains should go in the `./domains/` folder.
deepxube will recursively search this directory and import all modules so that domains are registered.

For example, see the `GridExample` domain in [`examples/domains/grid.py`](examples/domains/grid.py).

`GridExample` inherits from Mixin classes from `deepxube.base.domain`, which give it additional functionality.
By inheriting from `ActsEnumFixed`, `GridExample` implements `_get_actions_fixed` and has methods automatically implemented 
to randomly sample actions and to expand states by applying every possible action to that state.
By inheriting from `StartGoalWalkable`, `GridExample` implements `get_start_states`
and has a method automatically implemented to get problem instances (start state and goal pairs) via a random walk.

By using registers from `deepxube.factories.domain_factory`, `GridExample` can be obtained from its name.
Furthermore, a parser can be implemetned and registered to allow one to specify arguments for the constructor via the command line.
By convention, everything after the '.' are considered arguments.

Running `deepxube domain_info` in a directory with `domains/grid.py` should produce (amongst other available domains):
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
Hence, the `NNet Inputs: flat_sg_dynamic` in the domain information.


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