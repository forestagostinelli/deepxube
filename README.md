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
- [Domain Visualization](#domain-visualization)
- [Neural Network Inputs](#Neural-Network-Inputs)
- [Heuristic](#Heuristics)
- [Examples](#examples)



## Installation

`pip install deepxube`

See [INSTALL.md](INSTALL.md) for more details and [documentation](https://forestagostinelli.github.io/deepxube/deepxube.html).

The following information is not yet pip installable, but will be soon.

Command line help: `deepxube --help`

### Quick run
Copy the contents of the `examples/` directory and run:\
Get domain information: `deepxube domain_info`\
Visualize domain: `deepxube viz --domain grid_example.7 --steps 10`\
Get heuristic information: `deepxube heuristic_info`\
Train heuristic function: \

## Domains
User-defined domains should go in the `./domains/` folder.
deepxube will recursively search this directory and import all modules so that domains are registered. 
For example, see the `GridExample` domain in [`examples/domains/grid.py`](examples/domains/grid.py).

`GridExample` inherits from Mixin classes from `deepxube.base.domain`, which give it additional functionality (see the [domain documentation](https://forestagostinelli.github.io/deepxube/deepxube/base/domain.html)).
- `ActsEnumFixed`: `GridExample` implements `_get_actions_fixed` 
  - Methods obtained: `get_state_action_rand`, `expand`, `get_state_actions`, `get_num_acts`
- `StartGoalWalkable`: `GridExample` implements `sample_goal` and `get_start_states`
  - Methods obtained: `get_start_goal_pairs`

By using registers from `deepxube.factories.domain_factory`, `GridExample` can be obtained from its name.
Furthermore, a parser can be implemetned and registered to allow one to specify arguments for the constructor via the command line.
By convention, everything after the '.' are considered arguments.

Running `deepxube domain_info` in a directory with `domains/grid.py` should produce (amongst other available domains):
```terminaloutput
Domain: grid_example
        Parser: An integer for the dimension. E.g. 'grid_example.7'
        NNet Inputs: flat_sg_dynamic
```

See [Neural Network Inputs](#Neural-Network-Inputs) for more information on `NNet Inputs`.


## Domain Visualization
Visualization of states/goals and the domain transition function can be useful to validating it.
To accomplish this, a domain can inherit from `StateGoalVizable` to convert state/goal pairs to figures
and inherit from `StringToAct` to be able to type actions into the command line and see how it changes the state.

By running `deepxube viz --domain grid_example.7 --steps 10` will create a start/goal pair by taking a random walk of length 10 and visualize it.
One can vary the grid size by simply changing the number (e.g. `deepxube viz --domain grid_example.10 --steps 10`).
Action string representations are 0, 1, 2, and 3. After applying an action, the transition cost and whether or not the goal is reached will be printed.


## Neural Network Inputs
`deepxube` trains heuristic functions represented as neural networks. Different kinds of neural networks expect different kinds of inputs.
By inheriting from Mixins from `deepxube.base.nnet_input` a `NNetInput` class can be dynamically created for a domain.
`GridExample` inherits from `HasFlatSGIn` and implements `get_input_info_flat_sg` and `to_np_flat_sg`.
From this, if a neural network expects a flat (1D) input from a state/goal pair, then a `NNetInput` class that tells the neural network the 
dimension of the input, the number of inputs, and that converts state/goal pairs to numpy arrays is dynamically created.
Hence, the `NNet Inputs: flat_sg_dynamic` in the domain information.


## Heuristics

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