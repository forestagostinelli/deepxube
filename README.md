# <img src="./misc/images/scrambledCube.png" width="50"> DeepXube <img src="./misc/images/solvedCube.png" width="50">
![Tests](https://github.com/forestagostinelli/deepxube/actions/workflows/test.yml/badge.svg)

--------------------------------------------------------------------------------

DeepXube (pronounced "Deep Cube") aims to solve pathfinding using a combination of deep reinforcement learning and heuristic search.

1) Learn a heuristic function that maps states and goals to an estimate of the cost-to-go from the given state to the state given goal.
2) Use the learned heuristic function with heuristic search algorithms, such as batch weighted A* search or batch weighted Q* search, to solve problem instances.

DeepXube is a generalization of DeepCubeA ([code](https://github.com/forestagostinelli/DeepCubeA/), [paper](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf)).

For any issues, you can create a GitHub issue or contact Forest Agostinelli (foresta@cse.sc.edu).

**Outline**:

- [Installation](#installation)
- [Documentation](#documentation)
- [Command line](#command-line)
  - [Domains](#domains)
  - [Domain Visualization](#domain-visualization)
  - [Neural Network Inputs](#Neural-Network-Inputs)
  - [Heuristic](#Heuristics)

## Installation

`pip install deepxube`

See [INSTALL.md](INSTALL.md) for more information.

## Documentation
Documentation of all modules is [here](https://forestagostinelli.github.io/deepxube/deepxube.html).


## Command Line
The following information is not yet pip installable, but will be soon.

deepxube can be run from the command line via the `deepxube` command.

Run `deepxube --help` for detailed information. `--help` can also be run on 
positional arguments (e.g. `deepxube train --help`).

#### Quick run
Copy the contents of the `examples/` directory or clone the project and cd to `examples/` and run:

- Get domain information: `deepxube domain_info`
- Visualize domain: `deepxube viz --domain grid_example.7 --steps 10`
- Get heuristic information: `deepxube heuristic_info`
- Get pathfinding information: `deepxube pathfinding_info`
- Time to ensure basic functionality. Can use breakpoints in your code to debug: 
  - With deepxube residual neural network: `deepxube time --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_type V`
  - With deepxube residual neural network and deep Q-network: `deepxube time --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_type QFix`
  - With custom neural network: `deepxube time --domain grid_example.7 --heur gridnet.8CH_200FC --heur_type V`
- Train heuristic function: 
- Generate problem instances for testing trained heuristic function:

### Domains
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
        NNet Inputs:
                Name: grid_nnet_input, Type: <class 'domains.grid.GridNNetInput'>
                Name: flat_sg, Type: <class 'deepxube.base.nnet_input.HasFlatSGIn.FlatSGConcrete'>
                Name: flat_sg_actfix, Type: <class 'deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.FlatSGActFixConcrete'>
```

See [Neural Network Inputs](#Neural-Network-Inputs) for more information on `NNet Inputs`.


### Domain Visualization
Visualization of states/goals and the domain transition function can be useful to validating it.
To accomplish this, a domain can inherit from `StateGoalVizable` to convert state/goal pairs to figures
and inherit from `StringToAct` to be able to type actions into the command line and see how it changes the state.

By running `deepxube viz --domain grid_example.7 --steps 10` will create a start/goal pair by taking a random walk of length 10 and visualize it.
One can vary the grid size by simply changing the number (e.g. `deepxube viz --domain grid_example.10 --steps 10`).
Action string representations are 0, 1, 2, and 3. After applying an action, the transition cost and whether or not the goal is reached will be printed.


### Neural Network Inputs
`deepxube` trains heuristic functions represented as neural networks. Different kinds of neural networks expect different kinds of inputs.
By inheriting from Mixins from `deepxube.base.nnet_input` a `NNetInput` class can be dynamically created for a domain.

`GridExample` inherits from `HasFlatSGActsEnumFixedIn` and implements `get_input_info_flat_sg`, `to_np_flat_sg`, and `actions_to_indices`.
From this, if a neural network expects a flat (1D) input from a state/goal pair, then a `NNetInput` class that tells the neural network the 
dimension of the input, the number of inputs, and that converts state/goal pairs to numpy arrays is dynamically created. 
Furthermore, if deep Q-network is used with an output for each action, then the heuristic function is automatically modified to have the correct output dimension.  
Hence, the `flat_sg` and `flat_sg_actfix` in the domain information.

With this dynamic `NNetInput` creation, `GridExample` can be used with deepxube's built in `resnet_fc` heuristic function, which expects a flat input.

Custom neural network input types can also be created and registered. Given a heuristic function, deepxube searches for a registered `NNetInput` class that 
matches its expected input. If multiple exist, it uses the first one it finds.

### Heuristics
