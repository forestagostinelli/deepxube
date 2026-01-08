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
  - [Pathfinding](#pathfinding)
  - [Training](#training)
- [Future Additions](#future-additions)
- [References](#references)

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

- **Information**
  - Domain information: `deepxube domain_info`
  - Heuristic information: `deepxube heuristic_info`
  - Pathfinding information: `deepxube pathfinding_info`
- **Generate problem instances for solving:** `deepxube problem_inst --domain grid_example.7 --step_max 1000 --num 100 --file valid.pkl --redo`
- **Visualization**
  - Visualize random problem instance: `deepxube viz --domain grid_example.7 --steps 10`
  - Visualize a problem instance from a file: `deepxube viz --domain grid_example.7 --file valid.pkl --idx 13`
- **Time to ensure basic functionality** (Can use breakpoints in your code to debug): 
  - With deepxube residual neural network: `deepxube time --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_type V`
  - With deepxube residual neural network and deep Q-network: `deepxube time --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_type QFix`
  - With custom neural network: `deepxube time --domain grid_example.7 --heur gridnet.8CH_200FC --heur_type V`
- **Training**
  - Train heuristic function: `deepxube train --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_type V --pathfind bwas --step_max 100 --up_itrs 100 --search_itrs 20 --backup -1 --procs 1 --batch_size 50 --max_itrs 5000 --dir dummy/`
  - Use tensorboard to see training progress: `tensorboard --logdir=dummy/`
  - Plot more detailed training information with interactive slider for training iteration: `deepxube train_summary --dir dummy` 
- **Solving**
  - Solve problem instances with all-zero heuristic: `deepxube solve --domain grid_example.7 --heur_type V --pathfind bwas.1_1.0_0.0 --file valid.pkl --results results_zeros_ex/ --redo`
  - Solve problem instances with trained heuristic: `deepxube solve --domain grid_example.7 --heur resnet_fc.100H_2B_bn --heur_file dummy/heur.pt --heur_type V --pathfind bwas.1_1.0_0.0 --file valid.pkl --results results_trained_ex/ --redo`

### Domains
User-defined domains should go in the `./domains/` folder.
deepxube will recursively search this directory and import all modules so that domains are registered. 
For example, see the `GridExample` domain in [`examples/domains/grid.py`](examples/domains/grid.py).

`GridExample` inherits from Mixin classes from `deepxube.base.domain`, which give it additional functionality (see the [domain documentation](https://forestagostinelli.github.io/deepxube/deepxube/base/domain.html)).
- `ActsEnumFixed`: `GridExample` implements `get_actions_fixed` 
  - Methods obtained: `sample_action`, `get_state_actions`, `expand`, `get_num_acts`
- `StartGoalWalkable`: `GridExample` implements `sample_goal_from_state` and `sample_start_states`
  - Methods obtained: `sample_start_goal_pairs`

By using registers from `deepxube.factories.domain_factory`, `GridExample` can be obtained from its name.
Furthermore, a parser can be implemetned and registered to allow one to specify arguments for the constructor via the command line.
By convention, everything after '.' are considered arguments.

Running `deepxube domain_info` in a directory with `domains/grid.py` should produce information about grid_example (amongst other available domains):

### Domain Visualization
Visualization of states/goals and the domain transition function can be useful to validating it.
To accomplish this, a domain can inherit from `StateGoalVizable` to convert state/goal pairs to figures
and inherit from `StringToAct` to be able to type actions into the command line and see how it changes the state.

By running `deepxube viz --domain grid_example.7 --steps 10` will create a start/goal pair by taking a random walk of length 10 and visualize it.
One can vary the grid size by simply changing the number (e.g. `deepxube viz --domain grid_example.10 --steps 10`).
After applying an action, the transition cost and whether or not the goal is reached will be printed.


### Neural Network Inputs
`deepxube` trains heuristic functions represented as neural networks. Different kinds of neural networks expect different kinds of inputs.
By inheriting from Mixins from `deepxube.base.nnet_input` a `NNetInput` class can be dynamically created for a domain.

`GridExample` inherits from `HasFlatSGActsEnumFixedIn` and implements `get_input_info_flat_sg`, `to_np_flat_sg`, and `actions_to_indices`.
From this, if a neural network expects a flat (1D) input from a state/goal pair, then a `NNetInput` class that tells the neural network the 
dimension of the input, the number of inputs, and that converts state/goal pairs to numpy arrays is dynamically created. 
Furthermore, if deep Q-network is used with an output for each action, then the heuristic function is automatically modified to have the correct output dimension.  
Hence, the `flat_sg` and `flat_sg_actfix` in the domain information.

A `NNetInput` object has a `to_np` function which converts a problem instance, or a problem instance along with actions in the case of a deep Q-network that 
takes an action as input, to a list of numpy arrays. Each row of each array in the list of numpy arrays represents a different problem instance. Therefore, 
multiple arrays of different sizes can be used to represent a single problem instance, if needed.

With this dynamic `NNetInput` creation, `GridExample` can be used with deepxube's built in `resnet_fc` heuristic function, which expects a flat input.

Custom neural network input types can also be created and registered. Given a heuristic function, deepxube searches for a registered `NNetInput` class that 
matches its expected input. If multiple exist, it uses the first one it finds. An example of a custom network is shown in `GridNetParser` in `grid.py`.

### Heuristics
A heuristic function is constructed given a neural network input of a pre-determined type, the dimensionality of the output, and a boolean indicating whether 
or not the neural network represents a deep Q-network with a fixed output size. The forward portion of the neural network expects a list of Tensors 
(this corresponds to the list of numpy arrays from the `NNetInput`) and returns a Tensor representing heuristic values. If the output is a fixed set of actions,
the heuristic neural network automatically obtains the corresponding indices from the output.

An example of a custom neural network along with its parser is shown in `GridNet` and `GridNetParser` in `grid_heur.py`.


### Pathfinding
The current pathfinding algorithms implemented include the batched and weighted versions of A* and Q* search, a greedy policy with a standard 
heuristic function (V) and a deep Q-network (Q). There are other pathfinding algorithms specifically used for training a heuristic function via 
supervised learning that sets the target cost-to-go to be that of the path cost obtained from a random walk 
(see [References](#references) for more information).

### Training
Training is given the update frequency $U$ and a batch size of $N$ and a total of $U \cdot N$ training instances are generated. 

When doing reinforcement learning based training, search is performed with the heuristic function with the given pathfinding algorithm for $I$ iterations and all states expanded 
during search are added to the training set. deepxube expects the pathfinding algorithm to expand exactly one state for each search iteration.  Therefore, 
$\frac{U\cdot N}{I}$ problem instances are first generated, where problem instance $i$ is generated with a random walk of length $K_i$. 
If problem instance $i$ is solved, then a new problem is generated, also with a random walk of length $K_i$. deepxube expects $U\cdot N$ to be divisible by $I$.

## Future Additions
DeepCubeAI: Learning world models for training and search ([code](https://github.com/misaghsoltani/DeepCubeAI), [paper](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_225.pdf)).

DeepCubeAg: Specifying goals with answer set programming ([code](https://github.com/forestagostinelli/SpecGoal), [paper](https://ojs.aaai.org/index.php/ICAPS/article/view/31454/33614)).

CDGR: Conflict-driven goal reaching ([code](https://github.com/forestagostinelli/SpecGoalNegationAsFailure), [paper](https://ojs.aaai.org/index.php/SOCS/article/view/35970/38125)).

## References
DeepCubeA (Learning heuristic functions with reinforcement learning):
```
@article{agostinelli2019solving,
  title={Solving the Rubik’s cube with deep reinforcement learning and search},
  author={Agostinelli, Forest and McAleer, Stephen and Shmakov, Alexander and Baldi, Pierre},
  journal={Nature Machine Intelligence},
  volume={1},
  number={8},
  pages={356--363},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```

DeepCubeAg (Learning heuristic functions that generalize across goals):
```
@inproceedings{agostinelli2024specifying,
  title={Specifying goals to deep neural networks with answer set programming},
  author={Agostinelli, Forest and Panta, Rojina and Khandelwal, Vedant},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={34},
  pages={2--10},
  year={2024}
}
```

Q* Search (Learning DQN heuristic functions and exploiting structure for faster search with Q* search):
```
@inproceedings{agostinelli2024q,
  title={Q* search: Heuristic search with deep q-networks},
  author={Agostinelli, Forest and Shperberg, Shahaf S and Shmakov, Alexander and McAleer, Stephen and Fox, Roy and Baldi, Pierre},
  booktitle={ICAPS Workshop on Bridging the Gap between AI Planning and Reinforcement Learning},
  year={2024}
}
```

Limited-horizon Bellman-based learning (LHBL):
```
@inproceedings{hadar2025beyond,
  title={Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search},
  author={Hadar, Gal and Agostinelli, Forest and Shperberg, Shahaf S},
  booktitle = {AAAI},
  year={2026}
}
```

CayleyPy (Training heuristic functions using supervised learning on the length of random walks):
```
@inproceedings{Chervov2025NeurIPSRubiks,
  author    = {Alexander Chervov and Kirill Khoruzhii and Nikita Bukhal and Jalal Naghiyev and Vladislav Zamkovoy and Ivan Koltsov and Lyudmila Cheldieva and Arsenii Sychev and Arsenii Lenin and Mark Obozov and Egor Urvanov and Alexey M. Romanov},
  title     = {A machine learning approach that beats Rubik’s cubes},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025},
  note      = {Spotlight},
  url       = {https://neurips.cc/}
}
```
