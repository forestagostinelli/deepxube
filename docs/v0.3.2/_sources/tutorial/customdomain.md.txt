(custom_domain_section)=
# Creating a Custom Domain and Heuristic Function

We will create a simple grid domain where the agent can move up, down, left, or right along a two-dimensional grid to reach a goal 
square. We will create neural network inputs for a neural network that DeepXube provides as well as a neural network input for 
our own custom neural network.

In the directory in which you run deepxube, copy the code below to the `domains/grid_tutorial.py` file. 
DeepXube automatically looks in the `domains/` folder to see what is registered. This file will be explained part-by-part.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
```

```{tip}
Since the domain is registered, we should be able to see "grid_tut" with
`deepxube domain_info` after it is put in your `domains/` folder. 
More specific information can be obtained about the domain with 
`deepxube domain_info --name grid_tut`
```


## State, Action, Goal

To faciliate using states with Python dictionary objects and re-identifying states during search, all State objects must implement
`__hash__` and `__eq__`. This must also be done for Action objects.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start sag
:end-before: end init
```

```{tip}
Implementing `__repr__` for Action objects can be convenient since actions are printed to the screen when interacting with problem 
instances with `deepxube viz`.
```

## Domain

### Registration, Mixins, and Initialization
We will register the domain with the name `grid_tut`. This tells DeepXube that this name
refers to the domain being defined.

We will use the {class}`deepxube.base.domain.ActsEnumFixed` mixin since the action space is fixed (up, down, left, right) and enumerable. We will use the 
{class}`deepxube.base.domain.StartGoalWalkable` to generate problem instances by sampling a start state, taking a random walk, 
and using the terminal state to sample a goal.We will also use the {class}`deepxube.base.domain.StateGoalVizable` 
and {class}`deepxube.base.domain.StringToAct` to interact with the domain using `deepxube viz`. 
The domain will be given an argument for its dimensionality.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start def
:end-before: end init
```

```{important}
A default value should be set for all Domain arguments in case they are not set via the command line.
```

### Domain methods
The abstract methods from {class}`deepxube.base.domain.Domain` not implemented by
mixins are {meth}`deepxube.base.domain.Domain.is_solved` and 
{meth}`deepxube.base.domain.Domain.next_state`. `is_solved` checks if the x and y
location of the agent is at the goal x and y location and `next_state` moves the agent 
in the corresponding direction with a transition cost of 1 for all actions.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start domain methods
:end-before: end domain methods
```

### ActsEnumFixed methods
{class}`deepxube.base.domain.ActsEnumFixed` automatically implements 
{meth}`deepxube.base.domain.Domain.sample_state_action` based on the abstract method
{meth}`deepxube.base.domain.ActsEnumFixed.get_actions_fixed`. This is implemented by 
simply returning a copy of the list created in the `__init__` method containing all actions.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start actsenumfixed methods
:end-before: end actsenumfixed methods
```

### StartGoalWalkable methods
{class}`deepxube.base.domain.StartGoalWalkable` automatically implements
{meth}`deepxube.base.domain.Domain.sample_problem_instances` based on the abstract methods
{meth}`deepxube.base.domain.StartGoalWalkable.sample_start_states` and 
{meth}`deepxube.base.domain.GoalSampleableFromState.sample_goal_from_state`. 
`sample_start_states` is implemented by placing the agent at a random x, y location.
`sample_goal_from_state` is implemented by using the x, y of the agent's location as the
desired goal.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start startgoalwalkable methods
:end-before: end startgoalwalkable methods
```

### Visualization and Interaction Methods
{class}`deepxube.base.domain.StateGoalVizable` and {class}`deepxube.base.domain.StringToAct` allow for the visualization of problem
instances and interaction with them using the terminal. A simple grid is created with black and green to indicate the locations of
the agent and goal, respectively.


```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start viz methods
:end-before: end viz methods
```

### Representation Method

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start repr methods
:end-before: end repr methods
```

```{tip}
Implementing `__repr__` for Domain objects can be convenient since the domain is printed to the screen and output.txt file during 
training and solving. Having an identifiable name for the domain along with a clear representation of its parameters can be helpful 
when looking back on different runs. 
```

### Domain Parser

To allow the user to set parameters of the domain via the command line, one can implement a {class}`deepxube.base.factory.Parser` 
class and register it with the same name as the domain. The {class}`deepxube.base.factory.DelimParser` is a subclass that makes it
easy to define parsing and help messages.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start domain parser
:end-before: end domain parser
```

Now, grid domains of different dimensions can be created using the command-line:

`deepxube viz --domain grid_tut.7d --steps 100`

`deepxube viz --domain grid_tut.20d --steps 100`

## Neural Network Inputs

### Flat Input

This input gives the x, y coordinates of the agent and goal locations
to a one-dimensional representation. It is then converted to a one-hot
representation on the GPU with depth equal to the dimensionality of the
domain.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatin definition
:end-before: end gridflatin definition
```

```{tip}
The domain is passed to the neural network input and is accessible 
via `self.domain`.
```

```{tip}
Converting data to a one-hot representation on the GPU instead of the CPU
can speed up CPU to GPU transfer.
```

```{important}
Each element in the list of numpy arrays that the `to_np` method returns
must have its first dimension be equal to the number of inputs.
```

We can now train a heuristic function that takes a flat input for the 
grid domain. It should learn to solve over 95% of problem instances with 
20 iterations of A* search during training.

`deepxube train --domain grid_tut.7d --heur resnet_fc.100H_1B_bn --heur_type V --pathfind graph_v --step_max 100 --up_itrs 100 --search_itrs 20 --backup -1 --procs 2 --batch_size 200 --max_itrs 1000 --dir tutorial/grid_tut/flatin_v/`

```{literalinclude} ../../tutorial/grid_tut/flatin_v/output.txt
:language: none
:class: scroll-code
```

### Flat Input for a Q-Network with a Fixed Action Output

This neural network input assumes a fixed and enumerable action space 
and outputs a vector that corresponds to the transition cost plus 
cost-to-go of the resulting state for every possible 
action {cite}`mnih2015human`.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatinqfix definition
:end-before: end gridflatinqfix definition
```

```{important}
It is assumed that every element in `actions_l` is of the same length. 
```

```{important}
The last element in the list of numpy arrays returned by `to_np` must
be the index of the output that corresponds to each action in `actions_l`. 
```

We can now train a deep Q-network that takes a flat input for the
grid domain.

`deepxube train --domain grid_tut.7d --heur resnet_fc.100H_1B_bn --heur_type QFix --pathfind graph_q --step_max 100 --up_itrs 100 --search_itrs 20 --backup -1 --procs 2 --batch_size 200 --max_itrs 1000 --dir tutorial/grid_tut/flatin_qfix/`

```{literalinclude} ../../tutorial/grid_tut/flatin_qfix/output.txt
:language: none
:class: scroll-code
```

```{tip}
We can verify in the output that the final layer of the deep Q-network 
matches the number of actions (4)
`(2): Linear(in_features=100, out_features=4, bias=True)`
```

### Flat Input for a Q-Network with the Action as an Input

Ths neural network input assumes the action will be given to the neural 
network along with the state and goal. This can be useful for domains with
dynamic action spaces that have transition functions that are expensive to
compute since, when used with Q* search {cite}`agostinelli2024q`, the 
number of calls to the transition function is constant with respect to the
number of applicable actions.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatinactin definition
:end-before: end gridflatinactin definition
```

We can now train a deep Q-network that takes the action as in input and a 
flat input for the grid domain.

`deepxube train --domain grid_tut.7d --heur resnet_fc.100H_1B_bn --heur_type QIn --pathfind graph_q --step_max 100 --up_itrs 100 --search_itrs 20 --backup -1 --procs 2 --batch_size 200 --max_itrs 1000 --dir tutorial/grid_tut/flatin_qin/ `

```{literalinclude} ../../tutorial/grid_tut/flatin_qin/output.txt
:language: none
:class: scroll-code
```

## Custom Neural Network
Instead of using a neural network that comes with DeepXube a custom neural
network, along with its own parser and custom neural network input, 
can be implemented.

We will implement a neural network that passes the two-dimensional grid 
to convolutional layers, flattens it, passes it to a fully-connected layer,
and then to the output layer.


### Neural Network Input

The information given to the neural network is the dimensionality of the 
grid. The state and goal will be converted to two 2D NxN grids with an indicator
in one grid for the location of the agent and in the other grid for the 
location of the goal.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start grid nnet input definition
:end-before: end grid nnet input definition
```

### Neural Network
While the neural network uses DeepXube modules to implement convolutional layers followed by a fully connected layer, arbitrary PyTorch code 
can be used to implement neural networks. The user implements {mod}`deepxube.base.heuristic.HeurNNet._forward`, which is used by superclass.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start grid nnet definition
:end-before: end grid nnet definition
```

```{important}
The neural network should return the type of neural network input it is expecting with 
{mod}`deepxube.base.heuristic.DeepXubeNNet.nnet_input_type`. The neural network can access it with `self.nnet_input`.
```

```{note}
The `out_dim` argument is 1 except in the case where a qfix neural network is used.
```

```{note}
The `q_fix` input is not used directly by the neural network, but is used by the superclass.
```

```{important}
DeepXube expects the first three arguments, `nnet_input: FlatIn, out_dim: int, q_fix: bool` to have these exact names so the 
neural network can be properly initialized.
```

```{tip}
The custom neural network can be seen with `deepxube heuristic_info` and more specific information can be seen with 
`deepxube heuristic_info --name gridnet_tut`.
```


### Parser

A parser for the custom neural network can be implemented to allow for setting hyperparameters via the command-line.

```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start grid nnet parser definition
:end-before: end grid nnet parser definition
```

We can now train a network with 16 channels and a fully connected layer of size 100:

`deepxube train --domain grid_tut.7d --heur gridnet_tut.16ch_100fc --heur_type V --pathfind graph_v --step_max 100 --up_itrs 100 --search_itrs 20 --backup -1 --procs 2 --batch_size 200 --max_itrs 1000 --dir tutorial/grid_tut/gridnet_v/`

```{literalinclude} ../../tutorial/grid_tut/gridnet_v/output.txt
:language: none
:class: scroll-code
```

## Custom Problem Instances

To specify certain problem instances to solve with DeepXube, save a dictionary with a key for the states and a key for the goals.

```{literalinclude} ../../make_gridtut_prob_insts.py
:language: python
:class: scroll-code
```

```{tip}
The two problem instances can be visualized:

`deepxube viz --domain grid_tut.7d --file tutorial/grid_tut/custom_insts.pkl --idx 0`

`deepxube viz --domain grid_tut.7d --file tutorial/grid_tut/custom_insts.pkl --idx 1`
```

| ![Instance 0](../../tutorial/grid_tut/inst0.png) | ![Instance 1](../../tutorial/grid_tut/inst1.png) |
|--------------------------------------------------|--------------------------------------------------|
| Instance 0                                       | Instance 1                                       |

The problem instances can then be solved with the trained custom neural
network:

`deepxube solve --domain grid_tut.7d --heur gridnet_tut.16ch_100fc --heur_file tutorial/grid_tut/gridnet_v/heur.pt --heur_type V --pathfind graph_v.1B_1.0W --file tutorial/grid_tut/custom_insts.pkl --results tutorial/grid_tut/results_custom_insts/ --redo`

```{literalinclude} ../../tutorial/grid_tut/results_custom_insts/output.txt
:language: none
:class: scroll-code
```

## Timing and Debugging

The functionality of the domain and of a given neural network can be timed with `deepxube time`.
Breakpoints can be set anywhere in any of the tested methods, including in the `__init__` and 
{mod}`deepxube.base.heuristic.HeurNNet._forward` portions of the neural network.

`deepxube time --domain grid_tut.7d --heur gridnet_tut.16ch_100fc --heur_type V --step_min 0 --step_max 10 --num_insts 100`