# DeepXube Basics

## Overview
The objective of DeepXube is to automate the solution of pathfinding problems by using deep reinforcement learning to 
learn heuristic functions that guide heuristic search to solve problems. As a result, user implementation is reduced to the 
definition of the {class}`deepxube.base.domain.Domain`, {class}`deepxube.base.nnet_input.NNetInput`, 
and {class}`deepxube.base.heuristic.DeepXubeNNet`. Furthermore, since DeepXube comes with common 
neural networks, such as residual neural networks, the implementation of a heuristic function architecture may not be necessary.

```{tip}
See {ref}`custom_domain_section` for implementing custom domain, neural network input, and heuristic function architecture classes.
```

## Registration and Parsers

DeepXube uses Python registration decorators to register domains, neural network inputs,
neural networks, and pathfinding algorithms. This way, domains, neural networks, 
and pathfinding algorithms can be referred to via the command line using plain text and 
the correct neural network input can be obtained given the domain and neural network. 

Furthermore, parsers can be implemented for domains, neural networks, and pathfinding 
algorithms. This allows arguments to be given to these classes via the command line.
The convention used is anything following a "." are arguments given to that class. For 
example `deepxube viz --domain npuzzle.4` and `deepxube viz --domain npuzzle.5` produce 
the 15-puzzle (4x4 sliding-tile puzzle) and 24-puzzle (5x5 sliding-tile puzzle) respectively.

## Domains

The {class}`deepxube.base.domain.Domain` class defines the relationship between states, actions, and goals. DeepXube makes use
of "mixins" that add functionality and simplify domain definition by automatically implementing methods based on the 
implementation of simpler methods. A summary of the primary mixins are shown here:

```{figure} ../_static/images/domain_mixins.png
:alt: DeepXube Domain Mixins
:width: 100%
:align: center

The primary mixin classes used to construct domains. Methods in white are abstract and methods in black are implemented by the class. 
Mixins inherit all functionality of their ancestors.
```


While problem instances can be generated using arbitrary code, DeepXube provides two mixins to faciliate the generation of problem
instances with random walks:

```{figure} ../_static/images/probinstgen.png
:alt: DeepXube problem instance generation
:width: 100%
:align: center

A visualization of the problem instance generation function along with two mixin classes for generating problem instances by 
either 1) sampling a start state, taking a random walk, and sampling a goal from the terminal state of that random walk; 
or 2) sampling a goal and a corresponding goal state, taking a random walk in reverse, and using the terminal state of that 
random walk as the start state.
```


```{tip}
To see all registered domains use `deepxube domain_info`.

To see the parser help, mixins, neural network inputs, and 
compatible pathfinding algorithms for a particular domain use 
`deepxube domain_info --name <name>` (i.e. `deepxube domain_info --name cube3`)
```


## Neural Network Inputs and Neural Networks
The {class}`deepxube.base.nnet_input.NNetInput` class defines the information about the data given to a 
neural network (i.e. its dimensionality) and a function that converts data (i.e. state/goal pairs) to a representation suitable for 
the neural network. The {class}`deepxube.base.heuristic.DeepXubeNNet` defines how the converted data is processed during training,
evaluation, and what loss function and optimizer is used.

For training heuristic functions, the {class}`deepxube.base.heuristic.HeurNNet` class is used and for training policies, the 
{class}`deepxube.base.heuristic.PolicyNNet`.

Mixins define what kind of heuristic function is being created and impact what input and output data to expect:

- {class}`deepxube.base.nnet_input.StateGoalIn`: mixin for heuristic functions that map state/goal pairs to the estimated 
cost-to-go
- {class}`deepxube.base.nnet_input.StateGoalActIn`: mixin for heuristic functions that map state/goal/action tuples to the estimated
transition cost when taking the given action plus cost-to-go of the resulting state
- {class}`deepxube.base.nnet_input.StateGoalActFixIn`, mixin for deep Q-network {cite:p}`mnih2015human` heuristic functions that map 
state/goal pairs to the estimated transition cost when taking plus cost-to-go of the resulting state for all actions

```{figure} ../_static/images/nnets.png
:alt: DeepXube Neural Network Inputs and Neural Networks
:width: 100%
:align: center

The interaction between domains, states, goals, actions, neural network inputs, and heuristic/policy functions.
```

```{important}
DeepXube integration with policy neural networks is preliminary and may significantly change in the future.
```

```{tip}
To see all registered heuristic neural networks use `deepxube heuristic_info`.

To see the parser help and required neural network input type, use  
`deepxube heuristic_info --name <name>` (i.e. `deepxube heuristic_info --name resnet_fc`)
```

## Pathfinding
DeepXube has pathfinding algorithms that search over both nodes and edges, that operate on enumerable and infinite action spaces, 
and that perform beam search or graph search. Pathfinding algorithms specify what mixin class a domain must subclass as well as 
what functions it uses.

```{figure} ../_static/images/pathfind.png
:alt: DeepXube Pathfinding
:width: 100%
:align: center

The available pathfinding algorithms. The mixin class that the domain must subclass is in blue and the functions used are in green. 
If both a heuristic and policy function are used, then the pathfinding algorithm is guided by the heuristic function and the policy function is used to sample actions.
```


```{tip}
While the methods for training policy functions may change, the pathfinding algorithms that expect policy functions only assume 
the ability to sample actions from the policy function and, therefore, are agnostic to how the policy function is trained.
```

```{tip}
To see all registered pathfinding algorithms use `deepxube pathfinding_info`.

To see the parser help, mixins, the required domain type, and required functions type, use  
`deepxube pathfinding_info --name <name>` (i.e. `deepxube pathfinding_info --name graph_v`)
```


## Training
Given a domain, neural network, and pathfinding algorithm, DeepXube uses the domain to generate problem instances, the pathfinding
algorithm to attempt to solve those problem instances, adds nodes/edges encountered during pathfinding to the training set, computes
an update for these nodes/edges using reinforcement learning, and trains the neural network using this data.

```{figure} ../_static/images/deepxube_overview.png
:alt: DeepXube training
:width: 100%
:align: center

Overview of the DeepXube training pipeline.
```

DeepXube leverages parallelism with GPUs to compute target network updates and train the neural network and uses 
parallelism with CPUs sample problem instances and perform pathfinding.

```{figure} ../_static/images/train_multiproc.png
:alt: DeepXube multiprocessing during training
:width: 100%
:align: center

Training is parallelized across $C$ CPUs and $G$ GPUs. CPUs sample problem instances, perform heuristic search, and 
compute targets for training. When computing targets, the CPU sends the corresponding input data and its process ID to the 
target network queue, which sends the data to the first available target network. The output of the target network is obtained 
and sent to the correct CPU via the given process ID.
```


### Reinforcement learning
Problem instances are generated with a given parameter, $K$, where each problem instance is generated with a value, 
$k$, where $k$ is uniformly distributed between 0 and $K$ and $k$ denotes the length of the random walk used to generate the problem. 
However, user-defined implementations of sampling problem instances can ignore $k$ if deemed necessary. 
Every $U$ iterations, there is a check to see if the parameters of the target network, $\theta^-$, should be updated to the 
current parameters, $\theta$. This check can always return true or be based on a threshold for the training loss. 
During the first $U$ training iterations, the target network is a function that returns zero for all inputs.


Given a training batch size, $N$, $U\cdot N$ training data instances are generated per update check. 
Search with a given pathfinding algorithm is then performed for a maximum of $I$ iterations. 
All nodes expanded or edges traversed during search are used for training. 
Therefore, $\frac{U\cdot N}{I}$ problem instances are generated and, if a problem instance, $i$, 
generated with random walk length, $k_i$, is solved, then a new problem instance is generated in its place using the same $k_i$. 
Since a good value for $K$ may not be known beforehand, DeepXube has the option to automatically adapt $K$ during training by 
starting with $K=1$ and doubling it when 50\% of instances are solved. This is done until $K$ reaches some given $K_{\text{max}}$.

#### Hindsight Experience Replay
The user may choose to use hindsight experience replay (HER) where, if the pathfinding algorithm fails to solve the problem in $I$ 
iterations, HER selects a deepest node in the search tree to generate a goal. This ensures that every search produces an example of 
a reached goal, which can facilitate learning when the generated problem instances are all difficult to solve.


```{figure} ../_static/images/trainRL.png
:alt: DeepXube rl training
:width: 100%
:align: center

The two RL training approaches. For both approaches, problem instances are generated and the selected pathfinding algorithm is 
used to try to reach the goal from the start state. When using HER, if a path to the goal is not found, a goal is sampled from 
the state associated with a deepest node in the search tree and is used in place of the original goal.
```

### Supervised Learning
Random walks can be used to create supervised labels for heuristic functions or policies. While these labels may be inaccurate, 
they can be accurate enough to learn effective heuristic functions and policies. Furthermore, they have the benefit of not needing 
a neural network to obtain training data; thererfore, often computing targets much faster than reinforcement learning approaches.

```{figure} ../_static/images/pathfind_sup.png
:alt: DeepXube supervised training
:width: 100%
:align: center

The mixins for supervised learning and the pathfinding algorithms along with their required mixin. Mixins from {mod}`deepxube.base.domain` 
can be used to automatically implement the functionality of the mixins for supervised learning.
```
