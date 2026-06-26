(custom_domain_section)=
# Creating a Custom Domain and Heuristic Function

We will create a simple grid domain where the agent can move up, down, left, or right along a two-dimensional grid to reach a goal 
square. We will create neural network inputs for a neural network that DeepXube provides as well as a neural network input for 
our own custom neural network.

In the directory in which you run deepxube, create a `domains/grid_tutorial.py` file. 
DeepXube automatically looks in the `domains/` folder to see what is registered.

## Implementation
The entire domain file is here. This includes the states, actions, goals, domain, neural network inputs, custom neural network, and parsers. 
This file will be explained part-by-part.
```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
```

```{tip}
Since the domain is registered, we should be able to see "grid_tut" with:
`deepxube domain_info` after it is put in your `domains/` folder.
```


### State, Action, Goal

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

### Domain

#### Registration, Mixins, and Initialization
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

#### Domain methods
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

#### ActsEnumFixed methods
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

#### StartGoalWalkable methods
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

#### Visualization and Interaction Methods
{class}`deepxube.base.domain.StateGoalVizable` and {class}`deepxube.base.domain.StringToAct` allow for the visualization of problem
instances and interaction with them using the terminal. A simple grid is created with black and green to indicate the locations of
the agent and goal, respectively.


```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start viz methods
:end-before: end viz methods
```

#### Representation Method

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

#### Domain Parser

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

```{note}
Everything after the "." in the domain name is given to the parser to be parsed.
```

### Neural Network Inputs

#### Flat Input
```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatin definition
:end-before: end gridflatin definition
```


#### Flat Input for a Q-Network with a Fixed Action Output
```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatinqfix definition
:end-before: end gridflatinqfix definition
```


#### Flat Input for a Q-Network with the Action as an Input
```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start gridflatinactin definition
:end-before: end gridflatinactin definition
```


### Custom Neural Network
```{literalinclude} ../../domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start grid nnet definition
```

## Timing and Debugging