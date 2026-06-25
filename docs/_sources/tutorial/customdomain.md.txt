(custom_domain_section)=
# Creating a Custom Domain and Heuristic Function

We will create a simple grid domain where the agent can move up, down, left, or right along a two-dimensional grid to reach a goal 
square. We will create neural network inputs for a neural network that DeepXube provides as well as a neural network input for 
our own custom neural network.

In the directory in which you run deepxube, create a `domains/grid_tutorial.py` directory. 
DeepXube automatically looks in the `domains/` folder to see what is registered.

## Implementation
The entire domain file is here. This includes the states, actions, goals, domain, neural network inputs, custom neural network, and parsers. 
This file will be explained part-by-part.
```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
```

### State, Action, Goal

To faciliate using states with Python dictionary objects and re-identifying states during search, all State objects must implement
`__hash__` and `__eq__`. This must also be done for Action objects.

```{tip}
Implementing `__repr__` for Action objects can be convenient since actions are printed to the screen when interacting with problem 
instances with `deepxube viz`.
```

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: state, action, goal
:end-before: @domain_factory.r
```

### Domain

#### Registration and Mixins
We will register the domain with the name `grid_tut`. This tells DeepXube that this name
refers to the domain being defined.

We will use the `ActsEnumFixed` mixin since the action space is fixed (up, down, left, right) and enumerable. We will use the 
`StartGoalWalkable` to generate problem instances by sampling a start state, taking a random walk, 
and using the terminal state to sample a goal.We will also use the `StateGoalVizable` 
and `StringToAct` to interact with the domain using `deepxube viz`. 
The domain will be given an argument for its dimensionality.

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: domain definition
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

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
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

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
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

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: start startgoalwalkable methods
:end-before: end startgoalwalkable methods
```

#### Visualization and Interaction Methods

Since the domain is registered, we should be able to see "grid_tut" with: 
`deepxube domain_info`.

```{tip}
Implementing `__repr__` for Domain objects can be convenient since the domain is printed to the screen and output.txt file during 
training and solving. Having an identifiable name for the domain along with a clear representation of its parameters can be helpful 
when looking back on different runs. 
```

#### Domain Parser

To allow the user to 

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: domain parser definition
:end-before: gridflatin definition
```

### Neural Network Inputs

#### Flat Input
```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: gridflatin definition
:end-before: gridflatinqfix definition
```


#### Flat Input for a Q-Network with a Fixed Action Output
```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: gridflatinqfix definition
:end-before: gridflatinactin definition
```


#### Flat Input for a Q-Network with the Action as an Input
```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: gridflatinactin definition
:end-before: grid nnet definition
```


### Custom Neural Network
```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: grid nnet definition
```
