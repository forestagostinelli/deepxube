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
Implementing `__repr__` for action objects can be convenient since actions are printed to the screen when interacting with problem 
instances with `deepxube viz`.
```

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: state, action, goal
:end-before: @domain_factory.r
```

### Domain

```{literalinclude} ../../tutorial/domains/grid_tutorial.py
:language: python
:class: scroll-code
:start-after: domain definition
:end-before: domain parser definition
```

#### Domain Parser
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
