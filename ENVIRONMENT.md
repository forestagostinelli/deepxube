# Environment
- [Current Environments](#current-environments)
- [Implementing Your Own Environment](#implementing-your-environment)
  - [State Class](#state-class) 
  - [Goal Class](#goal-class) 
  - [Environment Class](#environment-class)
    - [Base Class](#base-class)
    - [Answer Set Programming Integration](#environment-with-answer-set-programming-integration-class)
- [Testing Your Environment](#testing-your-implementation)

## Current Environments
- Rubik's cube `deepxube/environments/cube3.py`
- Sliding tile puzzle `deepxube/environments/n_puzzle.py`
- Sokoban `deepxube/environments/sokoban.py`
  - Sokoban requires data be downloaded. It will ask you if you want to download the data the first time you intialize the class.
You can do so by just testing the environment:\
`python tests/test_env.py --env sokoban --num_states 100 --step_max 30`.

## Implementing Your Own Environment
The abstract environment file `deepxube/environments/environment_abstract.py` contains abstract classes for states, 
goals, and environments.

### State Class
`class State(ABC):`

Represents a state.

### Goal Class
`class Goal(ABC):`

Represents a set of states that are considered goal states. In the simplest case, a goal could also just be a state. 
However, goals can also be logic, natural language, a sketch, etc.

### Environment Class

#### Base Class
`Environment(ABC):`

#### Environment with Answer Set Programming Integration Class
`EnvGrndAtoms(Environment):`


## Testing Your Implementation
After implementing the state, goal, and environment objects, you can test the basic functionality with `tests/test_env.py`.
For example:\
`python tests/test_env.py --env cube3 --num_states 100 --step_max 30`