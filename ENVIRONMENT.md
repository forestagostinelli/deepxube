# Environment

**Outline**:
- [Current Environments](#current-environments)
- [Implementing Your Own Environment](#implementing-your-environment)
  - [State Class](#state-class) 
  - [Goal Class](#goal-class) 
  - [Environment Class](#environment-class)
    - [Base Class](#base-class)
    - [Answer Set Programming Integration](#environment-with-answer-set-programming-integration-class)
- [Testing Your Environment](#testing-your-implementation)

## Current Environments
- Rubik's cube `deepxube/environments/cube3`
- Sliding tile puzzle `deepxube/environments/n_puzzle`
- Sokoban `deepxube/environments/sokoban`

## Implementing Your Environment

### State Class

### Goal Class

### Environment Class

#### Base Class

- Generating start states
- Transition and transition cost function
- Sampling goals from states
- Goal test function
- Neural network functions
- PDDL functions
- Visualization

#### Environment with Answer Set Programming Integration Class

- State/model conversion
- Goal/model conversion
- Background knowledge
- Ground atoms
- On model
- Fixed ground atoms


## Testing Your Implementation
After implementing the state, goal, and environment objects, you can test the basic functionality with `tests/test_env.py`.
For example:\
`python tests/test_env.py --env cube3 --num_states 100 --step_max 30`