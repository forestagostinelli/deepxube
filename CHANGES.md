# Changes

## 0.2.0
* Domain -> NNetInput -> Heuristic -> Pathfinding -> Updater
* Command line tool addition
* LHBL backup added
* Intializing solver checks if number of ground atoms > 0
* Updater now balances number of steps to generate states when taking multiple update steps
* Optimizer now persists across data generation and test
* No more random state sampling when testing for previously seen states
* Add cost-to-go backup info to greedy policy test output
* Remove redundant is_solved check for greedy policy
* Consolidate updater for DAVI
* Use SharedMemory for update and heuristic function runners
* Add option to skip computation of initial heuristic value for search
* Faster data buffer initialization
* Get more statistics when solving with clingo
* Domain has a function for getting a random action that is applicable to a given state
* option to adjust step_max based on solve percentage
* option to test on test set during training


## 0.1.6
* Conflict-driven goal reaching for goals specified with negation as failure
* Preliminary timeout feature for ASP solving
* Add child nodes to nodes during A* search
* Bellman backup for A* nodes
* Tree backup for A* nodes
* get_time_str takes argument for number of decimal places

## 0.1.5
* Add Action generic to Environment
* swap opencv-python dependency for pillow 
* ASP Spec/Solver
* n-puzzle background knowledge
* hash refactor
* type hint fixes
* Fixed boltzmann overflow
* State and goal to nnet rep is one function
* Add start state to sample_goal for more robust goal sampling

## 0.1.1
* Add opencv-python and imageio for visualizing Sokoban states
* Fix Sokoban data downloading
* Change Sokoban ground atoms to new format
* Add assumed true/false to ASP sample_minimal_model
* Refactor codebase
* Type hinting fixes based on mypy
