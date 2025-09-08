# Changes

## 0.1.7
* Intializing solver checks if number of ground atoms > 0
* Create nnet on process instead of sending it to process
* Updater balances number of steps to generate states when taking multiple update steps

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
