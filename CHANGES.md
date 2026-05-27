# Changes

## 0.2.2
* Optimizer creation and update during training now part of DeepXubeNNet class. Methods can be overridden for different behavior.
* step_min argument added to deepxube time
* Added ReLU2 activation function
* Fix start_idx usage in solve
* sample_next_state returns action taken
* Return actions taken for random_walk
* ActsRev implements method to sample reverse states, actions that return from reverse states, and transition costs along edges from resulting reverse states
* Make generating supervised labels for nodes/edges a Domain mixin
* Consolidate supervised pathfinding classes into one file.
* Sample from data buffer without replacement until all states seen
* Fix up_gen_itrs being set to 100. Now defaults to up_itrs unless specifically set to a given value.
* Separate Domain mixins/pathfinding algs for supervised node, edges, and sampling edges (for policy)
* Remove policy_rand argument
* Policy sampling method defined in PolicyNNet
* Add max_itrs arg to solve
* get_path returns list of transition costs
* Added rollout pathfinding alg, special case of beam search that does not check is_solved when searching
* PolicyNNet has _forward_train and _forward_eval abstract methods to unify forward method
* Add checkpointing during training
* Add more functionality to add neural network functions to domain and use in parallel
* Add special case of training heuristic with random policy when using pathfinding algorithm that uses a policy and not training a policy
* Add timing of getting supervised data to supervised pathfinding
* Add abstract method for loss info for training policy
* Add layer norm to resnet_fc
* Vectorize expand
* Make policy at update_num=0 sampled from Domain's sample_state_action
* Separate timings for HER and rb
* Video added to viz when visualizing solutions
* Can add random edges to graph and beam search when using policy
* Add random action option to viz
* Add --no_act option to viz
* mean(std/min/max) for itrs and path_costs in --up_v
* Make training more generic for DeepXubeNNet to allow more user flexibility

## 0.2.1
* Consolidate search: Beam search -> special cases: greedy_policy, graph search -> special cases: batch weighted A* search, batch weighted Q* search 
* Replay buffer added
* HER added
* add method to Updater to get state actions
* separate step and step_sync_main method in Updater
* Explicitly call del and gc during update to better free memory
* InstanceV tracks nodes_popped, InstanceQ tracks edges_popped, more memory efficient updater
* map_location=torch.device('cpu') when loading on cpu
* DataParallel during solve
* add ctg_backup when doing sync_main for updater_q_rl
* resnet_2d heuristic function
* Visualize solutions
* Fix BWQS verbose

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
