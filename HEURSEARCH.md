# Heuristic Search

## Examples
Using A* search to find paths between randomly generated states and goals
```
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.search.astar import AStar, Node, get_path

# get environment, states, and goals
env: Environment = <construct_your_environment>
states, goals = env.get_start_goal_pairs(<list_of_number_of_steps_to_take_for_each_pair>)
weights = [0.2] * len(states)
batch_size: int = 100

# load heuristic function
device, devices, on_gpu = nnet_utils.get_device()
heuristic_fn = nnet_utils.load_heuristic_fn(<path_to_model>, device, on_gpu, env.get_v_nnet(), env)

astar = AStar(env)
astar.add_instances(states, goals, weights, heuristic_fn)
while not min(x.finished for x in astar.instances):
    astar.step(heuristic_fn, batch_size, verbose=True)

# get the path to the goal for the first instance
goal_node: Optional[Node] = astar.instances[0].goal_node
path_states, path_actions, path_cost = get_path(goal_node)
```