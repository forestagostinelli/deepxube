from typing import List, Tuple, Dict, Callable, Optional, cast
from deepxube.environments.environment_abstract import Environment, State, Action, Goal
from deepxube.nnet.nnet_utils import HeurFN_T
import numpy as np
from heapq import heappush, heappop

from deepxube.utils import misc_utils
import time
from numpy.typing import NDArray


class Node:
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_action', 'parent']

    def __init__(self, state: State, goal: Goal, path_cost: float, is_solved: bool,
                 parent_action: Optional[Action], parent):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = None
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_action: Optional[Action] = parent_action
        self.parent: Optional[Node] = parent


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, root_node: Node, weight: float):
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_node: Optional[Node] = None
        self.finished: bool = False
        self.weight: float = weight
        self.num_nodes_generated: int = 0
        self.step_num: int = 0

        self.root_node: Node = root_node

        self.push_to_open([self.root_node])

    def push_to_open(self, nodes: List[Node]):
        for node in nodes:
            heappush(self.open_set, (cast(float, node.cost), self.heappush_count, node))
            self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        num_to_pop: int = min(num_nodes, len(self.open_set))

        popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]

        for node in popped_nodes:
            if node.is_solved and ((self.goal_node is None) or
                                   (cast(float, node.cost) < cast(float, self.goal_node.cost))):
                self.goal_node = node

        if (self.goal_node is not None) and (cast(float, self.goal_node.cost) <= cast(float, popped_nodes[0].cost)):
            self.finished = True
        self.popped_nodes.extend(popped_nodes)

        return popped_nodes

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost
            elif path_cost_prev > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost

        return nodes_not_in_closed


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]], env: Environment):
    # Get children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[List[float]]
    states_c_by_node, actions_c_by_node, tcs_np = env.expand(states)

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]
    goals_c_by_node: List[List[Goal]] = [[node.goal] * len(states_c) for node, states_c in
                                         zip(popped_nodes_flat, states_c_by_node)]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = misc_utils.flatten(states_c_by_node)
    goals_c, _ = misc_utils.flatten(goals_c_by_node)
    is_solved_c: List[bool] = list(env.is_solved(states_c, goals_c))
    is_solved_c_by_node: List[List[bool]] = misc_utils.unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    # TODO, fix for variable action spaces
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()

    path_costs_c_by_node: List[List[float]] = misc_utils.unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(path_costs_c_by_node,
                                                                               split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.unflatten(states_c_by_node, split_idxs)
    actions_c_by_inst_node: List[List[List[Action]]] = misc_utils.unflatten(actions_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.unflatten(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node = states_c_by_inst_node[inst_idx]
        actions_c_by_node = actions_c_by_inst_node[inst_idx]

        is_solved_c_by_node = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        for parent_node, tcs_node, path_costs_c, states_c, actions_c, is_solved_c in (zip(parent_nodes, tcs_by_node,
                                                                                          path_costs_c_by_node,
                                                                                          states_c_by_node,
                                                                                          actions_c_by_node,
                                                                                          is_solved_c_by_node)):
            state: State
            goal: Goal = parent_node.goal
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                action: Action = actions_c[move_idx]
                node_c: Node = Node(state, goal, path_cost, is_solved, action, parent_node)

                nodes_c_by_inst[inst_idx].append(node_c)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]]) -> List[List[Node]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx])

    return nodes_c_all


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        instance.push_to_open(nodes_inst)


def add_heuristic_and_cost(nodes: List[Node], heuristic_fn: HeurFN_T,
                           weights: List[float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(nodes) == 0:
        return np.zeros(0), np.zeros(0)

    # compute node cost
    states: List[State] = [node.state for node in nodes]
    goals: List[Goal] = [node.goal for node in nodes]
    heuristics = heuristic_fn(states, goals)
    path_costs: NDArray[np.float64] = np.array([node.path_cost for node in nodes])
    is_solved: NDArray[np.bool_] = np.array([node.is_solved for node in nodes])

    costs: NDArray[np.float64] = np.array(weights) * path_costs + heuristics * np.logical_not(is_solved)

    # add cost to node
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = heuristic
        node.cost = cost

    return path_costs, heuristics


class AStar:

    def __init__(self, env: Environment):
        """ Initialize AStar search

        :param env: Environment
        """
        self.env: Environment = env
        self.step_num: int = 0

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}
        self.instances: List[Instance] = []

    def add_instances(self, states: List[State], goals: List[Goal], weights: List[float], heuristic_fn: HeurFN_T):
        """ Add instances

        :param states: start states
        :param goals: goals
        :param weights: weights for weighted A* search
        :param heuristic_fn: heuristic function
        :return:
        """
        assert len(states) == len(goals), "Number of states and goals should be the same"
        assert len(goals) == len(weights), "Number of weights given should be the same as number of instances"
        # compute starting costs
        root_nodes: List[Node] = []
        is_solved_states: NDArray[np.bool_] = np.array(self.env.is_solved(states, goals))
        for state, goal, is_solved in zip(states, goals, is_solved_states):
            root_node: Node = Node(state, goal, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, heuristic_fn, weights)

        # initialize instances
        for root_node, weight in zip(root_nodes, weights):
            self.instances.append(Instance(root_node, weight))

    def step(self, heuristic_fn: HeurFN_T, batch_size: int, verbose: bool = False):
        """ Take a step of A* search

        :param heuristic_fn: heuristic function
        :param batch_size: batch size
        :param verbose: If true, prints out search step information
        :return:
        """
        start_time_itr = time.time()
        instances: List[Instance] = [instance for instance in self.instances if not instance.finished]

        # Pop from open
        start_time = time.time()
        popped_nodes_all: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]] = expand_nodes(instances, popped_nodes_all, self.env)
        expand_time = time.time() - start_time

        # Get heuristic of children, do heur before check so we can do backup
        start_time = time.time()
        nodes_c_all_flat, _ = misc_utils.flatten(nodes_c_all)
        weights, _ = misc_utils.flatten([[instance.weight] * len(nodes_c)
                                         for instance, nodes_c in zip(instances, nodes_c_all)])
        path_costs, heuristics = add_heuristic_and_cost(nodes_c_all_flat, heuristic_fn, weights)
        heur_time = time.time() - start_time

        # Check if children are in closed
        start_time = time.time()
        nodes_c_all = remove_in_closed(instances, nodes_c_all)
        check_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr

        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = float(np.min(heuristics))
                min_heur_pc = float(path_costs[np.argmin(heuristics)])
                max_heur = float(np.max(heuristics))
                max_heur_pc = float(path_costs[np.argmax(heuristics)])

                print("Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      "%.2f(%.2f)/%.2f(%.2f) " % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print("Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
                  "add: %.2f, itr: %.2f" % (pop_time, expand_time, check_time, heur_time, add_time, itr_time))

            print("")

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['check'] += check_time
        self.timings['heur'] += heur_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        self.step_num += 1
        for instance in instances:
            instance.step_num += 1

    def remove_instances(self, test_rem: Callable[[Instance], bool]) -> List[Instance]:
        """ Remove instances

        :param test_rem: A Callable that takes an instance as input and returns true if the instance should be removed
        :return: List of removed instances
        """
        instances_remove: List[Instance] = []
        instances_keep: List[Instance] = []
        for instance in self.instances:
            if test_rem(instance):
                instances_remove.append(instance)
            else:
                instances_keep.append(instance)

        self.instances = instances_keep

        return instances_remove


def get_path(node: Node) -> Tuple[List[State], List[Action], float]:
    """ Gets path from a the start state to the goal state associated with the input node

    :param node: goal node
    :return: List of states along path, List of actions in path, path cost
    """
    path: List[State] = []
    actions: List[Action] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        assert parent_node.parent_action is not None, "parent_action should not be None"
        actions.append(parent_node.parent_action)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    actions = actions[::-1]

    return path, actions, node.path_cost
