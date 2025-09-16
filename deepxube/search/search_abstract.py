from typing import List, Tuple, Callable, Optional, Any, TypeVar
from abc import ABC, abstractmethod
from deepxube.environments.environment_abstract import Environment, State, Action, Goal, Generic
from deepxube.utils.timing_utils import Times
from deepxube.nnet.nnet_utils import HeurFN_T
from deepxube.utils import misc_utils
import numpy as np
from numpy.typing import NDArray
import time


class Node:
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'children', 't_costs', 'bellman_backup_val']
    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: bool,
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['Node']):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.is_solved: bool = is_solved
        self.parent_action: Optional[Action] = parent_action
        self.parent_t_cost: Optional[float] = parent_t_cost
        self.parent: Optional[Node] = parent

        self.children: Optional[List[Node]] = None
        self.t_costs: Optional[List[float]] = None
        self.bellman_backup_val: Optional[float] = None

    def bellman_backup(self) -> float:
        self.bellman_backup_val: float
        if self.is_solved:
            self.bellman_backup_val = 0.0
        else:
            assert self.children is not None
            if len(self.children) == 0:
                self.bellman_backup_val = self.heuristic
            else:
                assert self.t_costs is not None

                self.bellman_backup_val = np.inf
                for node_c, tc in zip(self.children, self.t_costs):
                    self.bellman_backup_val = min(self.bellman_backup_val, tc + node_c.heuristic)

        return self.bellman_backup_val

    def upper_bound_parent_path(self, ctg_ub: float):
        self.bellman_backup_val = min(self.bellman_backup_val, ctg_ub)
        if self.parent is not None:
            assert self.parent_t_cost is not None
            self.parent.upper_bound_parent_path(ctg_ub + self.parent_t_cost)


class Instance:
    def __init__(self, root_node: Node, inst_info: Any):
        self.root_node: Node = root_node
        self.itr: int = 0  # update with every search iteration
        self.num_nodes_generated: int = 0
        self.inst_info: Any = inst_info
        self.nodes_expanded: List[Node] = []
        self.goal_node: Optional[Node] = None

    def has_soln(self) -> bool:
        if self.goal_node is None:
            return False
        else:
            return True

    def path_cost(self) -> float:
        if not self.has_soln():
            return np.inf
        else:
            assert self.goal_node is not None
            return self.goal_node.path_cost


I = TypeVar('I', bound=Instance)

class Search(ABC, Generic[I]):
    def __init__(self, env: Environment):
        self.env: Environment = env
        self.instances: List[I] = []
        self.times: Times = Times()

    @abstractmethod
    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: HeurFN_T,
                      inst_infos: Optional[List[Any]] = None, compute_init_heur: bool = True, **kwargs):
        pass

    @abstractmethod
    def step(self, heur_fn: HeurFN_T) -> Tuple[List[State], List[Goal], List[float]]:
        pass

    def expand_nodes(self, instances: List[I], nodes_by_inst: List[List[Node]], heur_fn: HeurFN_T) -> List[List[Node]]:
        start_time = time.time()
        # flatten (for speed)
        nodes: List[Node]
        split_idxs: List[int]
        nodes, split_idxs = misc_utils.flatten(nodes_by_inst)

        if len(nodes) == 0:
            return [[]]

        # Get children of nodes
        states: List[State] = [x.state for x in nodes]

        states_c: List[List[State]]
        actions_c: List[List[Action]]
        tcs: List[List[float]]
        states_c, actions_c, tcs = self.env.expand(states)

        goals_c: List[List[Goal]] = [[node.goal] * len(states_c) for node, states_c in zip(nodes, states_c)]
        self.times.record_time("expand", time.time() - start_time)

        # Get is_solved on all states at once (for speed)
        start_time = time.time()
        states_c_flat: List[State]

        states_c_flat, split_idxs_c = misc_utils.flatten(states_c)
        goals_c_flat, _ = misc_utils.flatten(goals_c)
        is_solved_c_flat: List[bool] = list(self.env.is_solved(states_c_flat, goals_c_flat))
        is_solved_c: List[List[bool]] = misc_utils.unflatten(is_solved_c_flat, split_idxs_c)
        self.times.record_time("is_solved", time.time() - start_time)

        start_time = time.time()
        heuristics_c_flat: List[float] = list(heur_fn(states_c_flat, goals_c_flat))
        heuristics_c: List[List[float]] = misc_utils.unflatten(heuristics_c_flat, split_idxs_c)
        self.times.record_time("heur", time.time() - start_time)

        # get children nodes
        start_time = time.time()
        nodes_c: List[Node] = []
        for node_idx, node in enumerate(nodes):
            path_costs_c_i: NDArray = node.path_cost + np.array(tcs[node_idx])
            nodes_c_i: List[Node] = []
            for c_idx in range(len(states_c[node_idx])):
                node_c: Node = Node(states_c[node_idx][c_idx], goals_c[node_idx][c_idx], float(path_costs_c_i[c_idx]),
                                    heuristics_c[node_idx][c_idx], is_solved_c[node_idx][c_idx],
                                    actions_c[node_idx][c_idx], tcs[node_idx][c_idx], node)
                nodes_c_i.append(node_c)
            node.children = nodes_c_i
            node.t_costs = tcs[node_idx]
            nodes_c.extend(nodes_c_i)
        self.times.record_time("make_nodes", time.time() - start_time)

        # get child nodes by instance
        start_time = time.time()
        nodes_c_by_state: List[List[Node]] = misc_utils.unflatten(nodes_c, split_idxs_c)
        nodes_c_by_inst_state: List[List[List[Node]]] = misc_utils.unflatten(nodes_c_by_state, split_idxs)
        nodes_c_by_inst: List[List[Node]] = []
        for nodes_c_by_inst_state_i in nodes_c_by_inst_state:
            nodes_c_by_inst.append(misc_utils.flatten(nodes_c_by_inst_state_i)[0])

        for instance, nodes_c_by_inst_i in zip(instances, nodes_c_by_inst):
            instance.num_nodes_generated += len(nodes_c_by_inst_i)

        for instance, nodes_by_inst_i in zip(instances, nodes_by_inst):
            instance.nodes_expanded.extend(nodes_by_inst_i)
        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_c_by_inst

    @abstractmethod
    def remove_finished_instances(self, itr_max: int) -> List[I]:
        pass

    def remove_instances(self, test_rem: Callable[[I], bool]) -> List[I]:
        """ Remove instances

        :param test_rem: A Callable that takes an instance as input and returns true if the instance should be removed
        :return: List of removed instances
        """
        instances_remove: List[I] = []
        instances_keep: List[I] = []
        for instance in self.instances:
            if test_rem(instance):
                instances_remove.append(instance)
            else:
                instances_keep.append(instance)

        self.instances = instances_keep

        return instances_remove

    def _create_root_nodes(self, states: List[State], goals: List[Goal], heur_fn: HeurFN_T,
                           compute_init_heur: bool) -> List[Node]:
        if compute_init_heur:
            heuristics: NDArray = heur_fn(states, goals)
        else:
            heuristics: NDArray = np.zeros(len(states)).astype(np.float64)

        root_nodes: List[Node] = []
        is_solved_l: List[bool] = self.env.is_solved(states, goals)
        for state, goal, heuristic, is_solved in zip(states, goals, heuristics, is_solved_l):
            root_node: Node = Node(state, goal, 0.0, heuristic, is_solved, None, None, None)
            root_nodes.append(root_node)

        return root_nodes




def get_path(node: Node) -> Tuple[List[State], List[Action], float]:
    """ Gets path from the start state to the goal state associated with the input node

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
