from typing import Generic, List, Optional, Any, Tuple, Callable, TypeVar, Dict, cast, Type

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal, Action, ActsEnum
from deepxube.base.heuristic import HeurFnV, HeurFnQ, HeurFn
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times

from abc import ABC, abstractmethod
import numpy as np
import time


class Node:
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'q_values', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'edge_dict', 'backup_val']

    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, q_values: Optional[Tuple[List[Action], List[float]]],
                 is_solved: Optional[bool], parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['Node']):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.q_values: Optional[Tuple[List[Action], List[float]]] = q_values
        self.is_solved: Optional[bool] = is_solved
        self.parent_action: Optional[Action] = parent_action
        self.parent_t_cost: Optional[float] = parent_t_cost
        self.parent: Optional[Node] = parent
        self.edge_dict: Dict[Action, Tuple[float, Node]] = dict()
        self.backup_val: float = np.inf

    def add_edge(self, action: Action, t_cost: float, node_next: "Node") -> None:
        assert action not in self.edge_dict.keys()
        self.edge_dict[action] = (t_cost, node_next)

    def bellman_backup(self) -> float:
        assert self.is_solved is not None

        if self.is_solved:
            self.backup_val = 0.0
        else:
            if len(self.edge_dict) > 0:
                self.backup_val = min(tc + node_c.heuristic for tc, node_c in self.edge_dict.values())
        return self.backup_val

    def upper_bound_parent_path(self, ctg_ub: float) -> None:
        self.backup_val = min(self.backup_val, ctg_ub)
        if self.parent is not None:
            assert self.parent_t_cost is not None
            self.parent.upper_bound_parent_path(ctg_ub + self.parent_t_cost)

    def tree_backup(self) -> float:
        if (self.is_solved is not None) and self.is_solved:
            self.backup_val = 0.0
        else:
            if len(self.edge_dict) == 0:
                self.backup_val = max(self.heuristic, 0.0)
            else:
                self.backup_val = min(tc + node_c.tree_backup() for tc, node_c in self.edge_dict.values())

        return self.backup_val

    def backup_act(self, action: Action) -> float:
        assert self.is_solved is not None
        if self.is_solved:
            return 0.0
        else:
            tc, node_next = self.edge_dict[action]
            # assert node_next.q_values is not None
            if node_next.backup_val < np.inf:
                return tc + node_next.backup_val
            else:
                return tc + node_next.heuristic

    def get_all_descendants(self) -> List['Node']:
        """ Get all descendants of node (excluding self)

        :return: List of nodes that are descendants
        """
        fifo: List[Node] = [x[1] for x in self.edge_dict.values()]
        descendants: List[Node] = []
        while len(fifo) > 0:
            descendant: Node = fifo.pop(0)
            for _, descendant_c in descendant.edge_dict.values():
                fifo.append(descendant_c)
            descendants.append(descendant)
        return descendants


class Instance(ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        self.root_node: Node = root_node
        self.itr: int = 0  # updater with every pathfinding iteration
        self.num_nodes_generated: int = 0
        self.inst_info: Any = inst_info
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

    @abstractmethod
    def finished(self) -> bool:
        pass


I = TypeVar('I', bound=Instance)
D = TypeVar('D', bound=Domain)


class PathFind(Generic[D, I], ABC):
    @staticmethod
    @abstractmethod
    def domain_type() -> Type[D]:
        pass

    def __init__(self, domain: D):
        assert isinstance(domain, self.domain_type()), f"Domain {domain} must be an instance of {self.domain_type()}."
        self.domain: D = domain
        self.instances: List[I] = []
        self.times: Times = Times()
        self.itr: int = 0

    @abstractmethod
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True) -> List[I]:
        """ Make instances from states and goals

        :param states: List of states
        :param goals: List of goals
        :param inst_infos: Optional list of information to add to an instance
        :param compute_root_heur: If true, compute the heuristic value of the root node
        :return: List of instances
        """
        pass

    def add_instances(self, instances: List[I]) -> None:
        self.instances.extend(instances)

    @abstractmethod
    def step(self, verbose: bool = False) -> Any:
        pass

    def remove_finished_instances(self, itr_max: int) -> List[I]:
        def remove_instance_fn(inst_in: Instance) -> bool:
            if inst_in.finished():
                return True
            if inst_in.itr >= itr_max:
                return True
            return False

        return self.remove_instances(remove_instance_fn)

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

    def set_is_solved(self, nodes: List[Node]) -> None:
        states: List[State] = []
        goals: List[Goal] = []
        for node in nodes:
            states.append(node.state)
            goals.append(node.goal)

        is_solved_l: List[bool] = self.domain.is_solved(states, goals)
        for node, is_solved in zip(nodes, is_solved_l, strict=True):
            node.is_solved = is_solved


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


H = TypeVar('H', bound=HeurFn)


class PathFindHeur(PathFind[D, I], Generic[D, I, H], ABC):
    @staticmethod
    @abstractmethod
    def heur_fn_type() -> Type[H]:
        pass

    def __init__(self, domain: D):
        super().__init__(domain)
        self.heur_fn: Optional[H] = None

    def set_heur_fn(self, heur_fn: H) -> None:
        self.heur_fn = heur_fn


class PathFindSup(PathFind[D, I]):
    """ Use the path cost of a random walk as the learning target.
    See Chervov, Alexander, et al. "A Machine Learning Approach That Beats Large Rubik's Cubes." NeurIPS(2025).

    """
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None,
                       compute_root_heur: bool = True) -> List[I]:
        raise NotImplementedError

    @abstractmethod
    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[I]:
        """ Make instances from a random walk

        """
        pass


class InstanceV(Instance, ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.nodes_popped: List[Node] = []


IV = TypeVar('IV', bound=InstanceV)


class PathFindV(PathFind[D, IV]):
    @abstractmethod
    def step(self, verbose: bool = False) -> List[Node]:
        pass

    @abstractmethod
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        pass

    @abstractmethod
    def _get_heur_vals(self, states: List[State], goals: List[Goal]) -> List[float]:
        pass

    def _expand_nodes(self, instances: List[IV], nodes_by_inst: List[List[Node]]) -> List[List[Node]]:
        start_time = time.time()
        # flatten (for speed)
        nodes: List[Node]
        split_idxs: List[int]
        nodes, split_idxs = misc_utils.flatten(nodes_by_inst)

        if len(nodes) == 0:
            return [[]]

        # Get children of nodes
        states: List[State] = [x.state for x in nodes]
        goals: List[Goal] = [x.goal for x in nodes]

        states_c_l: List[List[State]]
        actions: List[List[Action]]
        tcs: List[List[float]]
        states_c_l, actions, tcs = self.expand_states(states, goals)

        goals_c: List[List[Goal]] = [[node.goal] * len(state_c) for node, state_c in zip(nodes, states_c_l, strict=True)]
        states_c_flat: List[State]
        states_c_flat, split_idxs_c = misc_utils.flatten(states_c_l)
        goals_c_flat, _ = misc_utils.flatten(goals_c)
        self.times.record_time("expand", time.time() - start_time)

        # Get is_solved on all states at once (for speed)
        # start_time = time.time()
        # is_solved_c_flat: List[bool] = self.domain.is_solved(states_c_flat, goals_c_flat)
        # is_solved_c: List[List[bool]] = misc_utils.unflatten(is_solved_c_flat, split_idxs_c)
        # self.times.record_time("is_solved", time.time() - start_time)

        # heuristic function
        start_time = time.time()
        heuristics_c_flat: List[float] = self._get_heur_vals(states_c_flat, goals_c_flat)
        assert len(heuristics_c_flat) == len(states_c_flat) == len(goals_c_flat), \
            f"{len(heuristics_c_flat)}, {len(states_c_flat)}, {len(goals_c_flat)}"
        heuristics_c: List[List[float]] = misc_utils.unflatten(heuristics_c_flat, split_idxs_c)
        self.times.record_time("heur", time.time() - start_time)

        # get children nodes
        start_time = time.time()
        nodes_c: List[Node] = []
        for node_idx, node in enumerate(nodes):
            path_costs_c_i: NDArray = node.path_cost + np.array(tcs[node_idx])
            nodes_c_i: List[Node] = []
            for c_idx in range(len(states_c_l[node_idx])):
                action: Action = actions[node_idx][c_idx]
                t_cost: float = tcs[node_idx][c_idx]
                node_c: Node = Node(states_c_l[node_idx][c_idx], goals_c[node_idx][c_idx], float(path_costs_c_i[c_idx]), heuristics_c[node_idx][c_idx], None,
                                    None, action, t_cost, node)
                node.add_edge(action, t_cost, node_c)
                nodes_c_i.append(node_c)
            nodes_c.extend(nodes_c_i)
        self.times.record_time("make_nodes", time.time() - start_time)

        # get child nodes by instance
        start_time = time.time()
        nodes_c_by_state: List[List[Node]] = misc_utils.unflatten(nodes_c, split_idxs_c)
        nodes_c_by_inst_state: List[List[List[Node]]] = misc_utils.unflatten(nodes_c_by_state, split_idxs)
        nodes_c_by_inst: List[List[Node]] = []
        for nodes_c_by_inst_state_i in nodes_c_by_inst_state:
            nodes_c_by_inst.append(misc_utils.flatten(nodes_c_by_inst_state_i)[0])

        for instance, nodes_by_inst_i, nodes_c_by_inst_i in zip(instances, nodes_by_inst, nodes_c_by_inst, strict=True):
            instance.nodes_popped.extend(nodes_by_inst_i)
            instance.num_nodes_generated += len(nodes_c_by_inst_i)

        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_c_by_inst

    def _create_root_nodes(self, states: List[State], goals: List[Goal], compute_root_heur: bool = True) -> List[Node]:
        start_time = time.time()
        heuristics: List[float]
        if compute_root_heur:
            heuristics = self._get_heur_vals(states, goals)
        else:
            heuristics = [0.0 for _ in states]

        root_nodes: List[Node] = []
        for state, goal, heuristic in zip(states, goals, heuristics, strict=True):
            root_node: Node = Node(state, goal, 0.0, heuristic, None, None, None, None, None)
            root_nodes.append(root_node)
        self.times.record_time("root", time.time() - start_time)

        return root_nodes


class PathFindVExpandEnum(PathFindV[ActsEnum, IV], ABC):
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        return self.domain.expand(states)


class PathFindVHeur(PathFindV[D, IV], PathFindHeur[D, IV, HeurFnV], ABC):
    @staticmethod
    def heur_fn_type() -> Type[HeurFnV]:
        return HeurFnV

    def _get_heur_vals(self, states: List[State], goals: List[Goal]) -> List[float]:
        assert self.heur_fn is not None
        return self.heur_fn(states, goals)


class EdgeQ:
    __slots__ = ['node', 'action', 'q_val']

    def __init__(self, node: Node, action: Optional[Action], q_val: float):
        self.node: Node = node
        self.action: Optional[Action] = action
        self.q_val: float = q_val


class InstanceQ(Instance, ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.edges_popped: List[EdgeQ] = []


IQ = TypeVar('IQ', bound=InstanceQ)


class PathFindQ(PathFind[D, IQ]):
    @abstractmethod
    def step(self, verbose: bool = False) -> List[EdgeQ]:
        pass

    @abstractmethod
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        pass

    def get_next_nodes(self, instances: List[IQ], edges_by_inst: List[List[EdgeQ]]) -> List[List[Node]]:
        if len(instances) == 0:
            return []
        start_time = time.time()
        # flatten
        edges, split_idxs = misc_utils.flatten(edges_by_inst)
        nodes: List[Node] = [edge.node for edge in edges]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions: List[Optional[Action]] = [node_act.action for node_act in edges]
        path_costs: List[float] = [popped_node.path_cost for popped_node in nodes]

        # next states
        states_next = states.copy()
        tcs: List[float] = [0.0] * len(states_next)
        idxs_op: List[int] = [idx for idx, action in enumerate(actions) if action is not None]
        if len(idxs_op) > 0:
            states_op: List[State] = [states[idx_op] for idx_op in idxs_op]
            actions_op: List[Action] = [cast(Action, actions[idx_op]) for idx_op in idxs_op]
            states_next_op, tcs_op = self.domain.next_state(states_op, actions_op)
            for idx_op, state, tc in zip(idxs_op, states_next_op, tcs_op):
                states_next[idx_op] = state
                tcs[idx_op] = tc
        path_costs_next: List[float] = (np.array(path_costs) + np.array(tcs)).tolist()
        self.times.record_time("next_state", time.time() - start_time)

        # heuristic function
        start_time = time.time()
        q_vals_next, actions_next_l = self.get_qvals_acts(states_next, goals)
        heurs_next: List[float] = [min(x) for x in q_vals_next]
        self.times.record_time("heur", time.time() - start_time)

        # next nodes
        start_time = time.time()
        nodes_next: List[Node] = []
        for idx in range(len(edges)):
            node_next: Node
            action_i: Optional[Action] = actions[idx]
            if action_i is not None:
                node_next = Node(states_next[idx], goals[idx], path_costs_next[idx], heurs_next[idx], (actions_next_l[idx], q_vals_next[idx]), None, action_i,
                                 tcs[idx], nodes[idx])
                nodes[idx].add_edge(action_i, tcs[idx], node_next)
            else:
                node_next = nodes[idx]
                node_next.q_values = (actions_next_l[idx], q_vals_next[idx])
            nodes_next.append(node_next)
        self.times.record_time("make_nodes", time.time() - start_time)

        # updater instances
        start_time = time.time()
        nodes_next_by_inst: List[List[Node]] = misc_utils.unflatten(nodes_next, split_idxs)
        for instance, edges_by_inst_i, nodes_next_by_inst_i in zip(instances, edges_by_inst, nodes_next_by_inst, strict=True):
            instance.edges_popped.extend(edges_by_inst_i)
            instance.num_nodes_generated += len(nodes_next_by_inst_i)
        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_next_by_inst

    def get_qvals_acts(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[float]], List[List[Action]]]:
        actions_l: List[List[Action]] = self.get_state_actions(states, goals)
        qvals_l: List[List[float]] = self._get_heur_vals(states, goals, actions_l)
        return qvals_l, actions_l

    def _create_root_nodes(self, states: List[State], goals: List[Goal], compute_root_heur: bool = True) -> List[Node]:
        start_time = time.time()

        qvals_l: List[List[float]]
        actions_l: List[List[Action]]
        heuristics: List[float]
        if compute_root_heur:
            qvals_l, actions_l = self.get_qvals_acts(states, goals)
            heuristics = [min(x) for x in qvals_l]
        else:
            actions_l = self.get_state_actions(states, goals)
            qvals_l = [[0.0] * len(actions) for _, actions in zip(states, actions_l, strict=True)]
            heuristics = [0.0 for _ in states]

        root_nodes: List[Node] = []
        for state, goal, heuristic, actions, qvals in zip(states, goals, heuristics, actions_l, qvals_l, strict=True):
            root_node: Node = Node(state, goal, 0.0, heuristic, (actions, qvals), None, None, None, None)
            root_nodes.append(root_node)
        self.times.record_time("root", time.time() - start_time)

        return root_nodes

    @abstractmethod
    def _get_heur_vals(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        pass


class PathFindQExpandEnum(PathFindQ[ActsEnum, IQ], ABC):
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        return self.domain.get_state_actions(states)


class PathFindQHeur(PathFindQ[D, IQ], PathFindHeur[D, IQ, HeurFnQ], ABC):
    @staticmethod
    def heur_fn_type() -> Type[HeurFnQ]:
        return HeurFnQ

    def _get_heur_vals(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        assert self.heur_fn is not None
        return self.heur_fn(states, goals, actions_l)
