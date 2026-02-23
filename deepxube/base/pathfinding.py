from typing import Generic, List, Optional, Any, Tuple, Callable, TypeVar, Dict, Type

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal, Action, ActsEnum
from deepxube.base.heuristic import HeurFnV, HeurFnQ, HeurFn, PolicyFn
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times

from abc import ABC, abstractmethod
import numpy as np
import time


# pathfinding data structures

class Node:
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'q_values', 'act_probs', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'edge_dict', 'backup_val']

    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, q_values: Optional[Tuple[List[Action], List[float]]],
                 is_solved: Optional[bool], parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['Node']):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.q_values: Optional[Tuple[List[Action], List[float]]] = q_values
        self.act_probs: Optional[Tuple[List[Action], List[float]]] = None
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


class EdgeQ:
    __slots__ = ['node', 'action', 'q_val']

    def __init__(self, node: Node, action: Action, q_val: float):
        self.node: Node = node
        self.action: Action = action
        self.q_val: float = q_val


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


# pathfinding

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
        :param compute_root_heur: If true, compute the heuristic value of the root node. Some algorithms may have to ignore this argument and always compute it.
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


# pathfinding with functions

H = TypeVar('H', bound=HeurFn)


class PathFindHasHeur(PathFind[D, I], Generic[D, I, H], ABC):
    @staticmethod
    @abstractmethod
    def heur_fn_type() -> Type[H]:
        pass

    def __init__(self, domain: D):
        super().__init__(domain)
        self.heur_fn: Optional[H] = None

    def set_heur_fn(self, heur_fn: H) -> None:
        self.heur_fn = heur_fn

    @abstractmethod
    def _set_node_heurs(self, nodes: List[Node]) -> None:
        pass

    def _create_root_nodes_heur(self, states: List[State], goals: List[Goal], compute_root_heur: bool) -> List[Node]:
        start_time = time.time()

        root_nodes: List[Node] = []
        for state, goal in zip(states, goals, strict=True):
            root_node: Node = Node(state, goal, 0.0, 0.0, None, None, None, None, None)
            root_nodes.append(root_node)

        self.times.record_time("root", time.time() - start_time)

        if compute_root_heur:
            self._set_node_heurs(root_nodes)

        return root_nodes


class PathFindHasPolicy(PathFind[D, I], ABC):
    def __init__(self, domain: D):
        super().__init__(domain)
        self.policy_fn: Optional[PolicyFn] = None
        self.num_acts_samp: Optional[int] = None

    def set_policy_fn(self, policy_fn: PolicyFn, num_acts_samp: int) -> None:
        self.policy_fn = policy_fn
        self.num_acts_samp = num_acts_samp

    def _get_action_probs(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
        assert self.policy_fn is not None
        assert self.num_acts_samp is not None
        return self.policy_fn(self.domain, states, goals, self.num_acts_samp)

    def _create_root_nodes_policy(self, states: List[State], goals: List[Goal]) -> List[Node]:
        start_time = time.time()

        root_nodes: List[Node] = []
        for state, goal in zip(states, goals, strict=True):
            root_node: Node = Node(state, goal, 0.0, 0.0, None, None, None, None, None)
            root_nodes.append(root_node)

        self.times.record_time("root", time.time() - start_time)

        return root_nodes

    def _set_node_act_probs(self, nodes: List[Node]) -> None:
        start_time = time.time()
        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions_l, probs_l = self._get_action_probs(states, goals)

        assert len(actions_l) == len(probs_l) == len(states) == len(goals), \
            f"{len(actions_l)}, {len(probs_l)}, {len(states)}, {len(goals)}"

        for node, actions, probs in zip(nodes, actions_l, probs_l, strict=True):
            node.act_probs = (actions, probs)

        self.times.record_time("policy_fn", time.time() - start_time)


# pathfinding on nodes or edges of graph

class InstanceNode(Instance, ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.nodes_popped: List[Node] = []


INode = TypeVar('INode', bound=InstanceNode)


class InstanceEdge(Instance, ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.edges_popped: List[EdgeQ] = []


IEdge = TypeVar('IEdge', bound=InstanceEdge)


class PathFindNode(PathFind[D, INode]):
    @abstractmethod
    def step(self, verbose: bool = False) -> List[Node]:
        pass

    @abstractmethod
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        pass

    def _expand_nodes(self, instances: List[INode], nodes_by_inst: List[List[Node]]) -> List[List[Node]]:
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

        # get children nodes
        start_time = time.time()
        nodes_c: List[Node] = []
        for node_idx, node in enumerate(nodes):
            path_costs_c_i: NDArray = node.path_cost + np.array(tcs[node_idx])
            nodes_c_i: List[Node] = []
            for c_idx in range(len(states_c_l[node_idx])):
                action: Action = actions[node_idx][c_idx]
                t_cost: float = tcs[node_idx][c_idx]
                node_c: Node = Node(states_c_l[node_idx][c_idx], goals_c[node_idx][c_idx], float(path_costs_c_i[c_idx]), 0.0, None,
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


class PathFindEdge(PathFind[D, IEdge]):
    @abstractmethod
    def step(self, verbose: bool = False) -> List[EdgeQ]:
        pass

    @abstractmethod
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        pass

    def get_next_nodes(self, instances: List[IEdge], edges_by_inst: List[List[EdgeQ]]) -> List[List[Node]]:
        if len(instances) == 0:
            return []
        start_time = time.time()
        # flatten
        edges, split_idxs = misc_utils.flatten(edges_by_inst)
        nodes: List[Node] = [edge.node for edge in edges]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        path_costs: List[float] = [node.path_cost for node in nodes]
        actions: List[Action] = [edge.action for edge in edges]

        # next states
        states_next, tcs = self.domain.next_state(states, actions)
        path_costs_next: List[float] = (np.array(path_costs) + np.array(tcs)).tolist()
        self.times.record_time("next_state", time.time() - start_time)

        # next nodes
        start_time = time.time()
        nodes_next: List[Node] = []
        for idx in range(len(edges)):
            node_next: Node = Node(states_next[idx], goals[idx], path_costs_next[idx], 0.0, None, None, actions[idx], tcs[idx], nodes[idx])

            nodes[idx].add_edge(actions[idx], tcs[idx], node_next)
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


# pathfinding with enumerable action space


DActsEnum = TypeVar('DActsEnum', bound=ActsEnum)


class PathFindNodeActsEnum(PathFindNode[DActsEnum, INode], ABC):
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        return self.domain.expand(states)


class PathFindEdgeActsEnum(PathFindEdge[DActsEnum, IEdge], ABC):
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        return self.domain.get_state_actions(states)


# pathfinding with heuristic function

class PathFindNodeHasHeur(PathFindNode[D, INode], PathFindHasHeur[D, INode, HeurFnV], ABC):
    @staticmethod
    def heur_fn_type() -> Type[HeurFnV]:
        return HeurFnV

    def _get_heur_vals(self, states: List[State], goals: List[Goal]) -> List[float]:
        assert self.heur_fn is not None
        return self.heur_fn(states, goals)

    def _set_node_heurs(self, nodes: List[Node]) -> None:
        start_time = time.time()
        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        heuristics: List[float] = self._get_heur_vals(states, goals)

        assert len(heuristics) == len(states) == len(goals), \
            f"{len(heuristics)}, {len(states)}, {len(goals)}"

        for node, heuristic in zip(nodes, heuristics, strict=True):
            node.heuristic = heuristic

        self.times.record_time("heur_fn", time.time() - start_time)


class PathFindEdgeHasHeur(PathFindEdge[D, IEdge], PathFindHasHeur[D, IEdge, HeurFnQ], ABC):
    @staticmethod
    def heur_fn_type() -> Type[HeurFnQ]:
        return HeurFnQ

    def _get_acts_qvals(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]],]:
        actions_l: List[List[Action]] = self.get_state_actions(states, goals)
        assert self.heur_fn is not None
        qvals_l: List[List[float]] = self.heur_fn(states, goals, actions_l)
        return actions_l, qvals_l

    def _set_node_heurs(self, nodes: List[Node]) -> None:
        start_time = time.time()
        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions_l, qvals_l = self._get_acts_qvals(states, goals)
        heuristics: List[float] = [min(x) for x in qvals_l]

        assert len(heuristics) == len(actions_l) == len(qvals_l) == len(states) == len(goals), \
            f"{len(heuristics)}, {len(actions_l)}, {len(qvals_l)}, {len(states)}, {len(goals)}"

        for node, heuristic, actions, qvals in zip(nodes, heuristics, actions_l, qvals_l, strict=True):
            node.heuristic = heuristic
            node.q_values = (actions, qvals)

        self.times.record_time("heur_fn", time.time() - start_time)


# pathfinding with policy

class PathFindNodeActsPolicy(PathFindNode[D, INode], PathFindHasPolicy[D, INode], ABC):
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        actions_l: List[List[Action]] = self._get_action_probs(states, goals)[0]

        # repeat states according to actions
        actions_flat, split_idxs = misc_utils.flatten(actions_l)

        states_flat: List[State] = []
        for state, actions in zip(states, actions_l, strict=True):
            states_flat.extend([state] * len(actions))

        assert len(states_flat) == len(actions_flat), f"{len(states_flat)}, {len(actions_flat)}"

        # get next states
        states_exp_flat, tcs_flat = self.domain.next_state(states_flat, actions_flat)

        # unflatten
        states_exp: List[List[State]] = misc_utils.unflatten(states_exp_flat, split_idxs)
        tcs_l: List[List[float]] = misc_utils.unflatten(tcs_flat, split_idxs)

        return states_exp, actions_l, tcs_l


class PathFindEdgeActsPolicy(PathFindEdge[D, IEdge], PathFindHasPolicy[D, IEdge], ABC):
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        return self._get_action_probs(states, goals)[0]


# pathfinding supervised (for training)

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