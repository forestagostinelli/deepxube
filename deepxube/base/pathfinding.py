from typing import Generic, List, Optional, Any, Tuple, Callable, TypeVar, Dict, Type

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal, Action, ActsEnum
from deepxube.base.pathfind_fns import PFNs, PFNsHeurV, PFNsHeurQ, PFNsPolicy
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


def get_path(node: Node) -> Tuple[List[State], List[Action], List[float], float]:
    """ Gets path from the start state to the goal state associated with the input node

    :param node: goal node
    :return: List of states along path, List of actions in path, path cost
    """
    path: List[State] = []
    actions: List[Action] = []
    tcs: List[float] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        assert parent_node.parent_action is not None, "parent_action should not be None"
        assert parent_node.parent_t_cost is not None, "parent_t_cost should not be None"
        actions.append(parent_node.parent_action)
        tcs.append(parent_node.parent_t_cost)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    actions = actions[::-1]
    tcs = tcs[::-1]

    assert sum(tcs) == node.path_cost, "sum of transition costs should equal path cost"
    return path, actions, tcs, node.path_cost


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
        self._nodes_curr: List[Node] = [self.root_node]
        self._nodes_popped: List[Node] = []
        self._edges_popped: List[EdgeQ] = []

    @abstractmethod
    def frontier_size(self) -> int:
        pass

    def get_nodes(self) -> List[Node]:
        return self._nodes_curr.copy()

    def set_next_nodes(self, nodes_next: List[Node]) -> None:
        self._nodes_curr = nodes_next.copy()

    @abstractmethod
    def record_goal(self, nodes: List[Node]) -> None:
        pass

    def add_nodes_popped(self, nodes_popped: List[Node]) -> None:
        self._nodes_popped.extend(nodes_popped)

    def get_nodes_popped(self) -> List[Node]:
        return self._nodes_popped.copy()

    def add_edges_popped(self, edges_popped: List[EdgeQ]) -> None:
        self._edges_popped.extend(edges_popped)

    def get_edges_popped(self) -> List[EdgeQ]:
        return self._edges_popped.copy()

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


I = TypeVar('I', bound=Instance)  # noqa: E741
D = TypeVar('D', bound=Domain)
PFNsT = TypeVar('PFNsT', bound=PFNs)


# pathfinding

class PathFind(Generic[D, PFNsT, I], ABC):
    @staticmethod
    @abstractmethod
    def domain_type() -> Type[D]:
        pass

    @staticmethod
    @abstractmethod
    def pathfind_functions_type() -> Type[PFNsT]:
        pass

    @staticmethod
    @abstractmethod
    def description() -> str:
        pass

    @classmethod
    def get_incompat_reason(cls, domain: Domain, pathfind_fns_t: Type[PFNs]) -> Optional[str]:
        if not isinstance(domain, cls.domain_type()):
            return f"Domain {domain} is not an instance of {cls.domain_type()}"
        elif not issubclass(pathfind_fns_t, cls.pathfind_functions_type()):
            return f"PathFind functions type {pathfind_fns_t} is not a subclass of {cls.pathfind_functions_type()}"

        return None

    def __init__(self, domain: D, pathfind_fns: PFNsT):
        incompat_reason: Optional[str] = self.get_incompat_reason(domain, type(pathfind_fns))
        if incompat_reason is not None:
            raise TypeError(incompat_reason)

        self.domain: D = domain
        self.pathfind_fns: PFNsT = pathfind_fns
        self.instances: List[I] = []
        self.times: Times = Times()
        self.itr: int = 0

    @abstractmethod
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_vals: bool = True) -> List[I]:
        """ Make instances from states and goals

        :param states: List of states
        :param goals: List of goals
        :param inst_infos: Optional list of information to add to an instance
        :param compute_root_vals: If true, compute the values for the root node. Some algorithms may have to ignore this argument and always compute it.
        :return: List of instances
        """
        pass

    def add_instances(self, instances: List[I]) -> None:
        self.instances.extend(instances)

    @abstractmethod
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        pass

    @abstractmethod
    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        pass

    @abstractmethod
    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        pass

    def remove_finished_instances(self, itr_max: int) -> List[I]:
        def remove_instance_fn(inst_in: I) -> bool:
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
        start_time = time.time()
        states: List[State] = []
        goals: List[Goal] = []
        for node in nodes:
            states.append(node.state)
            goals.append(node.goal)

        is_solved_l: List[bool] = self.domain.is_solved(states, goals)
        for node, is_solved in zip(nodes, is_solved_l, strict=True):
            node.is_solved = is_solved

        self.times.record_time("is_solved", time.time() - start_time)

    @abstractmethod
    def _set_node_vals(self, nodes_by_inst: List[List[Node]], instances: List[I]) -> None:
        pass

    def _create_root_nodes(self, states: List[State], goals: List[Goal]) -> List[Node]:
        start_time = time.time()

        root_nodes: List[Node] = []
        for state, goal in zip(states, goals, strict=True):
            root_node: Node = Node(state, goal, 0.0, 0.0, None, None, None, None, None)
            root_nodes.append(root_node)

        self.times.record_time("root", time.time() - start_time)

        return root_nodes

    def _verbose(self, instances: List[I], nodes_by_inst: List[List[Node]]) -> None:
        nodes_flat: List[Node] = misc_utils.flatten(nodes_by_inst)[0]
        if len(nodes_flat) > 0:
            heuristics: List[float] = [node.heuristic for node in nodes_flat]
            path_costs: List[float] = [node.path_cost for node in nodes_flat]
            frontier_sizes: List[int] = [instance.frontier_size() for instance in instances]
            per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
            per_finished: float = 100.0 * float(np.mean([inst.finished() for inst in instances]))

            print(f"Itr: {self.itr}, Heur(PathCost)(Min/Max): "
                  f"{float(np.min(heuristics)):.2E}({float(path_costs[np.argmin(heuristics)]):.2E})/"
                  f"{float(np.max(heuristics)):.2E}({float(path_costs[np.argmax(heuristics)]):.2E}),"
                  f" Frontier sizes(Min/Max): {min(frontier_sizes)}/{max(frontier_sizes)}, %has_soln: {per_has_soln}, %finished: {per_finished}")
        print(f"Times - {self.times.get_time_str()}\n")


# pathfinding on nodes or edges of graph

class InstanceNode(Instance, ABC):
    @abstractmethod
    def filter_expanded_nodes(self, nodes: List[Node]) -> List[Node]:
        pass

    @abstractmethod
    def push_pop_nodes(self, nodes: List[Node], costs: List[float]) -> List[Node]:
        pass


INode = TypeVar('INode', bound=InstanceNode)


class InstanceEdge(Instance, ABC):
    @abstractmethod
    def filter_popped_nodes(self, nodes: List[Node]) -> List[Node]:
        pass

    @abstractmethod
    def push_pop_edges(self, edges: List[EdgeQ], costs: List[float]) -> List[EdgeQ]:
        pass


IEdge = TypeVar('IEdge', bound=InstanceEdge)


class PathFindNode(PathFind[D, PFNsT, INode]):
    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        instances: List[INode] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return [], []

        # pop from open
        start_time = time.time()
        nodes_popped_by_inst: List[List[Node]] = [instance.get_nodes() for instance in instances]
        nodes_popped_flat: List[Node] = misc_utils.flatten(nodes_popped_by_inst)[0]
        self.times.record_time("pop", time.time() - start_time)

        # is solved
        self.set_is_solved(nodes_popped_flat)
        # for node in nodes_popped_flat:
        #    print(node.state, node.is_solved)

        # record goal
        start_time = time.time()
        for instance, nodes in zip(instances, nodes_popped_by_inst, strict=True):
            instance.record_goal(nodes)
        self.times.record_time("goal", time.time() - start_time)

        # expand
        nodes_exp_by_inst: List[List[Node]] = self._expand(instances, nodes_popped_by_inst)

        # eval nodes
        self._set_node_vals(nodes_exp_by_inst, instances)

        # filter expanded nodes
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_exp_by_inst[inst_idx] = instance.filter_expanded_nodes(nodes_exp_by_inst[inst_idx])
        self.times.record_time("filt", time.time() - start_time)

        # get costs
        costs_by_inst: List[List[float]] = self._compute_costs(instances, nodes_exp_by_inst)

        # push
        start_time = time.time()
        nodes_next_by_inst: List[List[Node]] = []
        for instance, nodes_exp, costs in zip(instances, nodes_exp_by_inst, costs_by_inst, strict=True):
            nodes_next_by_inst.append(instance.push_pop_nodes(nodes_exp, costs))
        self.times.record_time("pushpop", time.time() - start_time)

        # get next edges
        start_time = time.time()
        edges_next_flat: List[EdgeQ] = []
        for instance, nodes_next in zip(instances, nodes_next_by_inst, strict=True):
            edges_popped_inst: List[EdgeQ] = []
            for node in nodes_next:
                assert (node.parent is not None) and (node.parent_action is not None) and (node.parent_t_cost is not None)
                edges_popped_inst.append(EdgeQ(node.parent, node.parent_action, node.parent_t_cost + node.heuristic))
            instance.add_edges_popped(edges_popped_inst)
            edges_next_flat.extend(edges_popped_inst)
        self.times.record_time("edges_next", time.time() - start_time)

        start_time = time.time()
        for instance, nodes_next in zip(instances, nodes_next_by_inst, strict=True):
            instance.set_next_nodes(nodes_next)
        self.times.record_time("set_next", time.time() - start_time)

        # verbose
        if verbose:
            self._verbose(instances, nodes_popped_by_inst)

        # update iterations
        self.itr += 1
        for instance in instances:
            instance.itr += 1

        return nodes_popped_flat, edges_next_flat

    def _expand(self, instances: List[INode], nodes_by_inst: List[List[Node]]) -> List[List[Node]]:
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

                edge_dict_val: Optional[Tuple[float, Node]] = node.edge_dict.get(action)
                node_c: Node
                if edge_dict_val is not None:
                    node_c = edge_dict_val[1]
                else:
                    t_cost: float = tcs[node_idx][c_idx]
                    node_c = Node(states_c_l[node_idx][c_idx], goals_c[node_idx][c_idx], float(path_costs_c_i[c_idx]), 0.0, None, None, action, t_cost, node)
                    node.add_edge(action, t_cost, node_c)

                nodes_c_i.append(node_c)
            nodes_c.extend(nodes_c_i)
        self.times.record_time("nodes", time.time() - start_time)

        # get child nodes by instance
        start_time = time.time()
        nodes_c_by_state: List[List[Node]] = misc_utils.unflatten(nodes_c, split_idxs_c)
        nodes_c_by_inst_state: List[List[List[Node]]] = misc_utils.unflatten(nodes_c_by_state, split_idxs)
        nodes_c_by_inst: List[List[Node]] = []
        for nodes_c_by_inst_state_i in nodes_c_by_inst_state:
            nodes_c_by_inst.append(misc_utils.flatten(nodes_c_by_inst_state_i)[0])

        for instance, nodes_by_inst_i, nodes_c_by_inst_i in zip(instances, nodes_by_inst, nodes_c_by_inst, strict=True):
            instance.add_nodes_popped(nodes_by_inst_i)
            instance.num_nodes_generated += len(nodes_c_by_inst_i)

        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_c_by_inst

    @abstractmethod
    def _compute_costs(self, instances: List[INode], nodes_by_inst: List[List[Node]]) -> List[List[float]]:
        pass


class PathFindEdge(PathFind[D, PFNsT, IEdge]):  # TODO add nodes popped
    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        instances: List[IEdge] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return [], []

        # pop from open
        start_time = time.time()
        nodes_popped_by_inst: List[List[Node]] = [instance.get_nodes() for instance in instances]
        nodes_popped_flat: List[Node] = misc_utils.flatten(nodes_popped_by_inst)[0]
        self.times.record_time("pop", time.time() - start_time)

        # is solved
        self.set_is_solved(nodes_popped_flat)

        # record goal
        start_time = time.time()
        for instance, nodes in zip(instances, nodes_popped_by_inst, strict=True):
            instance.record_goal(nodes)
        self.times.record_time("goal", time.time() - start_time)

        # filter popped nodes
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_popped_by_inst[inst_idx] = instance.filter_popped_nodes(nodes_popped_by_inst[inst_idx])
        self.times.record_time("filt", time.time() - start_time)

        # expand
        edges_exp_by_inst: List[List[EdgeQ]] = self._get_edges(nodes_popped_by_inst)

        # get costs
        costs_by_inst: List[List[float]] = self._compute_costs(instances, edges_exp_by_inst)

        # push
        start_time = time.time()
        edges_next_by_inst: List[List[EdgeQ]] = []
        for instance, edges_exp, costs in zip(instances, edges_exp_by_inst, costs_by_inst, strict=True):
            edges_next_by_inst.append(instance.push_pop_edges(edges_exp, costs))
        self.times.record_time("pushpop", time.time() - start_time)

        # get next nodes
        nodes_next_by_inst: List[List[Node]] = self.get_next_nodes(instances, edges_next_by_inst)

        # eval nodes
        self._set_node_vals(nodes_next_by_inst, instances)

        start_time = time.time()
        for instance, nodes_next in zip(instances, nodes_next_by_inst, strict=True):
            instance.set_next_nodes(nodes_next)
        self.times.record_time("set_next", time.time() - start_time)

        # verbose
        if verbose:
            self._verbose(instances, nodes_popped_by_inst)

        # update iterations
        self.itr += 1
        for instance in instances:
            instance.itr += 1

        return nodes_popped_flat, misc_utils.flatten(edges_next_by_inst)[0]

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
            edge_dict_val: Optional[Tuple[float, Node]] = nodes[idx].edge_dict.get(actions[idx])
            node_next: Node
            if edge_dict_val is not None:
                node_next = edge_dict_val[1]
            else:
                node_next = Node(states_next[idx], goals[idx], path_costs_next[idx], 0.0, None, None, actions[idx], tcs[idx], nodes[idx])
                nodes[idx].add_edge(actions[idx], tcs[idx], node_next)
            nodes_next.append(node_next)
        self.times.record_time("nodes", time.time() - start_time)

        # update instances
        start_time = time.time()
        nodes_next_by_inst: List[List[Node]] = misc_utils.unflatten(nodes_next, split_idxs)
        for instance, edges_by_inst_i, nodes_next_by_inst_i in zip(instances, edges_by_inst, nodes_next_by_inst, strict=True):
            instance.add_edges_popped(edges_by_inst_i)
            instance.num_nodes_generated += len(nodes_next_by_inst_i)
        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_next_by_inst

    def _get_edges(self, nodes_by_inst: List[List[Node]]) -> List[List[EdgeQ]]:
        # make edges
        start_time = time.time()
        edges_by_inst: List[List[EdgeQ]] = []

        for nodes in nodes_by_inst:
            edges: List[EdgeQ] = []
            for node in nodes:
                action_vals: Optional[Tuple[List[Action], List[float]]] = None
                if node.q_values is not None:
                    action_vals = node.q_values
                elif node.act_probs is not None:
                    action_vals = node.act_probs

                assert action_vals is not None
                for action, act_val in zip(action_vals[0], action_vals[1], strict=True):
                    edges.append(EdgeQ(node, action, act_val))
            edges_by_inst.append(edges)
        self.times.record_time("edges", time.time() - start_time)

        return edges_by_inst

    @abstractmethod
    def _compute_costs(self, instances: List[IEdge], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        pass


# pathfinding with functions

PFNsP_T = TypeVar('PFNsP_T', bound=PFNsPolicy)
PFNsHV_T = TypeVar('PFNsHV_T', bound=PFNsHeurV)
PFNsHQ_T = TypeVar('PFNsHQ_T', bound=PFNsHeurQ)


class PathFindSetPolicy(PathFind[D, PFNsP_T, I], ABC):
    def _set_node_vals(self, nodes_by_inst: List[List[Node]], instances: List[I]) -> None:
        start_time = time.time()
        nodes: List[Node] = misc_utils.flatten(nodes_by_inst)[0]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions_l, probs_l = self.pathfind_fns.policy(states, goals)

        assert len(actions_l) == len(probs_l) == len(states) == len(goals), \
            f"{len(actions_l)}, {len(probs_l)}, {len(states)}, {len(goals)}"

        for node, actions, probs in zip(nodes, actions_l, probs_l, strict=True):
            assert len(actions) == len(probs), f"{len(actions)}, {len(probs)}"
            node.act_probs = (actions, probs)

        self.times.record_time("policy", time.time() - start_time)


class PathFindSetHeurV(PathFind[D, PFNsHV_T, I], ABC):
    def _set_node_vals(self, nodes_by_inst: List[List[Node]], instances: List[I]) -> None:
        start_time = time.time()
        nodes: List[Node] = misc_utils.flatten(nodes_by_inst)[0]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]

        heuristics: List[float] = self.pathfind_fns.heurv(states, goals)

        assert len(heuristics) == len(states) == len(goals), \
            f"{len(heuristics)}, {len(states)}, {len(goals)}"

        for node, heuristic in zip(nodes, heuristics, strict=True):
            node.heuristic = heuristic

        self.times.record_time("heur", time.time() - start_time)


class PathFindSetHeurQ(PathFind[D, PFNsHQ_T, I], ABC):
    def _set_node_vals(self, nodes_by_inst: List[List[Node]], instnaces: List[I]) -> None:
        start_time = time.time()
        nodes: List[Node] = misc_utils.flatten(nodes_by_inst)[0]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions_l: List[List[Action]] = self.get_state_actions(states, goals)

        self.times.record_time("actions", time.time() - start_time)

        start_time = time.time()
        qvals_l: List[List[float]] = self.pathfind_fns.heurq(states, goals, actions_l)
        heuristics: List[float] = [min(x) for x in qvals_l]

        assert len(heuristics) == len(actions_l) == len(qvals_l) == len(states) == len(goals), \
            f"{len(heuristics)}, {len(actions_l)}, {len(qvals_l)}, {len(states)}, {len(goals)}"

        for node, heuristic, actions, qvals in zip(nodes, heuristics, actions_l, qvals_l, strict=True):
            node.heuristic = heuristic
            node.q_values = (actions, qvals)

        self.times.record_time("heur", time.time() - start_time)


# pathfinding with action spaces

DActsEnum = TypeVar('DActsEnum', bound=ActsEnum)


class PathFindActsEnum(PathFind[DActsEnum, PFNsT, I], ABC):
    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        return self.domain.expand(states)

    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        return self.domain.get_state_actions(states)


class PathFindActsPolicy(PathFind[D, PFNsP_T, I], ABC):
    def __init__(self, *args: Any, num_rand_edges: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_rand_edges: int = num_rand_edges

    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        actions_l: List[List[Action]] = self._get_actions(states, goals)

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

    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        return self._get_actions(states, goals)

    def _get_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        actions_l: List[List[Action]] = self.pathfind_fns.policy(states, goals)[0]

        if self.num_rand_edges > 0:
            states_rep_l: List[List[State]] = [[state] * self.num_rand_edges for state in states]
            states_rep_flat, split_idxs = misc_utils.flatten(states_rep_l)

            actions_rand_flat: List[Action] = self.domain.sample_state_action(states_rep_flat)

            actions_rand_l: List[List[Action]] = misc_utils.unflatten(actions_rand_flat, split_idxs)
            for state_idx in range(len(states)):
                actions_l[state_idx].extend(actions_rand_l[state_idx])

        return actions_l


# pathfinding supervised (for training)

class PathFindSup(PathFind[D, PFNs, I]):
    """ Use the path cost of a random walk as the learning target.
    See Chervov, Alexander, et al. "A Machine Learning Approach That Beats Large Rubik's Cubes." NeurIPS(2025).

    """
    @staticmethod
    def pathfind_functions_type() -> Type[PFNs]:
        return PFNs

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_vals: bool = True) -> List[I]:
        raise NotImplementedError

    def expand_states(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        raise NotImplementedError

    def get_state_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        raise NotImplementedError

    @abstractmethod
    def make_instances_sup(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[I]:
        """ Make nodes/edges with labels

        """
        pass

    def _set_node_vals(self, nodes_by_inst: List[List[Node]], instances: List[I]) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
