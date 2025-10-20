from typing import Generic, List, Optional, Any, Tuple, Callable, TypeVar, Dict

from numpy.typing import NDArray

from deepxube.base.env import Env, State, Goal, Action, EnvEnumerableActs
from deepxube.base.heuristic import HeurFnV, HeurFnQ
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times

from abc import ABC, abstractmethod
import numpy as np
import time


class Node(ABC):
    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: Optional[bool],
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['Node']):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.is_solved: Optional[bool] = is_solved
        self.parent_action: Optional[Action] = parent_action
        self.parent_t_cost: Optional[float] = parent_t_cost
        self.parent: Optional[Node] = parent

    @abstractmethod
    def backup(self) -> float:
        pass


N = TypeVar('N', bound=Node)


class Instance(ABC, Generic[N]):
    def __init__(self, root_node: N, inst_info: Any):
        self.root_node: N = root_node
        self.itr: int = 0  # updater with every pathfinding iteration
        self.num_nodes_generated: int = 0
        self.inst_info: Any = inst_info
        self.goal_node: Optional[N] = None

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
E = TypeVar('E', bound=Env)


class PathFind(ABC, Generic[E, N, I]):
    def __init__(self, env: E):
        self.env: E = env
        self.instances: List[I] = []
        self.times: Times = Times()
        self.itr: int = 0

    def add_instances(self, instances: List[I]) -> None:
        self.instances.extend(instances)

    @abstractmethod
    def step(self) -> Any:
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

    def set_is_solved(self, nodes: List[N]) -> None:
        states: List[State] = []
        goals: List[Goal] = []
        for node in nodes:
            states.append(node.state)
            goals.append(node.goal)

        is_solved_l: List[bool] = self.env.is_solved(states, goals)
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


class NodeV(Node):
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'children', 't_costs', 'bellman_backup_val']

    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: Optional[bool],
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['NodeV']):
        super().__init__(state, goal, path_cost, heuristic, is_solved, parent_action, parent_t_cost, parent)
        self.parent: Optional[NodeV] = parent
        self.children: Optional[List[NodeV]] = None
        self.t_costs: Optional[List[float]] = None
        self.bellman_backup_val: Optional[float] = None

    def backup(self) -> float:
        assert self.is_solved is not None
        if self.is_solved:
            self.bellman_backup_val = 0.0
        else:
            assert self.children is not None
            if len(self.children) == 0:
                self.bellman_backup_val = self.heuristic
            else:
                assert self.t_costs is not None

                self.bellman_backup_val = np.inf
                for node_c, tc in zip(self.children, self.t_costs, strict=True):
                    self.bellman_backup_val = min(self.bellman_backup_val, tc + node_c.heuristic)

        return self.bellman_backup_val

    def upper_bound_parent_path(self, ctg_ub: float) -> None:
        assert self.bellman_backup_val is not None
        self.bellman_backup_val = min(self.bellman_backup_val, ctg_ub)
        if self.parent is not None:
            assert self.parent_t_cost is not None
            self.parent.upper_bound_parent_path(ctg_ub + self.parent_t_cost)


class PathFindV(PathFind[E, NodeV, I]):
    def __init__(self, env: E, heur_fn: HeurFnV):
        super().__init__(env)
        self.heur_fn: HeurFnV = heur_fn

    @abstractmethod
    def step(self, verbose: bool = False) -> List[NodeV]:
        pass

    def expand_nodes(self, instances: List[I], nodes_by_inst: List[List[NodeV]]) -> List[List[NodeV]]:
        start_time = time.time()
        # flatten (for speed)
        nodes: List[NodeV]
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
        states_c_l, actions, tcs = self.expand(states, goals)

        goals_c: List[List[Goal]] = [[node.goal] * len(state_c) for node, state_c in
                                     zip(nodes, states_c_l, strict=True)]
        states_c_flat: List[State]
        states_c_flat, split_idxs_c = misc_utils.flatten(states_c_l)
        goals_c_flat, _ = misc_utils.flatten(goals_c)
        self.times.record_time("expand", time.time() - start_time)

        # Get is_solved on all states at once (for speed)
        # start_time = time.time()
        # is_solved_c_flat: List[bool] = self.env.is_solved(states_c_flat, goals_c_flat)
        # is_solved_c: List[List[bool]] = misc_utils.unflatten(is_solved_c_flat, split_idxs_c)
        # self.times.record_time("is_solved", time.time() - start_time)

        # heuristic function
        start_time = time.time()
        heuristics_c_flat: List[float] = self.heur_fn(states_c_flat, goals_c_flat)
        assert len(heuristics_c_flat) == len(states_c_flat) == len(goals_c_flat), \
            f"{len(heuristics_c_flat)}, {len(states_c_flat)}, {len(goals_c_flat)}"
        heuristics_c: List[List[float]] = misc_utils.unflatten(heuristics_c_flat, split_idxs_c)
        self.times.record_time("heur", time.time() - start_time)

        # get children nodes
        start_time = time.time()
        nodes_c: List[NodeV] = []
        for node_idx, node in enumerate(nodes):
            path_costs_c_i: NDArray = node.path_cost + np.array(tcs[node_idx])
            nodes_c_i: List[NodeV] = []
            for c_idx in range(len(states_c_l[node_idx])):
                node_c: NodeV = NodeV(states_c_l[node_idx][c_idx], goals_c[node_idx][c_idx],
                                      float(path_costs_c_i[c_idx]), heuristics_c[node_idx][c_idx],
                                      None, actions[node_idx][c_idx], tcs[node_idx][c_idx], node)
                nodes_c_i.append(node_c)
            node.children = nodes_c_i
            node.t_costs = tcs[node_idx]
            nodes_c.extend(nodes_c_i)
        self.times.record_time("make_nodes", time.time() - start_time)

        # get child nodes by instance
        start_time = time.time()
        nodes_c_by_state: List[List[NodeV]] = misc_utils.unflatten(nodes_c, split_idxs_c)
        nodes_c_by_inst_state: List[List[List[NodeV]]] = misc_utils.unflatten(nodes_c_by_state, split_idxs)
        nodes_c_by_inst: List[List[NodeV]] = []
        for nodes_c_by_inst_state_i in nodes_c_by_inst_state:
            nodes_c_by_inst.append(misc_utils.flatten(nodes_c_by_inst_state_i)[0])

        for instance, nodes_c_by_inst_i in zip(instances, nodes_c_by_inst, strict=True):
            instance.num_nodes_generated += len(nodes_c_by_inst_i)

        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_c_by_inst

    def create_root_nodes(self, states: List[State], goals: List[Goal], compute_init_heur: bool = True) -> List[NodeV]:
        start_time = time.time()
        heuristics: List[float]
        if compute_init_heur:
            heuristics = self.heur_fn(states, goals)
        else:
            heuristics = [0.0 for _ in states]

        root_nodes: List[NodeV] = []
        for state, goal, heuristic in zip(states, goals, heuristics, strict=True):
            root_node: NodeV = NodeV(state, goal, 0.0, heuristic, None, None, None, None)
            root_nodes.append(root_node)
        self.times.record_time("root", time.time() - start_time)

        return root_nodes

    @abstractmethod
    def expand(self, states: List[State],
               goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        pass


# mixins
class PathFindVExpandEnum(PathFindV[EnvEnumerableActs, I], ABC):
    def expand(self, states: List[State],
               goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        return self.env.expand(states)


class NodeQ(Node):
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'actions', 'q_values', 'act_dict', 'bellman_backup_val']

    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: Optional[bool],
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['NodeQ'],
                 actions: List[Action], q_values: List[float]):
        super().__init__(state, goal, path_cost, heuristic, is_solved, parent_action, parent_t_cost, parent)
        self.parent: Optional[NodeQ] = parent
        self.actions: List[Action] = actions
        self.q_values: List[float] = q_values
        self.act_dict: Dict[Action, Tuple[float, NodeQ]] = dict()
        self.bellman_backup_val: Optional[float] = None

    def backup(self) -> float:
        assert self.is_solved is not None
        if self.is_solved:
            self.bellman_backup_val = 0.0
        else:
            if len(self.act_dict) == 0:
                self.bellman_backup_val = self.heuristic
            else:
                self.bellman_backup_val = np.inf
                for tc, node_next in self.act_dict.values():
                    self.bellman_backup_val = min(self.bellman_backup_val, tc + node_next.heuristic)
        return self.bellman_backup_val

    def backup_act(self, action: Action) -> float:
        assert self.is_solved is not None
        if self.is_solved:
            return 0.0
        else:
            tc, node_next = self.act_dict[action]
            return tc + min(node_next.q_values)


class NodeQAct:
    __slots__ = ['node', 'action', 'q_val']

    def __init__(self, node: NodeQ, action: Optional[Action], q_val: float):
        self.node: NodeQ = node
        self.action: Optional[Action] = action
        self.q_val: float = q_val


class PathFindQ(PathFind[E, NodeQ, I]):
    def __init__(self, env: E, heur_fn: HeurFnQ):
        super().__init__(env)
        self.heur_fn: HeurFnQ = heur_fn

    @abstractmethod
    def step(self, verbose: bool = False) -> List[NodeQAct]:
        pass

    def get_next_nodes(self, instances: List[I], node_acts_by_inst: List[List[NodeQAct]]) -> List[List[NodeQ]]:
        if len(instances) == 0:
            return []
        start_time = time.time()
        # flatten
        node_acts, split_idxs = misc_utils.flatten(node_acts_by_inst)
        for node_act in node_acts:
            assert node_act.action is not None
        nodes: List[NodeQ] = [node_act.node for node_act in node_acts]
        actions: List[Action] = [node_act.action for node_act in node_acts]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        path_costs: List[float] = [popped_node.path_cost for popped_node in nodes]

        # next states
        states_next, tcs = self.env.next_state(states, actions)
        path_costs_next: List[float] = (np.array(path_costs) + np.array(tcs)).tolist()
        self.times.record_time("next_state", time.time() - start_time)

        # is solved
        start_time = time.time()
        is_solved_next: List[bool] = self.env.is_solved(states_next, goals)
        self.times.record_time("is_solved", time.time() - start_time)

        # heuristic function
        start_time = time.time()
        q_vals_next, actions_next_l = self.get_qvals_acts(states_next, goals)
        heurs_next: List[float] = [min(x) for x in q_vals_next]
        self.times.record_time("heur", time.time() - start_time)

        # next nodes
        start_time = time.time()
        nodes_next: List[NodeQ] = []
        for idx in range(len(node_acts)):
            node_next: NodeQ = NodeQ(states_next[idx], goals[idx], path_costs_next[idx], heurs_next[idx],
                                     None, actions[idx], tcs[idx], nodes[idx], actions_next_l[idx],
                                     q_vals_next[idx])
            nodes[idx].act_dict[actions[idx]] = (tcs[idx], node_next)
            nodes_next.append(node_next)
        self.times.record_time("make_nodes", time.time() - start_time)

        # updater instances
        start_time = time.time()
        nodes_next_by_inst: List[List[NodeQ]] = misc_utils.unflatten(nodes_next, split_idxs)
        for instance, nodes_next_by_inst_i in zip(instances, nodes_next_by_inst, strict=True):
            instance.num_nodes_generated += len(nodes_next_by_inst_i)
        self.times.record_time("up_inst", time.time() - start_time)

        return nodes_next_by_inst

    @abstractmethod
    def get_qvals_acts(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[float]], List[List[Action]]]:
        pass

    def create_root_nodes(self, states: List[State], goals: List[Goal], compute_init_heur: bool = True) -> List[NodeQ]:
        start_time = time.time()
        qvals_l, actions_l = self.get_qvals_acts(states, goals)

        heuristics: List[float]
        if compute_init_heur:
            heuristics = [min(x) for x in qvals_l]
        else:
            heuristics = [0.0 for _ in states]

        root_nodes: List[NodeQ] = []
        is_solved_l: List[bool] = self.env.is_solved(states, goals)
        for state, goal, heuristic, actions, is_solved, qvals in zip(states, goals, heuristics, actions_l, is_solved_l, qvals_l, strict=True):
            root_node: NodeQ = NodeQ(state, goal, 0.0, heuristic, None, None, None, None, actions, qvals)
            root_nodes.append(root_node)
        self.times.record_time("root", time.time() - start_time)

        return root_nodes


class PathFindQExpandEnum(PathFindQ[EnvEnumerableActs, I], ABC):
    def get_qvals_acts(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[float]], List[List[Action]]]:
        actions_l: List[List[Action]] = self.env.get_state_actions(states)
        qvals_l: List[List[float]] = self.heur_fn(states, goals, actions_l)
        return qvals_l, actions_l
