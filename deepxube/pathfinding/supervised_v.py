from abc import ABC
from typing import List, Any, Optional, Type, TypeVar, Tuple
from deepxube.base.domain import Domain, State, Goal, StartGoalWalkable, GoalStartRevWalkableActsRev, Action
from deepxube.base.pathfinding import InstanceV, Node, PathFindV, PathFindSup
from deepxube.factories.pathfinding_factory import pathfinding_factory
import time


class InstanceSupV(InstanceV):
    def __init__(self, root_node: Node, path_cost_sup: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.path_cost_sup: float = path_cost_sup

    def finished(self) -> bool:
        return self.itr > 0


D = TypeVar('D', bound=Domain)


class PathFindVSup(PathFindV[D, InstanceSupV], PathFindSup[D, InstanceSupV], ABC):
    def step(self, verbose: bool = False) -> List[Node]:
        nodes: List[Node] = []
        for instance in self.instances:
            node_root: Node = instance.root_node
            node_root.heuristic = instance.path_cost_sup
            node_root.backup_val = instance.path_cost_sup
            nodes.append(node_root)
            instance.nodes_popped.append(node_root)
            instance.itr += 1
        start_time = time.time()
        self.set_is_solved(nodes)
        self.times.record_time("is_solved", time.time() - start_time)

        return nodes

    def _expand(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        raise NotImplementedError

    def _get_heur_vals(self, states: List[State], goals: List[Goal]) -> List[float]:
        raise NotImplementedError

    def _make_instances(self, states_start: List[State], goals: List[Goal], path_costs: List[float], inst_infos: Optional[List[Any]]) -> List[InstanceSupV]:
        nodes_root: List[Node] = self._create_root_nodes(states_start, goals, compute_root_heur=False)

        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None for _ in states_start]

        instances: List[InstanceSupV] = []
        for node_root, path_cost, inst_info in zip(nodes_root, path_costs, inst_infos):
            instances.append(InstanceSupV(node_root, path_cost, inst_info))
        self.times.record_time("instances", time.time() - start_time)

        return instances


@pathfinding_factory.register_class("sup_v_rw")
class PathFindVSupRW(PathFindVSup[StartGoalWalkable]):
    @staticmethod
    def domain_type() -> Type[StartGoalWalkable]:
        return StartGoalWalkable

    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceSupV]:
        start_time = time.time()
        states_start: List[State] = self.domain.sample_start_states(len(steps_gen))
        self.times.record_time("get_start_states", time.time() - start_time)

        start_time = time.time()
        states_goal, path_costs = self.domain.random_walk(states_start, steps_gen)
        self.times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[Goal] = self.domain.sample_goal_from_state(states_start, states_goal)
        self.times.record_time("sample_goal", time.time() - start_time)

        return self._make_instances(states_start, goals, path_costs, inst_infos)


@pathfinding_factory.register_class("sup_v_rw_rev")
class PathFindVSupRWRev(PathFindVSup[GoalStartRevWalkableActsRev]):
    @staticmethod
    def domain_type() -> Type[GoalStartRevWalkableActsRev]:
        return GoalStartRevWalkableActsRev

    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceSupV]:
        start_time = time.time()
        states_goal, goals = self.domain.sample_goal_state_goal_pairs(len(steps_gen))
        self.times.record_time("samp_goal_state_goal", time.time() - start_time)

        start_time = time.time()
        states_start, path_costs = self.domain.random_walk(states_goal, steps_gen)
        self.times.record_time("random_walk", time.time() - start_time)

        return self._make_instances(states_start, goals, path_costs, inst_infos)
