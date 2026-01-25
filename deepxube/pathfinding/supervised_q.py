from abc import ABC
from typing import List, Any, Optional, Type, TypeVar
from deepxube.base.domain import Domain, State, Goal, StartGoalWalkable, GoalStartRevWalkableActsRev, Action
from deepxube.base.pathfinding import Instance, Node, EdgeQ, PathFindQ, PathFindSup
from deepxube.factories.pathfinding_factory import pathfinding_factory
import numpy as np
import time


class InstanceSupQ(Instance):
    def __init__(self, root_node: Node, action: Action, path_cost_sup: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.action: Action = action
        self.path_cost_sup: float = path_cost_sup

    def finished(self) -> bool:
        return self.itr > 0


D = TypeVar('D', bound=Domain)


class PathFindQSup(PathFindQ[D, InstanceSupQ], PathFindSup[D, InstanceSupQ], ABC):
    def step(self, verbose: bool = False) -> List[EdgeQ]:
        edges: List[EdgeQ] = []
        for instance in self.instances:
            node_root: Node = instance.root_node
            edges.append(EdgeQ(node_root, instance.action, instance.path_cost_sup))
            node_root.backup_val = instance.path_cost_sup
            instance.itr += 1
        start_time = time.time()
        self.set_is_solved([edge.node for edge in edges])
        self.times.record_time("is_solved", time.time() - start_time)

        return edges

    def _get_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        raise NotImplementedError

    def _get_heur_vals(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        raise NotImplementedError

    def _make_instances(self, states_start: List[State], goals: List[Goal], acts_init: List[Action], path_costs: List[float],
                        inst_infos: Optional[List[Any]]) -> List[InstanceSupQ]:
        # make root nodes
        start_time = time.time()
        nodes_root: List[Node] = []
        for state_start, goal in zip(states_start, goals, strict=True):
            node_root: Node = Node(state_start, goal, 0.0, 0.0, None, None, None, None, None)
            nodes_root.append(node_root)
        self.times.record_time("root", time.time() - start_time)

        # make instances
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None for _ in states_start]

        instances: List[InstanceSupQ] = []
        for node_root, act_init, path_cost, inst_info in zip(nodes_root, acts_init, path_costs, inst_infos):
            instances.append(InstanceSupQ(node_root, act_init, path_cost, inst_info))
        self.times.record_time("instances", time.time() - start_time)

        return instances


@pathfinding_factory.register_class("sup_q_rw")
class PathFindQSupRW(PathFindQSup[StartGoalWalkable]):
    @staticmethod
    def domain_type() -> Type[StartGoalWalkable]:
        return StartGoalWalkable

    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceSupQ]:
        # start states
        start_time = time.time()
        states_start: List[State] = self.domain.sample_start_states(len(steps_gen))
        self.times.record_time("get_start_states", time.time() - start_time)

        # first step
        start_time = time.time()
        acts_init: List[Action] = self.domain.sample_state_action(states_start)
        states_start_1step, path_costs_1step = self.domain.next_state(states_start, acts_init)
        for idx in np.where(np.array(steps_gen) == 0)[0]:
            states_start_1step[idx] = states_start[idx]
            path_costs_1step[idx] = 0.0
        self.times.record_time("first_step", time.time() - start_time)

        # random walk
        start_time = time.time()
        steps_gen_min_1: List[int] = np.maximum(np.array(steps_gen) - 1, 0).tolist()
        states_goal, path_costs = self.domain.random_walk(states_start_1step, steps_gen_min_1)
        path_costs = (np.array(path_costs_1step) + np.array(path_costs)).tolist()
        self.times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[Goal] = self.domain.sample_goal_from_state(states_start, states_goal)
        self.times.record_time("sample_goal", time.time() - start_time)

        return self._make_instances(states_start, goals, acts_init, path_costs, inst_infos)


@pathfinding_factory.register_class("sup_q_rw_rev")
class PathFindQSupRWRev(PathFindQSup[GoalStartRevWalkableActsRev]):
    @staticmethod
    def domain_type() -> Type[GoalStartRevWalkableActsRev]:
        return GoalStartRevWalkableActsRev

    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceSupQ]:
        start_time = time.time()
        states_goal, goals = self.domain.sample_goal_state_goal_pairs(len(steps_gen))
        self.times.record_time("samp_goal_state_goal", time.time() - start_time)

        start_time = time.time()
        steps_gen_min_1: List[int] = np.maximum(np.array(steps_gen) - 1, 0).tolist()
        states_start_1step, path_costs = self.domain.random_walk(states_goal, steps_gen_min_1)
        self.times.record_time("random_walk", time.time() - start_time)

        start_time = time.time()
        acts_init_rev: List[Action] = self.domain.sample_state_action(states_start_1step)
        states_start, path_costs_1step = self.domain.next_state(states_start_1step, acts_init_rev)
        acts_init: List[Action] = self.domain.rev_action(states_start, acts_init_rev)
        for idx in np.where(np.array(steps_gen) == 0)[0]:
            states_start[idx] = states_start_1step[idx]
            path_costs_1step[idx] = 0.0

        path_costs = (np.array(path_costs_1step) + np.array(path_costs)).tolist()
        self.times.record_time("first_step", time.time() - start_time)

        return self._make_instances(states_start, goals, acts_init, path_costs, inst_infos)
