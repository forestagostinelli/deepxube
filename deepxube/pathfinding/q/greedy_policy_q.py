from typing import List, Any
from abc import ABC
from deepxube.base.env import EnvEnumerableActs, State, Goal
from deepxube.base.pathfinding import E, Instance, NodeQ, PathFindQ, NodeQAct, PathFindQExpandEnum
from deepxube.utils.misc_utils import boltzmann
import numpy as np
import random
import time


class InstanceGrPolQ(Instance[NodeQ]):
    def __init__(self, root_node: NodeQ, temp: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_node: NodeQ = self.root_node
        self.temp: float = temp
        assert (eps >= 0.0) and (eps <= 1.0)
        self.eps: float = eps

    def record_goal(self) -> None:
        assert self.curr_node.is_solved is not None
        if self.curr_node.is_solved:
            if (self.goal_node is None) or (self.goal_node.path_cost > self.curr_node.path_cost):
                self.goal_node = self.curr_node

    def finished(self) -> bool:
        return self.has_soln()


class GreedyPolicyQ(PathFindQ[E, InstanceGrPolQ], ABC):
    def step(self, verbose: bool = False) -> List[NodeQAct]:
        # get unsolved instances
        instances: List[InstanceGrPolQ] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get curr nodes
        nodes_curr: List[NodeQ] = [inst.curr_node for inst in instances]

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes_curr)
        for instance in instances:
            instance.record_goal()
        self.times.record_time("is_solved", time.time() - start_time)

        # get qvals and actions
        start_time = time.time()
        states: List[State] = [node.state for node in nodes_curr]
        goals: List[Goal] = [node.goal for node in nodes_curr]
        q_vals_l, actions_l = self.get_qvals_acts(states, goals)
        self.times.record_time("heur", time.time() - start_time)

        # sample next actions
        start_time = time.time()
        nodeq_acts: List[NodeQAct] = []
        for q_vals, actions, node, instance in zip(q_vals_l, actions_l, nodes_curr, instances, strict=True):
            next_idx: int
            if random.random() < instance.eps:
                next_idx = random.randrange(0, len(actions))
            else:
                probs: List[float] = boltzmann((-np.array(q_vals)).tolist(), instance.temp)
                next_idx = int(np.random.multinomial(1, np.array(probs)).argmax())
            nodeq_acts.append(NodeQAct(node, actions[next_idx], q_vals[next_idx]))

        self.times.record_time("samp_acts", time.time() - start_time)

        # take actions
        nodes_next_l: List[List[NodeQ]] = self.get_next_nodes(instances, [[nodeq_act] for nodeq_act in nodeq_acts])
        for nodes_next, instance in zip(nodes_next_l, instances, strict=True):
            assert len(nodes_next) == 1
            instance.curr_node = nodes_next[0]

        self.itr += 1
        for instance in instances:
            instance.itr += 1

        return nodeq_acts


class GreedyPolicyQEnum(GreedyPolicyQ[EnvEnumerableActs], PathFindQExpandEnum[InstanceGrPolQ]):
    pass
