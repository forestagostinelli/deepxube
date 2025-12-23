from typing import List, Any
from abc import ABC
from deepxube.base.domain import ActsEnum, Action
from deepxube.base.pathfinding import E, Instance, NodeQ, PathFindQ, Edge, PathFindQExpandEnum
from deepxube.utils.misc_utils import boltzmann
import numpy as np
import random
import time


class InstanceGrPolQ(Instance[NodeQ]):
    def __init__(self, root_node: NodeQ, temp: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_edge: Edge = Edge(self.root_node, None, self.root_node.heuristic)
        self.temp: float = temp
        assert (eps >= 0.0) and (eps <= 1.0)
        self.eps: float = eps

    def record_goal(self, node: NodeQ) -> None:
        assert node.is_solved is not None
        if node.is_solved:
            if (self.goal_node is None) or (self.goal_node.path_cost > node.path_cost):
                self.goal_node = node

    def finished(self) -> bool:
        return self.has_soln()


class GreedyPolicyQ(PathFindQ[E, InstanceGrPolQ], ABC):
    def step(self, verbose: bool = False) -> List[Edge]:
        # get unsolved instances
        instances: List[InstanceGrPolQ] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get curr edges
        edges: List[Edge] = [inst.curr_edge for inst in instances]

        # next node
        nodes_next_l: List[List[NodeQ]] = self.get_next_nodes(instances, [[edge] for edge in edges])
        nodes_next: List[NodeQ] = [nodes_next_i[0] for nodes_next_i in nodes_next_l]

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes_next)
        for instance, node_next in zip(instances, nodes_next):
            instance.record_goal(node_next)
        self.times.record_time("is_solved", time.time() - start_time)

        # sample next actions
        start_time = time.time()
        edges_next: List[Edge] = []
        for node_next, instance in zip(nodes_next, instances, strict=True):
            actions: List[Action] = node_next.actions
            assert node_next.q_values is not None
            q_vals: List[float] = node_next.q_values

            next_idx: int
            if random.random() < instance.eps:
                next_idx = random.randrange(0, len(actions))
            elif instance.temp > 0:
                probs: List[float] = boltzmann((-np.array(q_vals)).tolist(), instance.temp)
                next_idx = int(np.random.multinomial(1, np.array(probs)).argmax())
            else:
                next_idx = int(np.argmin(q_vals))
            edges_next.append(Edge(node_next, actions[next_idx], q_vals[next_idx]))

        self.times.record_time("samp_acts", time.time() - start_time)

        # take actions
        for edge_next, instance in zip(edges_next, instances, strict=True):
            instance.curr_edge = edge_next
            instance.itr += 1

        self.itr += 1

        return edges


class GreedyPolicyQEnum(GreedyPolicyQ[ActsEnum], PathFindQExpandEnum[InstanceGrPolQ]):
    pass
