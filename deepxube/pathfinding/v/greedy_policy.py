from typing import List, Any
from deepxube.base.pathfinding import Instance, NodeV, PathFindVExpandEnum
import numpy as np
import random
import time


class InstanceGrPolV(Instance[NodeV]):
    def __init__(self, root_node: NodeV, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_node: NodeV = self.root_node
        self.eps: float = eps

    def record_goal(self) -> None:
        assert self.curr_node.is_solved is not None
        if self.curr_node.is_solved:
            if (self.goal_node is None) or (self.goal_node.path_cost > self.curr_node.path_cost):
                self.goal_node = self.curr_node

    def finished(self) -> bool:
        return self.has_soln()


class GreedyPolicyVEnum(PathFindVExpandEnum[InstanceGrPolV]):
    def step(self, verbose: bool = False) -> List[NodeV]:
        # get unsolved instances
        instances: List[InstanceGrPolV] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get curr nodes
        nodes_curr: List[NodeV] = [inst.curr_node for inst in instances]

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes_curr)
        for instance in instances:
            instance.record_goal()
        self.times.record_time("is_solved", time.time() - start_time)

        # expand
        self.expand_nodes(instances, [[node_curr] for node_curr in nodes_curr])

        # take action
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx, instance in enumerate(instances):
            # get next state
            curr_node: NodeV = instance.curr_node
            assert curr_node.children is not None
            assert curr_node.t_costs is not None
            t_costs: List[float] = curr_node.t_costs
            children: List[NodeV] = curr_node.children
            qvals: List[float] = [t_cost + child.heuristic for t_cost, child in zip(t_costs, children)]

            next_idx: int
            if rand_vals[idx] < instance.eps:
                next_idx = random.choice(list(range(len(qvals))))
            else:
                next_idx = int(np.argmin(qvals))
            node_next: NodeV = children[next_idx]

            instance.curr_node = node_next
            instance.itr += 1
        self.times.record_time("get_next", time.time() - start_time)

        return nodes_curr
