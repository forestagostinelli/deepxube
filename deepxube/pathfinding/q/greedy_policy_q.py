from typing import List, Any
from deepxube.base.pathfinding import Instance, NodeQ, PathFindVExpandEnum
import numpy as np
import random
import time


class InstanceGrV(Instance[NodeQ]):
    def __init__(self, root_node: NodeQ, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_node: NodeQ = self.root_node
        self.eps: float = eps

    def finished(self) -> bool:
        return self.has_soln()


class Greedy(PathFindVExpandEnum[InstanceGrV]):
    def step(self) -> List[NodeV]:
        # get unsolved instances
        instances: List[InstanceGrV] = self._get_unsolved_instances()
        if len(instances) == 0:
            return []

        nodes_curr: List[NodeV] = [inst.curr_node for inst in instances]
        self.expand_nodes(instances, [[node_curr] for node_curr in nodes_curr])

        # take action
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx, instance in enumerate(instances):
            # check solved
            if instance.curr_node.is_solved:
                instance.goal_node = instance.curr_node
            else:
                # get next state
                curr_node: NodeV = instance.curr_node
                assert curr_node.children is not None
                assert curr_node.t_costs is not None
                t_costs: List[float] = curr_node.t_costs
                children: List[NodeV] = curr_node.children
                tc_p_ctg_next: List[float] = [t_cost + child.heuristic for t_cost, child in zip(t_costs, children)]

                child_idx: int = int(np.argmin(tc_p_ctg_next))
                if rand_vals[idx] < instance.eps:
                    child_idx = random.choice(list(range(len(tc_p_ctg_next))))
                node_next: NodeV = children[child_idx]

                instance.curr_node = node_next
            instance.itr += 1
        self.times.record_time("get_next", time.time() - start_time)

        return nodes_curr

    def _get_unsolved_instances(self) -> List[InstanceGrV]:
        instances_unsolved: List[InstanceGrV] = [instance for instance in self.instances if not instance.has_soln()]
        return instances_unsolved
