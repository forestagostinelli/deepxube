from typing import List, Optional, Any, Tuple, Callable
from deepxube.base.env import State, Goal
from deepxube.base.pathfinding import Instance, NodeV, PathFindV, InstArgs
import numpy as np
import random
import time


class InstArgsGr(InstArgs):
    def __init__(self, eps: float):
        super().__init__()
        self.eps: float = eps


class InstanceGrV(Instance[NodeV, InstArgsGr]):
    def __init__(self, root_node: NodeV, inst_args: InstArgsGr, inst_info: Any):
        super().__init__(root_node, inst_args, inst_info)
        self.curr_node: NodeV = self.root_node


class Greedy(PathFindV[InstanceGrV, InstArgsGr]):
    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: Callable, inst_args_l: List[InstArgsGr],
                      inst_infos: Optional[List[Any]] = None, compute_init_heur: bool = True):
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None] * len(states)

        assert len(states) == len(goals) == len(inst_infos) == len(inst_args_l), "Number should be the same"

        root_nodes: List[NodeV] = self._create_root_nodes(states, goals, heur_fn, compute_init_heur)

        for root_node, inst_args, inst_info in zip(root_nodes, inst_args_l, inst_infos):
            instance: InstanceGrV = InstanceGrV(root_node, inst_args, inst_info)
            self.instances.append(instance)
        self.times.record_time("add", time.time() - start_time)

    def step(self, heur_fn: Callable) -> Tuple[List[State], List[Goal], List[float]]:
        # get unsolved instances
        instances: List[InstanceGrV] = self._get_unsolved_instances()
        if len(instances) == 0:
            return [], [], []

        self.expand_nodes(instances, [[inst.curr_node] for inst in instances], heur_fn)
        start_time = time.time()
        states: List[State] = [inst.curr_node.state for inst in instances]
        goals: List[Goal] = [inst.curr_node.goal for inst in instances]
        ctgs: List[float] = [inst.curr_node.backup() for inst in instances]
        self.times.record_time("bellman", time.time() - start_time)

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
                if rand_vals[idx] < instance.inst_args.eps:
                    child_idx = random.choice(list(range(len(tc_p_ctg_next))))
                node_next: NodeV = children[child_idx]

                instance.curr_node = node_next
            instance.itr += 1
        self.times.record_time("get_next", time.time() - start_time)

        return states, goals, ctgs

    def remove_finished_instances(self, itr_max: int) -> List[InstanceGrV]:
        def remove_instance_fn(inst_in: InstanceGrV) -> bool:
            if inst_in.has_soln():
                return True
            if inst_in.itr >= itr_max:
                return True
            return False

        return self.remove_instances(remove_instance_fn)

    def _get_unsolved_instances(self) -> List[InstanceGrV]:
        instances_unsolved: List[InstanceGrV] = [instance for instance in self.instances if not instance.has_soln()]
        return instances_unsolved
