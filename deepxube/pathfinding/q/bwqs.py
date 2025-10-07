from typing import List, Tuple, Dict, Optional, Any
from deepxube.base.pathfinding import Instance, NodeQ, PathFindQ, InstArgs, NodeQAct
from deepxube.base.env import Env, State, Goal, Action
from deepxube.base.heuristic import HeurFnQ
from heapq import heappush, heappop
import time


OpenSetElem = Tuple[float, int, NodeQ, List[Optional[Action]], List[float]]


class InstArgsBWQS(InstArgs):
    def __init__(self, batch_size: int = 1, weight: float = 1.0, eps: float = 0.0):
        super().__init__()
        self.batch_size: int = batch_size
        self.weight: float = weight
        self.eps: float = eps


class InstanceBWQS(Instance[NodeQ, InstArgsBWQS]):
    def __init__(self, root_node: NodeQ, inst_args: InstArgsBWQS, inst_info: Any):
        super().__init__(root_node, inst_args, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.effective_open_size: int = 0
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.finished: bool = False

        self.push_to_open([self.root_node], [[None]], [[0.0]])

    def push_to_open(self, nodes: List[NodeQ], actions_sort_l: List[List[Optional[Action]]],
                     costs_sort_l: List[List[float]]):
        for node, actions_sort, costs_sort in zip(nodes, actions_sort_l, costs_sort_l):
            heappush(self.open_set, (costs_sort[0], self.heappush_count, node, actions_sort, costs_sort))
            self.heappush_count += 1
            self.effective_open_size += len(actions_sort)

    def pop_from_open(self) -> List[NodeQAct]:
        num_to_pop: int = min(self.inst_args.batch_size, self.effective_open_size)
        popped_nodeacts: List[NodeQAct] = []

        for _ in range(num_to_pop):
            _, _, node, actions_sort, costs_sort = heappop(self.open_set)
            popped_nodeacts.append(NodeQAct(node, actions_sort.pop(0)))
            costs_sort.pop(0)

            if len(actions_sort) > 0:
                self.push_to_open([node], [actions_sort], [costs_sort])

        self.effective_open_size -= num_to_pop

        return popped_nodeacts


class BWQS(PathFindQ[InstanceBWQS, InstArgsBWQS]):
    def step(self, heur_fn: HeurFnQ) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        instances: List[InstanceBWQS] = [instance for instance in self.instances if not instance.finished]

        # Pop from open
        start_time = time.time()
        nodeacts_by_inst_popped: List[List[NodeQAct]] = [instance.pop_from_open() for instance in instances]
        self.times.record_time("pop", time.time() - start_time)

        # Next state


    def remove_finished_instances(self, itr_max: int) -> List[InstanceBWQS]:
        def remove_instance_fn(inst_in: InstanceBWQS) -> bool:
            if inst_in.finished:
                return True
            if inst_in.itr >= itr_max:
                return True
            return False

        return self.remove_instances(remove_instance_fn)

    def _get_instance(self, root_node: NodeQ, inst_args: InstArgsBWQS, inst_info: Any) -> InstanceBWQS:
        return InstanceBWQS(root_node, inst_args, inst_info)
