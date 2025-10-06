from typing import List, Tuple, Dict, Optional, Any
from deepxube.base.pathfinding import Instance, NodeQ, PathFindQ, InstArgs, NodeQAct
from deepxube.base.environment import Environment, State, Goal, Action
from deepxube.base.heuristic import HeurFnQ
from heapq import heappush, heappop


OpenSetElem = Tuple[float, int, NodeQ, List[Action], List[float]]


class InstanceBWQS(Instance):
    def __init__(self, root_node: NodeQ, inst_info: Any, weight: float, eps: float):
        super().__init__(root_node, inst_info)
        self.root_node: NodeQ = root_node
        self.open_set: List[OpenSetElem] = []
        self.effective_open_size: int = 0
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.finished: bool = False
        self.weight: float = weight
        self.eps: float = eps

    def push_to_open(self, nodes: List[NodeQ], actions_sort_l: List[List[Action]], costs_sort_l: List[List[float]]):
        for node, actions_sort, costs_sort in zip(nodes, actions_sort_l, costs_sort_l):
            heappush(self.open_set, (costs_sort[0], self.heappush_count, node, actions_sort, costs_sort))
            self.heappush_count += 1
            self.effective_open_size += len(actions_sort)

    def pop_from_open(self, num_nodes: int) -> List[NodeQAct]:
        num_to_pop: int = min(num_nodes, self.effective_open_size)
        popped_nodeacts: List[NodeQAct] = []

        for _ in range(num_to_pop):
            _, _, node, actions_sort, costs_sort = heappop(self.open_set)
            popped_nodeacts.append(NodeQAct(node, actions_sort.pop(0)))
            costs_sort.pop(0)

            if len(actions_sort) > 0:
                self.push_to_open([node], [actions_sort], [costs_sort])

        self.effective_open_size -= num_to_pop

        return popped_nodeacts


class InstArgsBWQS(InstArgs):
    def __init__(self, weight: float, eps: float):
        super().__init__()
        self.weight: float = weight
        self.eps: float = eps


class BWQS(PathFindQ[InstanceBWQS, InstArgsBWQS]):
    def step(self, heur_fn: HeurFnQ) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        pass
