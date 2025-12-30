import random
from abc import ABC
from typing import List, Tuple, Dict, Optional, Any, TypeVar

from deepxube.base.domain import Domain, ActsEnum, State
from deepxube.base.pathfinding import Instance, NodeQ, PathFindQ, Edge, PathFindQExpandEnum
from deepxube.utils import misc_utils
from heapq import heappush, heappop, heapify
import numpy as np
import time


OpenSetElem = Tuple[float, int, Edge]


class InstanceBWQS(Instance[NodeQ]):
    def __init__(self, root_node: NodeQ, batch_size: int, weight: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = {}
        self.ub: float = np.inf
        self.lb: float = 0.0
        self.batch_size: int = batch_size
        self.weight: float = weight
        self.eps: float = eps

        self.push_to_open([Edge(self.root_node, None, self.root_node.heuristic)], [self.root_node.heuristic])

    def push_to_open(self, edges: List[Edge], costs: List[float]) -> None:
        for edge, cost in zip(edges, costs, strict=True):
            heappush(self.open_set, (cost, self.heappush_count, edge))
            self.heappush_count += 1

    def check_closed(self, nodes: List[NodeQ]) -> List[NodeQ]:
        nodes_ret: List[NodeQ] = []
        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                nodes_ret.append(node)
        return nodes_ret

    def pop_from_open(self) -> List[Edge]:
        num_to_pop: int = min(self.batch_size, len(self.open_set))

        elems_popped: List[OpenSetElem] = []
        for _ in range(num_to_pop):
            if random.random() < self.eps:
                pop_idx: int = random.randrange(0, len(self.open_set))
                elems_popped.append(self.open_set.pop(pop_idx))
                heapify(self.open_set)
            else:
                elems_popped.append(heappop(self.open_set))
        edges_popped: List[Edge] = [elem_popped[2] for elem_popped in elems_popped]

        if len(elems_popped) > 0:
            cost_first: float = elems_popped[0][0]
            self.lb = max(cost_first, self.lb)

        return edges_popped

    def update_ub(self, nodes: List[NodeQ]) -> None:
        # keep solved nodes for training
        for node in nodes:
            if (node.is_solved is not None) and node.is_solved and (self.ub > node.path_cost):
                self.goal_node = node
                self.ub = node.path_cost

    def finished(self) -> bool:
        return (self.goal_node is not None) and (self.lb >= (self.weight * self.ub))


D = TypeVar('D', bound=Domain)


class BWQS(PathFindQ[D, InstanceBWQS], ABC):
    def step(self, verbose: bool = False) -> List[Edge]:
        # split instances by iteration
        instances: List[InstanceBWQS] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # pop from open
        start_time = time.time()
        edges_popped_by_inst: List[List[Edge]] = [instance.pop_from_open() for instance in instances]
        self.times.record_time("pop", time.time() - start_time)

        # next state
        nodes_next_by_inst: List[List[NodeQ]] = self.get_next_nodes(instances, edges_popped_by_inst)

        # is solved
        start_time = time.time()
        nodes_next_flat: List[NodeQ] = misc_utils.flatten(nodes_next_by_inst)[0]
        self.set_is_solved(nodes_next_flat)
        self.times.record_time("is_solved", time.time() - start_time)

        # ub
        start_time = time.time()
        for instance, nodes_next in zip(instances, nodes_next_by_inst, strict=True):
            instance.update_ub(nodes_next)
        self.times.record_time("ub", time.time() - start_time)

        # check closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_next_by_inst[inst_idx] = instance.check_closed(nodes_next_by_inst[inst_idx])
        self.times.record_time("check", time.time() - start_time)

        # make edges
        start_time = time.time()
        edges_next_by_inst: List[List[Edge]] = []

        for nodes_next in nodes_next_by_inst:
            edges_next: List[Edge] = []
            for node in nodes_next:
                assert node.q_values is not None
                for action, q_val in zip(node.actions, node.q_values, strict=True):
                    edges_next.append(Edge(node, action, q_val))
            edges_next_by_inst.append(edges_next)
        self.times.record_time("edges", time.time() - start_time)

        # costs
        start_time = time.time()
        edges_next_flat: List[Edge] = misc_utils.flatten(edges_next_by_inst)[0]
        weights, split_idxs = misc_utils.flatten([[instance.weight] * len(edges_next)
                                                  for instance, edges_next in
                                                  zip(instances, edges_next_by_inst, strict=True)])
        path_costs: List[float] = [edge.node.path_cost for edge in edges_next_flat]
        heuristics: List[float] = [edge.q_val for edge in edges_next_flat]
        costs_flat: List[float] = ((np.array(weights) * np.array(path_costs)) + np.array(heuristics)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)
        self.times.record_time("cost", time.time() - start_time)

        # push to open
        start_time = time.time()
        for instance, edges_next, costs in zip(instances, edges_next_by_inst, costs_by_inst, strict=True):
            instance.push_to_open(edges_next, costs)
        self.times.record_time("push", time.time() - start_time)

        # verbose
        if verbose:
            if len(heuristics) > 0:
                min_heur = float(np.min(heuristics))
                min_heur_pc = float(path_costs[np.argmin(heuristics)])
                max_heur = float(np.max(heuristics))
                max_heur_pc = float(path_costs[np.argmax(heuristics)])
                per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
                per_finished: float = 100.0 * float(np.mean([inst.finished for inst in instances]))

                print(f"Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      f"%.2f(%.2f)/%.2f(%.2f), %%has_soln: {per_has_soln}, "
                      f"%%finished: {per_finished}" % (self.itr, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print(f"Times - {self.times.get_time_str()}")
            print("")

        # update iterations
        self.itr += 1
        for instance in instances:
            instance.itr += 1

        # return
        edges_popped_flat: List[Edge] = misc_utils.flatten(edges_popped_by_inst)[0]
        # nodes_popped_flat: List[NodeQ] = [nodeact_popped.node for nodeact_popped in nodesacts_popped_flat]
        return edges_popped_flat


class BWQSEnum(BWQS[ActsEnum], PathFindQExpandEnum[InstanceBWQS]):
    pass
