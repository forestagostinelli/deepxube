from abc import ABC
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Type
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Goal
from deepxube.base.pathfinding import InstanceV, Node, PathFindVHeur, PathFindVExpandEnum
from deepxube.factories.pathfinding_factory import pathfinding_factory
import numpy as np
from heapq import heappush, heappop, heapify
import random

from deepxube.utils import misc_utils
import time


OpenSetElem = Tuple[float, int, Node]


class InstanceBWAS(InstanceV):
    def __init__(self, root_node: Node, batch_size: int, weight: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = {self.root_node.state: 0.0}
        self.ub: float = np.inf
        self.lb: float = 0.0
        self.batch_size: int = batch_size
        self.weight: float = weight
        self.eps: float = eps

        self.push_to_open([self.root_node], [self.root_node.heuristic])

    def push_to_open(self, nodes: List[Node], costs: List[float]) -> None:
        for node, cost in zip(nodes, costs, strict=True):
            heappush(self.open_set, (cost, self.heappush_count, node))
            self.heappush_count += 1

    def check_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_ret: List[Node] = []
        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                nodes_ret.append(node)
        return nodes_ret

    def pop_from_open(self) -> List[Node]:
        num_to_pop: int = min(self.batch_size, len(self.open_set))

        elems_popped: List[OpenSetElem] = []
        for _ in range(num_to_pop):
            if random.random() < self.eps:
                pop_idx: int = random.randrange(0, len(self.open_set))
                elems_popped.append(self.open_set.pop(pop_idx))
                heapify(self.open_set)
            else:
                elems_popped.append(heappop(self.open_set))

        nodes_popped: List[Node] = [elem_popped[2] for elem_popped in elems_popped]

        assert len(elems_popped) > 0

        cost_first: float = elems_popped[0][0]
        self.lb = max(cost_first, self.lb)

        return nodes_popped

    def update_ub(self, nodes: List[Node]) -> None:
        # keep solved nodes for training
        for node in nodes:
            assert node.is_solved is not None
            if node.is_solved and (self.ub > node.path_cost):
                self.goal_node = node
                self.ub = node.path_cost

    def finished(self) -> bool:
        case1: bool = (self.goal_node is not None) and (self.lb >= (self.weight * self.ub))
        case2: bool = len(self.open_set) == 0
        return case1 or case2


D = TypeVar('D', bound=Domain)


class BWASActsAny(PathFindVHeur[D, InstanceBWAS], ABC):
    def __init__(self, domain: D, batch_size: int = 1, weight: float = 1.0, eps: float = 0.0):
        super().__init__(domain)
        self.batch_size_default: int = batch_size
        self.weight_default: float = weight
        self.eps_default: float = eps

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       batch_size: Optional[int] = None, weight: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceBWAS]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals, compute_root_heur=compute_root_heur)
        batch_size_inst: int = batch_size if batch_size is not None else self.batch_size_default
        weight_inst: float = weight if weight is not None else self.weight_default
        eps_inst: float = eps if eps is not None else self.eps_default
        if inst_infos is None:
            inst_infos = [None for _ in states]
        return [InstanceBWAS(node_root, batch_size_inst, weight_inst, eps_inst, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]

    def step(self, verbose: bool = False) -> List[Node]:
        instances: List[InstanceBWAS] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # pop from open
        start_time = time.time()
        nodes_popped_by_inst: List[List[Node]] = [instance.pop_from_open() for instance in instances]
        self.times.record_time("pop", time.time() - start_time)

        # is solved
        start_time = time.time()
        nodes_popped_flat: List[Node] = misc_utils.flatten(nodes_popped_by_inst)[0]
        self.set_is_solved(nodes_popped_flat)
        self.times.record_time("is_solved", time.time() - start_time)

        # ub
        start_time = time.time()
        for instance, nodes_popped in zip(instances, nodes_popped_by_inst, strict=True):
            instance.update_ub(nodes_popped)
        self.times.record_time("ub", time.time() - start_time)

        # expand nodes
        nodes_c_by_inst: List[List[Node]] = self._expand_nodes(instances, nodes_popped_by_inst)

        # check closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_c_by_inst[inst_idx] = instance.check_closed(nodes_c_by_inst[inst_idx])
        self.times.record_time("check", time.time() - start_time)

        # cost
        start_time = time.time()
        nodes_c_flat: List[Node] = misc_utils.flatten(nodes_c_by_inst)[0]
        weights, split_idxs = misc_utils.flatten([[instance.weight] * len(nodes_c)
                                                  for instance, nodes_c in
                                                  zip(instances, nodes_c_by_inst, strict=True)])
        path_costs: List[float] = [node.path_cost for node in nodes_c_flat]
        heuristics: List[float] = [node.heuristic for node in nodes_c_flat]
        costs_flat: List[float] = ((np.array(weights) * np.array(path_costs)) + np.array(heuristics)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)
        self.times.record_time("cost", time.time() - start_time)

        # push to open
        start_time = time.time()
        for instance, nodes_c, costs in zip(instances, nodes_c_by_inst, costs_by_inst, strict=True):
            instance.push_to_open(nodes_c, costs)
        self.times.record_time("push", time.time() - start_time)

        # verbose
        if verbose:
            if len(heuristics) > 0:
                min_heur = float(np.min(heuristics))
                min_heur_pc = float(path_costs[np.argmin(heuristics)])
                max_heur = float(np.max(heuristics))
                max_heur_pc = float(path_costs[np.argmax(heuristics)])
                per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in self.instances]))
                per_finished: float = 100.0 * float(np.mean([inst.finished() for inst in self.instances]))

                print(f"Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      f"%.2E(%.2f)/%.2E(%.2f), %%has_soln: {per_has_soln}, "
                      f"%%finished: {per_finished}" % (self.itr, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print(f"Times - {self.times.get_time_str()}")
            print("")

        # update iterations
        self.itr += 1
        for instance in instances:
            instance.itr += 1

        # return
        return nodes_popped_flat

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size_default}, weight={self.weight_default}, eps={self.eps_default})"


@pathfinding_factory.register_class("bwas")
class BWAS(BWASActsAny[ActsEnum], PathFindVExpandEnum[InstanceBWAS]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_parser("bwas")
class BWASParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        assert len(args_str_l) == 3
        return {"batch_size": int(args_str_l[0]), "weight": float(args_str_l[1]), "eps": float(args_str_l[2])}

    def help(self) -> str:
        return "The batch size, weight, and random node expansion probability (eps). E.g. 'bwas.1_0.9_0.1'"
