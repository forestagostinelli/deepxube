import random
from abc import ABC
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Type

from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Goal
from deepxube.base.pathfinding import InstanceQ, Node, PathFindQHeur, EdgeQ, PathFindQExpandEnum
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.utils import misc_utils
from heapq import heappush, heappop, heapify
import numpy as np
import time


OpenSetElem = Tuple[float, int, EdgeQ]


class InstanceBWQS(InstanceQ):
    def __init__(self, root_node: Node, batch_size: int, weight: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = {}
        self.ub: float = np.inf
        self.lb: float = 0.0
        self.batch_size: int = batch_size
        self.weight: float = weight
        self.eps: float = eps

        self.popped_nodes_last: List[Node] = [self.root_node]

        # self.push_to_open([EdgeQ(self.root_node, None, self.root_node.heuristic)], [self.root_node.heuristic])

    def push_to_open(self, edges: List[EdgeQ], costs: List[float]) -> None:
        for edge, cost in zip(edges, costs, strict=True):
            heappush(self.open_set, (cost, self.heappush_count, edge))
            self.heappush_count += 1

    def check_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_ret: List[Node] = []
        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                nodes_ret.append(node)
        return nodes_ret

    def pop_from_open(self) -> List[EdgeQ]:
        num_to_pop: int = min(self.batch_size, len(self.open_set))

        elems_popped: List[OpenSetElem] = []
        for _ in range(num_to_pop):
            if random.random() < self.eps:
                pop_idx: int = random.randrange(0, len(self.open_set))
                elems_popped.append(self.open_set.pop(pop_idx))
                heapify(self.open_set)
            else:
                elems_popped.append(heappop(self.open_set))
        edges_popped: List[EdgeQ] = [elem_popped[2] for elem_popped in elems_popped]

        if len(elems_popped) > 0:
            cost_first: float = elems_popped[0][0]
            self.lb = max(cost_first, self.lb)

        return edges_popped

    def update_ub(self, nodes: List[Node]) -> None:
        # keep solved nodes for training
        for node in nodes:
            if (node.is_solved is not None) and node.is_solved and (self.ub > node.path_cost):
                self.goal_node = node
                self.ub = node.path_cost

    def finished(self) -> bool:
        return (self.goal_node is not None) and (self.lb >= (self.weight * self.ub))


D = TypeVar('D', bound=Domain)


class BWQSActsAny(PathFindQHeur[D, InstanceBWQS], ABC):
    def __init__(self, domain: D, batch_size: int = 1, weight: float = 1.0, eps: float = 0.0):
        super().__init__(domain)
        self.batch_size_default: int = batch_size
        self.weight_default: float = weight
        self.eps_default: float = eps

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       batch_size: Optional[int] = None, weight: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceBWQS]:
        """ Always computes the root heurisitc always (regardless of argument)

        """

        nodes_root: List[Node] = self._create_root_nodes(states, goals, compute_root_heur=True)
        batch_size_inst: int = batch_size if batch_size is not None else self.batch_size_default
        weight_inst: float = weight if weight is not None else self.weight_default
        eps_inst: float = eps if eps is not None else self.eps_default
        if inst_infos is None:
            inst_infos = [None for _ in states]
        return [InstanceBWQS(node_root, batch_size_inst, weight_inst, eps_inst, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]

    def step(self, verbose: bool = False) -> List[EdgeQ]:
        # split instances by iteration
        instances: List[InstanceBWQS] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get last popped nodes
        nodes_popped_last_by_inst: List[List[Node]] = [instance.popped_nodes_last for instance in instances]

        # is solved
        start_time = time.time()
        nodes_next_flat: List[Node] = misc_utils.flatten(nodes_popped_last_by_inst)[0]
        self.set_is_solved(nodes_next_flat)
        self.times.record_time("is_solved", time.time() - start_time)

        # ub
        start_time = time.time()
        for instance, nodes_popped_last in zip(instances, nodes_popped_last_by_inst, strict=True):
            instance.update_ub(nodes_popped_last)
        self.times.record_time("ub", time.time() - start_time)

        # check closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_popped_last_by_inst[inst_idx] = instance.check_closed(nodes_popped_last_by_inst[inst_idx])
        self.times.record_time("check", time.time() - start_time)

        # make edges
        start_time = time.time()
        edges_by_inst: List[List[EdgeQ]] = []

        for nodes_popped_last in nodes_popped_last_by_inst:
            edges: List[EdgeQ] = []
            for node in nodes_popped_last:
                assert node.q_values is not None
                for action, q_val in zip(node.q_values[0], node.q_values[1], strict=True):
                    edges.append(EdgeQ(node, action, q_val))
            edges_by_inst.append(edges)
        self.times.record_time("edges", time.time() - start_time)

        # costs
        start_time = time.time()
        edges_flat: List[EdgeQ] = misc_utils.flatten(edges_by_inst)[0]
        weights_flat, split_idxs = misc_utils.flatten([[instance.weight] * len(edges) for instance, edges in zip(instances, edges_by_inst, strict=True)])
        path_costs_flat: List[float] = [edge.node.path_cost for edge in edges_flat]
        qvals_flat: List[float] = [edge.q_val for edge in edges_flat]
        costs_flat: List[float] = ((np.array(weights_flat) * np.array(path_costs_flat)) + np.array(qvals_flat)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)
        self.times.record_time("cost", time.time() - start_time)

        # push to open
        start_time = time.time()
        for instance, edges, costs in zip(instances, edges_by_inst, costs_by_inst, strict=True):
            instance.push_to_open(edges, costs)
        self.times.record_time("push", time.time() - start_time)

        # pop from open
        start_time = time.time()
        edges_popped_by_inst: List[List[EdgeQ]] = [instance.pop_from_open() for instance in instances]
        self.times.record_time("pop", time.time() - start_time)

        # next nodes
        nodes_next_by_inst: List[List[Node]] = self.get_next_nodes(instances, edges_popped_by_inst)
        for instance, nodes_next in zip(instances, nodes_next_by_inst):
            instance.popped_nodes_last = nodes_next

        # verbose
        edges_popped_flat: List[EdgeQ] = misc_utils.flatten(edges_popped_by_inst)[0]
        if verbose:
            if len(edges_popped_flat) > 0:
                qvals_popped_flat: List[float] = [edge.q_val for edge in edges_popped_flat]
                min_heur = float(np.min(qvals_popped_flat))
                min_heur_pc = float(path_costs_flat[np.argmin(qvals_popped_flat)])
                max_heur = float(np.max(qvals_popped_flat))
                max_heur_pc = float(path_costs_flat[np.argmax(qvals_popped_flat)])
                per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
                per_finished: float = 100.0 * float(np.mean([inst.finished() for inst in instances]))

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
        return edges_popped_flat

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size_default}, weight={self.weight_default}, eps={self.eps_default})"


@pathfinding_factory.register_class("bwqs")
class BWQS(BWQSActsAny[ActsEnum], PathFindQExpandEnum[InstanceBWQS]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_parser("bwqs")
class BWQSParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        assert len(args_str_l) == 3
        return {"batch_size": int(args_str_l[0]), "weight": float(args_str_l[1]), "eps": float(args_str_l[2])}

    def help(self) -> str:
        return "The batch size, weight, and random node expansion probability (eps). E.g. 'bwqs.1_0.9_0.1'"
