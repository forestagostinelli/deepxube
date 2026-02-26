from abc import ABC, abstractmethod
from typing import List, Any, Type, Optional, TypeVar, Generic, Tuple, Dict
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Goal
from deepxube.base.pathfinding import (Instance, InstanceNode, InstanceEdge, Node, EdgeQ, PathFind, PathFindEdgeActsPolicy,
                                       PathFindNodeActsPolicy, PathFindNodeHasHeur, PathFindEdgeHasHeur, PathFindNodeActsEnum, PathFindEdgeActsEnum)
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.utils import misc_utils
from heapq import heappush, heappop, heapify
import numpy as np
import random
import time
import re


SchOver = TypeVar("SchOver")


class InstanceGraph(Instance, Generic[SchOver]):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.open_set: List[Tuple[float, int, SchOver]] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = {}
        self.ub: float = np.inf
        self.lb: float = self.root_node.heuristic
        self.batch_size: int = 1
        self.weight: float = 1.0
        self.eps: float = 0.0

    def set_batch_size(self, batch_size: int) -> None:
        assert batch_size >= 1
        self.batch_size = batch_size

    def set_weight(self, weight: float) -> None:
        assert (weight <= 1) and (weight >= 0)
        self.weight = weight

    def set_eps(self, eps: float) -> None:
        assert (eps <= 1) and (eps >= 0)
        self.eps = eps

    def frontier_size(self) -> int:
        return len(self.open_set)

    def record_goal(self, nodes: List[Node]) -> None:
        # keep solved nodes for training
        for node in nodes:
            assert node.is_solved is not None
            if node.is_solved and (self.ub > node.path_cost):
                self.goal_node = node
                self.ub = node.path_cost

    def finished(self) -> bool:
        case1: bool = (self.goal_node is not None) and (self.lb >= (self.weight * self.ub))
        case2: bool = (self.itr > 0) and (len(self.open_set) == 0)
        return case1 or case2

    def _push_to_open(self, sch_over_l: List[SchOver], costs: List[float]) -> None:
        for sch_over, cost in zip(sch_over_l, costs, strict=True):
            heappush(self.open_set, (cost, self.heappush_count, sch_over))
            self.heappush_count += 1

    def _pop_from_open(self) -> List[SchOver]:
        num_to_pop: int = min(self.batch_size, len(self.open_set))

        elems_popped: List[Tuple[float, int, SchOver]] = []
        for _ in range(num_to_pop):
            if random.random() < self.eps:
                pop_idx: int = random.randrange(0, len(self.open_set))
                elems_popped.append(self.open_set.pop(pop_idx))
                heapify(self.open_set)
            else:
                elems_popped.append(heappop(self.open_set))

        sch_over_popped: List[SchOver] = [elem_popped[2] for elem_popped in elems_popped]

        if len(elems_popped) > 0:
            cost_first: float = elems_popped[0][0]
            self.lb = max(cost_first, self.lb)

        return sch_over_popped

    def _check_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_ret: List[Node] = []
        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                nodes_ret.append(node)
        return nodes_ret


D = TypeVar('D', bound=Domain)
IGraph = TypeVar('IGraph', bound=InstanceGraph)


class GraphSearch(PathFind[D, IGraph], ABC):
    def __init__(self, domain: D, batch_size: int = 1, weight: float = 1.0, eps: float = 0.0):
        super().__init__(domain)
        self.batch_size_default: int = batch_size
        self.weight_default: float = weight
        self.eps_default: float = eps

    def _construct_instances(self, inst_cls: type[IGraph], nodes_root: List[Node], inst_infos: Optional[List[Any]], batch_size: Optional[int],
                             weight: Optional[float], eps: Optional[float]) -> List[IGraph]:
        if inst_infos is None:
            inst_infos = [None for _ in nodes_root]

        batch_size_inst: int = batch_size if batch_size is not None else self.batch_size_default
        weight_inst: float = weight if weight is not None else self.weight_default
        eps_inst: float = eps if eps is not None else self.eps_default

        instances: List[IGraph] = [inst_cls(node_root, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]
        for instance in instances:
            instance.set_batch_size(batch_size_inst)
            instance.set_weight(weight_inst)
            instance.set_eps(eps_inst)

        return instances

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size_default}, weight={self.weight_default}, eps={self.eps_default})"


class InstanceNodeGraph(InstanceNode, InstanceGraph[Node]):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.closed_dict[self.root_node.state] = 0.0

    def filter_expanded_nodes(self, nodes: List[Node]) -> List[Node]:
        return self._check_closed(nodes)

    def push_pop_nodes(self, nodes: List[Node], costs: List[float]) -> List[Node]:
        self._push_to_open(nodes, costs)
        return self._pop_from_open()


class InstanceEdgeGraph(InstanceEdge, InstanceGraph[EdgeQ]):
    def filter_popped_nodes(self, nodes: List[Node]) -> List[Node]:
        return self._check_closed(nodes)

    def push_pop_edges(self, edges: List[EdgeQ], costs: List[float]) -> List[EdgeQ]:
        self._push_to_open(edges, costs)
        return self._pop_from_open()


class GraphSearchHeurNode(GraphSearch[D, InstanceNodeGraph], PathFindNodeHasHeur[D, InstanceNodeGraph], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, weight: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceNodeGraph]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, compute_root_heur)
        return self._construct_instances(InstanceNodeGraph, nodes_root, inst_infos, beam_size, weight, eps)

    def _compute_costs(self, instances: List[InstanceNodeGraph], nodes_by_inst: List[List[Node]]) -> List[List[float]]:
        start_time = time.time()
        nodes_c_flat: List[Node] = misc_utils.flatten(nodes_by_inst)[0]
        weights, split_idxs = misc_utils.flatten([[instance.weight] * len(nodes_c) for instance, nodes_c in zip(instances, nodes_by_inst, strict=True)])
        path_costs: List[float] = [node.path_cost for node in nodes_c_flat]
        heuristics: List[float] = [node.heuristic for node in nodes_c_flat]
        costs_flat: List[float] = ((np.array(weights) * np.array(path_costs)) + np.array(heuristics)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)

        self.times.record_time("cost", time.time() - start_time)

        return costs_by_inst


class GraphSearchHeurEdge(GraphSearch[D, InstanceEdgeGraph], PathFindEdgeHasHeur[D, InstanceEdgeGraph], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       batch_size: Optional[int] = None, weight: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeGraph]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, True)
        return self._construct_instances(InstanceEdgeGraph, nodes_root, inst_infos, batch_size, weight, eps)

    def _compute_costs(self, instances: List[InstanceEdgeGraph], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        start_time = time.time()
        edges_flat: List[EdgeQ] = misc_utils.flatten(edges_by_inst)[0]
        weights_flat, split_idxs = misc_utils.flatten([[instance.weight] * len(edges) for instance, edges in zip(instances, edges_by_inst, strict=True)])
        path_costs_flat: List[float] = [edge.node.path_cost for edge in edges_flat]
        qvals_flat: List[float] = [edge.q_val for edge in edges_flat]
        costs_flat: List[float] = ((np.array(weights_flat) * np.array(path_costs_flat)) + np.array(qvals_flat)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)

        self.times.record_time("cost", time.time() - start_time)

        return costs_by_inst

    def _eval_nodes(self, instances: List[InstanceEdgeGraph], nodes_by_inst: List[List[Node]]) -> None:
        self._set_node_heurs(misc_utils.flatten(nodes_by_inst)[0])


@pathfinding_factory.register_class("graph_v")
class GraphSearchHeurNodeActsEnum(GraphSearchHeurNode[ActsEnum], PathFindNodeActsEnum[ActsEnum, InstanceNodeGraph]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_class("graph_q")
class GraphSearchHeurEdgeActsEnum(GraphSearchHeurEdge[ActsEnum], PathFindEdgeActsEnum[ActsEnum, InstanceEdgeGraph]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_class("graph_v_p")
class GraphSearchHeurNodeActsPolicy(GraphSearchHeurNode[Domain], PathFindNodeActsPolicy[Domain, InstanceNodeGraph]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain


@pathfinding_factory.register_class("graph_q_p")
class GraphSearchHeurEdgeActsPolicy(GraphSearchHeurEdge[Domain], PathFindEdgeActsPolicy[Domain, InstanceEdgeGraph]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain


class GraphSearchParser(Parser, ABC):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            batch_size_re = re.search(r"^(\S+)B$", args_str_i)
            weight_re = re.search(r"^(\S+)W", args_str_i)
            eps_re = re.search(r"^(\S+)E", args_str_i)
            if batch_size_re is not None:
                kwargs["batch_size"] = int(batch_size_re.group(1))
            elif weight_re is not None:
                kwargs["weight"] = float(weight_re.group(1))
            elif eps_re is not None:
                kwargs["eps"] = float(eps_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("<int>B (batch size), <float>W (weight), <float>E (epsilon for chance to randomly pop node).\n"
                f"E.g. {self._alg_name()}.10B_0.5W_0.1E")

    @abstractmethod
    def _alg_name(self) -> str:
        pass


@pathfinding_factory.register_parser("graph_v")
class GraphSearchNodeParser(GraphSearchParser):
    def _alg_name(self) -> str:
        return "graph_v"


@pathfinding_factory.register_parser("graph_q")
class GraphSearchEdgeParser(GraphSearchParser):
    def _alg_name(self) -> str:
        return "graph_q"


@pathfinding_factory.register_parser("graph_v_p")
class GraphSearchNodeHasPolicyParser(GraphSearchParser):
    def _alg_name(self) -> str:
        return "graph_v_p"


@pathfinding_factory.register_parser("graph_q_p")
class GraphSearchEdgeHasPolicyParser(GraphSearchParser):
    def _alg_name(self) -> str:
        return "graph_q_p"
