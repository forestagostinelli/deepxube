from abc import ABC, abstractmethod
from typing import List, Any, Type, Optional, TypeVar, Tuple, Dict, Set
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Action, Goal
from deepxube.base.pathfinding import (Instance, SchOver, InstanceNode, InstanceEdge, Node, EdgeQ, PathFind, PathFindEdgeActsPolicy,
                                       PathFindNodeActsPolicy, PathFindNodeHasHeur, PathFindEdgeHasHeur, PathFindNodeActsEnum, PathFindEdgeActsEnum)
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.utils import misc_utils
from deepxube.utils.misc_utils import boltzmann
import numpy as np
import time
import re


class InstanceBeam(Instance[SchOver], ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.beam: List[Node] = [root_node]
        self.beam_size: int = 1
        self.temp: float = 0.0
        self.eps: float = 0.0

    def set_beam_size(self, beam_size: int) -> None:
        assert beam_size >= 1
        self.beam_size = beam_size

    def set_temp(self, temp: float) -> None:
        assert temp >= 0.0
        self.temp = temp

    def set_eps(self, eps: float) -> None:
        assert (eps >= 0.0) and (eps <= 1.0)
        self.eps = eps

    def frontier_size(self) -> int:
        return len(self.beam)

    def pop_nodes(self) -> List[Node]:
        return self.beam.copy()

    def record_goal(self, nodes: List[Node]) -> None:
        for node in nodes:
            assert node.is_solved is not None
            if node.is_solved:
                if (self.goal_node is None) or (self.goal_node.path_cost > node.path_cost):
                    self.goal_node = node

    def select_idxs_from_logits(self, logits: List[float]) -> List[int]:
        num_logits: int = len(logits)
        next_idxs: List[int]
        if len(logits) <= self.beam_size:
            next_idxs = list(range(num_logits))
        else:
            # get next idxs
            if self.temp == 0:
                next_idxs = (np.argpartition(logits, -self.beam_size)[-self.beam_size:]).tolist()
            else:
                # select next according to boltzmann
                probs: List[float] = boltzmann(logits, self.temp)
                next_idxs = np.random.choice(num_logits, size=self.beam_size, replace=False, p=probs).tolist()

            # randomly select index
            if self.eps > 0:
                replace_rand_idxs: List[int] = np.where(np.random.random(len(next_idxs)) < self.eps)[0].tolist()
                num_replace: int = len(replace_rand_idxs)
                if num_replace > 0:
                    next_idxs_rand: List[int] = np.random.choice(num_logits, replace=False, size=num_replace).tolist()
                    for replace_idx, next_idx_rand in zip(replace_rand_idxs, next_idxs_rand):
                        next_idxs[replace_idx] = next_idx_rand
                    next_idxs = list(set(next_idxs))
        return next_idxs

    def finished(self) -> bool:
        return self.has_soln()


D = TypeVar('D', bound=Domain)
IBeam = TypeVar('IBeam', bound=InstanceBeam)


class BeamSearch(PathFind[D, IBeam, SchOver], ABC):
    def __init__(self, domain: D, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0):
        super().__init__(domain)
        self.beam_size_default: int = beam_size
        self.temp_default: float = temp
        self.eps_default: float = eps

    def _construct_instances(self, inst_cls: type[IBeam], nodes_root: List[Node], inst_infos: Optional[List[Any]], beam_size: Optional[int],
                             temp: Optional[float], eps: Optional[float]) -> List[IBeam]:
        if inst_infos is None:
            inst_infos = [None for _ in nodes_root]

        beam_size_inst: int = beam_size if beam_size is not None else self.beam_size_default
        temp_inst: float = temp if temp is not None else self.temp_default
        eps_inst: float = eps if eps is not None else self.eps_default

        instances: List[IBeam] = [inst_cls(node_root, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]
        for instance in instances:
            instance.set_beam_size(beam_size_inst)
            instance.set_temp(temp_inst)
            instance.set_eps(eps_inst)

        return instances

    def __repr__(self) -> str:
        return f"{type(self).__name__}(beam_size={self.beam_size_default}, temp={self.temp_default}, eps={self.eps_default})"


class InstanceNodeBeam(InstanceNode, InstanceBeam[Node]):
    def filter_expanded_nodes(self, nodes: List[Node]) -> List[Node]:
        return nodes

    def push_nodes(self, nodes: List[Node], costs: List[float]) -> None:
        next_idxs: List[int] = self.select_idxs_from_logits(costs)
        self.beam = [nodes[idx] for idx in next_idxs]


class InstanceEdgeBeam(InstanceEdge, InstanceBeam):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.beam_edges: List[EdgeQ] = []

    def filter_popped_nodes(self, nodes: List[Node]) -> List[Node]:
        return nodes

    def push_edges(self, edges: List[EdgeQ], costs: List[float]) -> List[EdgeQ]:
        next_idxs: List[int] = self.select_idxs_from_logits(costs)
        return [edges[idx] for idx in next_idxs]

    def set_next_nodes(self, nodes_next: List[Node]) -> None:
        self.beam = nodes_next.copy()


@pathfinding_factory.register_class("beam_p")
class BeamSearchPolicy(BeamSearch[Domain, InstanceEdgeBeam, EdgeQ], PathFindEdgeActsPolicy[Domain, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals)
        self._set_node_act_probs(nodes_root)
        return self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _compute_costs(self, instances: List[InstanceEdgeBeam], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        start_time = time.time()
        logits_by_inst: List[List[float]] = []
        for edges in edges_by_inst:
            logits: List[float] = [edge.q_val for edge in edges]  # corresponds to prob densities
            logits_by_inst.append(logits)

        self.times.record_time("logits", time.time() - start_time)

        return logits_by_inst

    def _eval_nodes(self, instances: List[InstanceEdgeBeam], nodes_by_inst: List[List[Node]]) -> None:
        self._set_node_act_probs(misc_utils.flatten(nodes_by_inst)[0])


class BeamSearchHeurNode(BeamSearch[D, InstanceNodeBeam, Node], PathFindNodeHasHeur[D, InstanceNodeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceNodeBeam]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, compute_root_heur)
        return self._construct_instances(InstanceNodeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _compute_costs(self, instances: List[InstanceNodeBeam], nodes_by_inst: List[List[Node]]) -> List[List[float]]:
        start_time = time.time()
        logits_by_inst: List[List[float]] = []
        for nodes in nodes_by_inst:
            logits: List[float] = []
            for node in nodes:
                assert node.parent_t_cost is not None
                logits.append(-(node.parent_t_cost + node.heuristic))

            logits_by_inst.append(logits)

        self.times.record_time("logits", time.time() - start_time)

        return logits_by_inst


class BeamSearchHeurEdge(BeamSearch[D, InstanceEdgeBeam, EdgeQ], PathFindEdgeHasHeur[D, InstanceEdgeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, True)
        return self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _compute_costs(self, instances: List[InstanceEdgeBeam], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        start_time = time.time()
        logits_by_inst: List[List[float]] = []
        for edges in edges_by_inst:
            logits: List[float] = [-edge.q_val for edge in edges]
            logits_by_inst.append(logits)

        self.times.record_time("logits", time.time() - start_time)

        return logits_by_inst

    def _eval_nodes(self, instances: List[InstanceEdgeBeam], nodes_by_inst: List[List[Node]]) -> None:
        self._set_node_heurs(misc_utils.flatten(nodes_by_inst)[0])


@pathfinding_factory.register_class("beam_v")
class BeamSearchHeurNodeActsEnum(BeamSearchHeurNode[ActsEnum], PathFindNodeActsEnum[ActsEnum, InstanceNodeBeam]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_class("beam_q")
class BeamSearchHeurEdgeActsEnum(BeamSearchHeurEdge[ActsEnum], PathFindEdgeActsEnum[ActsEnum, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_class("beam_v_p")
class BeamSearchHeurNodeActsPolicy(BeamSearchHeurNode[Domain], PathFindNodeActsPolicy[Domain, InstanceNodeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain


@pathfinding_factory.register_class("beam_q_p")
class BeamSearchHeurEdgeActsPolicy(BeamSearchHeurEdge[Domain], PathFindEdgeActsPolicy[Domain, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain


class BeamSearchParser(Parser, ABC):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            beam_re = re.search(r"^(\S+)B$", args_str_i)
            temp_re = re.search(r"^(\S+)T", args_str_i)
            eps_re = re.search(r"^(\S+)E", args_str_i)
            if beam_re is not None:
                kwargs["beam_size"] = int(beam_re.group(1))
            elif temp_re is not None:
                kwargs["temp"] = float(temp_re.group(1))
            elif eps_re is not None:
                kwargs["eps"] = float(eps_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("<int>B (beam size), <float>T (temperature for Boltzmann distribution), <float>E (epsilon for chance to randomly select node).\n"
                f"E.g. {self._alg_name()}.10B_1.0T_0.1E")

    @abstractmethod
    def _alg_name(self) -> str:
        pass


@pathfinding_factory.register_parser("beam_p")
class BeamSearchPolicyParser(BeamSearchParser):
    def _alg_name(self) -> str:
        return "beam_p"


@pathfinding_factory.register_parser("beam_v")
class BeamSearchNodeParser(BeamSearchParser):
    def _alg_name(self) -> str:
        return "beam_v"


@pathfinding_factory.register_parser("beam_q")
class BeamSearchEdgeParser(BeamSearchParser):
    def _alg_name(self) -> str:
        return "beam_q"


@pathfinding_factory.register_parser("beam_v_p")
class BeamSearchNodeHasPolicyParser(BeamSearchParser):
    def _alg_name(self) -> str:
        return "beam_v_p"


@pathfinding_factory.register_parser("beam_q_p")
class BeamSearchEdgeHasPolicyParser(BeamSearchParser):
    def _alg_name(self) -> str:
        return "beam_q_p"
