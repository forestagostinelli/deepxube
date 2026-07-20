from abc import ABC, abstractmethod
from typing import List, Any, Type, Optional, TypeVar, Dict
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Goal
from deepxube.base.pathfinding import (Instance, InstanceNode, InstanceEdge, Node, EdgeQ, PFNsT, PFNsHV_T, PFNsHQ_T, PathFind, PathFindNode, PathFindEdge,
                                       PathFindActsPolicy, PathFindSetPolicy, PathFindSetHeurV, PathFindSetHeurQ, PathFindActsEnum)
from deepxube.base.pathfind_fns import PFNsHeurV, PFNsHeurQ, PFNsPolicy, PFNsHeurVPolicy, PFNsHeurQPolicy
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.utils.misc_utils import boltzmann
import numpy as np
import time
import re


class InstanceBeam(Instance, ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.beam_size: int = 1
        self.temp: float = 0.0
        self.eps: float = 0.0
        self.rollout: bool = False

    def set_beam_size(self, beam_size: int) -> None:
        assert beam_size >= 1
        self.beam_size = beam_size

    def set_temp(self, temp: float) -> None:
        assert temp >= 0.0
        self.temp = temp

    def set_eps(self, eps: float) -> None:
        assert (eps >= 0.0) and (eps <= 1.0)
        self.eps = eps

    def set_rollout(self, rollout: bool) -> None:
        self.rollout = rollout

    def frontier_size(self) -> int:
        return len(self._nodes_curr)

    def record_goal(self, nodes: List[Node]) -> None:
        if self.rollout:
            assert len(nodes) == 1
            self.goal_node = nodes[0]
        else:
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
        if self.rollout:
            return False
        else:
            return self.has_soln()


D = TypeVar('D', bound=Domain)
IBeam = TypeVar('IBeam', bound=InstanceBeam)


class BeamSearch(PathFind[D, PFNsT, IBeam], ABC):
    def __init__(self, *args: Any, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False, **kwargs: Any):
        self.beam_size_default: int = beam_size
        self.temp_default: float = temp
        self.eps_default: float = eps
        self.rollout: bool = rollout
        super().__init__(*args, **kwargs)

    def _construct_instances(self, inst_cls: type[IBeam], nodes_root: List[Node], inst_infos: Optional[List[Any]], beam_size: Optional[int],
                             temp: Optional[float], eps: Optional[float], compute_root_vals: bool) -> List[IBeam]:
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
            instance.set_rollout(self.rollout)

        if compute_root_vals:
            self._set_node_vals([[node] for node in nodes_root], instances)

        return instances

    def __repr__(self) -> str:
        return f"{type(self).__name__}(beam_size={self.beam_size_default}, temp={self.temp_default}, eps={self.eps_default}, rollout={self.rollout})"


class InstanceNodeBeam(InstanceNode, InstanceBeam):
    def filter_expanded_nodes(self, nodes: List[Node]) -> List[Node]:
        return nodes

    def push_pop_nodes(self, nodes: List[Node], costs: List[float]) -> List[Node]:
        next_idxs: List[int] = self.select_idxs_from_logits(costs)
        return [nodes[idx] for idx in next_idxs]


class InstanceEdgeBeam(InstanceEdge, InstanceBeam):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.beam_edges: List[EdgeQ] = []

    def filter_popped_nodes(self, nodes: List[Node]) -> List[Node]:
        return nodes

    def push_pop_edges(self, edges: List[EdgeQ], costs: List[float]) -> List[EdgeQ]:
        next_idxs: List[int] = self.select_idxs_from_logits(costs)
        return [edges[idx] for idx in next_idxs]


@pathfinding_factory.register_class("beam_p")
class BeamSearchPolicy(BeamSearch[Domain, PFNsPolicy, InstanceEdgeBeam], PathFindEdge[Domain, PFNsPolicy, InstanceEdgeBeam],
                       PathFindActsPolicy[Domain, PFNsPolicy, InstanceEdgeBeam], PathFindSetPolicy[Domain, PFNsPolicy, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_functions_type() -> Type[PFNsPolicy]:
        return PFNsPolicy

    @staticmethod
    def description() -> str:
        return "Beam search with policy that samples edges with corresponding probabilities"

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_vals: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals)
        instances: List[InstanceEdgeBeam] = self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps, True)
        return instances

    def _compute_costs(self, instances: List[InstanceEdgeBeam], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        start_time = time.time()
        logits_by_inst: List[List[float]] = []
        for edges in edges_by_inst:
            logits: List[float] = [edge.q_val for edge in edges]  # corresponds to prob densities
            logits_by_inst.append(logits)

        self.times.record_time("logits", time.time() - start_time)

        return logits_by_inst


class BeamSearchHeurNode(BeamSearch[D, PFNsHV_T, InstanceNodeBeam], PathFindNode[D, PFNsHV_T, InstanceNodeBeam],
                         PathFindSetHeurV[D, PFNsHV_T, InstanceNodeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_vals: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceNodeBeam]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals)
        return self._construct_instances(InstanceNodeBeam, nodes_root, inst_infos, beam_size, temp, eps, compute_root_vals)

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


class BeamSearchHeurEdge(BeamSearch[D, PFNsHQ_T, InstanceEdgeBeam], PathFindEdge[D, PFNsHQ_T, InstanceEdgeBeam],
                         PathFindSetHeurQ[D, PFNsHQ_T, InstanceEdgeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_vals: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals)
        return self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps, True)

    def _compute_costs(self, instances: List[InstanceEdgeBeam], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        start_time = time.time()
        logits_by_inst: List[List[float]] = []
        for edges in edges_by_inst:
            logits: List[float] = [-edge.q_val for edge in edges]
            logits_by_inst.append(logits)

        self.times.record_time("logits", time.time() - start_time)

        return logits_by_inst


@pathfinding_factory.register_class("beam_v")
class BeamSearchHeurNodeActsEnum(BeamSearchHeurNode[ActsEnum, PFNsHeurV], PathFindActsEnum[ActsEnum, PFNsHeurV, InstanceNodeBeam]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum

    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurV]:
        return PFNsHeurV

    @staticmethod
    def description() -> str:
        return "Beam search with heuristic that prioritizes nodes"


@pathfinding_factory.register_class("beam_q")
class BeamSearchHeurEdgeActsEnum(BeamSearchHeurEdge[ActsEnum, PFNsHeurQ], PathFindActsEnum[ActsEnum, PFNsHeurQ, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum

    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQ]:
        return PFNsHeurQ

    @staticmethod
    def description() -> str:
        return "Beam search with heuristic that prioritizes edges"


@pathfinding_factory.register_class("beam_v_p")
class BeamSearchHeurNodeActsPolicy(BeamSearchHeurNode[Domain, PFNsHeurVPolicy], PathFindActsPolicy[Domain, PFNsHeurVPolicy, InstanceNodeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurVPolicy]:
        return PFNsHeurVPolicy

    @staticmethod
    def description() -> str:
        return "Beam search with heuristic that prioritizes nodes and policy that samples edges"

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(beam_size={self.beam_size_default}, temp={self.temp_default}, eps={self.eps_default}, rollout={self.rollout}, "
                f"num_rand_edges={self.num_rand_edges})")


@pathfinding_factory.register_class("beam_q_p")
class BeamSearchHeurEdgeActsPolicy(BeamSearchHeurEdge[Domain, PFNsHeurQPolicy], PathFindActsPolicy[Domain, PFNsHeurQPolicy, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQPolicy]:
        return PFNsHeurQPolicy

    @staticmethod
    def description() -> str:
        return "Beam search with heuristic that prioritizes edges and policy that samples edges"

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(beam_size={self.beam_size_default}, temp={self.temp_default}, eps={self.eps_default}, rollout={self.rollout}, "
                f"num_rand_edges={self.num_rand_edges})")


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
        kwargs["rollout"] = False
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


class BeamSearchHasPolicyParser(Parser, ABC):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            beam_re = re.search(r"^(\S+)B$", args_str_i)
            temp_re = re.search(r"^(\S+)T", args_str_i)
            eps_re = re.search(r"^(\S+)E", args_str_i)
            num_rand_edges = re.search(r"^(\S+)R", args_str_i)
            if beam_re is not None:
                kwargs["beam_size"] = int(beam_re.group(1))
            elif temp_re is not None:
                kwargs["temp"] = float(temp_re.group(1))
            elif eps_re is not None:
                kwargs["eps"] = float(eps_re.group(1))
            elif num_rand_edges is not None:
                kwargs["num_rand_edges"] = int(num_rand_edges.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        kwargs["rollout"] = False
        return kwargs

    def help(self) -> str:
        return ("<int>B (beam size), <float>T (temperature for Boltzmann distribution), <float>E (epsilon for chance to randomly select node), "
                "<int>R (num rand edges).\n"
                f"E.g. {self._alg_name()}.10B_1.0T_0.1E_5R")

    @abstractmethod
    def _alg_name(self) -> str:
        pass


@pathfinding_factory.register_parser("beam_v_p")
class BeamSearchNodeHasPolicyParser(BeamSearchHasPolicyParser):
    def _alg_name(self) -> str:
        return "beam_v_p"


@pathfinding_factory.register_parser("beam_q_p")
class BeamSearchEdgeHasPolicyParser(BeamSearchHasPolicyParser):
    def _alg_name(self) -> str:
        return "beam_q_p"


# Rollout - special case of beam search where no goal test is done

@pathfinding_factory.register_class("rollout_p")
class RolloutPolicy(BeamSearchPolicy):
    @staticmethod
    def description() -> str:
        return "Rollout policy put do not terminate if a goal is seen"


class RolloutParser(Parser, ABC):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            temp_re = re.search(r"^(\S+)T", args_str_i)
            eps_re = re.search(r"^(\S+)E", args_str_i)
            if temp_re is not None:
                kwargs["temp"] = float(temp_re.group(1))
            elif eps_re is not None:
                kwargs["eps"] = float(eps_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        kwargs["beam_size"] = 1
        kwargs["rollout"] = True
        return kwargs

    def help(self) -> str:
        return ("<float>T (temperature for Boltzmann distribution), <float>E (epsilon for chance to randomly select node).\n"
                f"E.g. {self._alg_name()}.1.0T_0.1E")

    @abstractmethod
    def _alg_name(self) -> str:
        pass


@pathfinding_factory.register_parser("rollout_p")
class RolloutPolicyParser(RolloutParser):
    def _alg_name(self) -> str:
        return "rollout_p"
