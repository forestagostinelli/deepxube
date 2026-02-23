from abc import ABC, abstractmethod
from typing import List, Any, Type, Optional, TypeVar, Generic, Tuple, Dict
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Action, Goal
from deepxube.base.pathfinding import (Instance, InstanceNode, InstanceEdge, Node, EdgeQ, PathFind, PathFindEdgeActsPolicy,
                                       PathFindNodeActsPolicy, PathFindNodeHasHeur, PathFindEdgeHasHeur, PathFindNodeActsEnum, PathFindEdgeActsEnum)
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.utils import misc_utils
from deepxube.utils.misc_utils import boltzmann
import numpy as np
import time
import re


class InstanceBeam(Instance):
    def __init__(self, root_node: Node, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.beam: List[Node] = [self.root_node]
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

    def record_goal(self) -> None:
        for node in self.beam:
            assert node.is_solved is not None
            if node.is_solved:
                if (self.goal_node is None) or (self.goal_node.path_cost > node.path_cost):
                    self.goal_node = node

    def finished(self) -> bool:
        return self.has_soln()


D = TypeVar('D', bound=Domain)
IBeam = TypeVar('IBeam', bound=InstanceBeam)
StepRet = TypeVar('StepRet')


class BeamSearch(PathFind[D, IBeam], Generic[D, IBeam, StepRet], ABC):
    def __init__(self, domain: D, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0):
        super().__init__(domain)
        self.beam_size_default: int = beam_size
        self.temp_default: float = temp
        self.eps_default: float = eps

    def step(self, verbose: bool = False) -> List[StepRet]:
        # get unsolved instances
        instances: List[IBeam] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get beams
        start_time = time.time()
        nodes_by_inst: List[List[Node]] = [inst.beam.copy() for inst in instances]
        nodes_flat: List[Node] = misc_utils.flatten(nodes_by_inst)[0]
        self.times.record_time("get_beams", time.time() - start_time)

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes_flat)
        for instance in instances:
            instance.record_goal()
        self.times.record_time("is_solved", time.time() - start_time)

        # get edges
        edges_l, logits_l = self._get_edges_and_logits(instances, nodes_by_inst)

        # get next edges
        start_time = time.time()
        edges_next_l: List[List[EdgeQ]] = []
        for instance, edges, logits in zip(instances, edges_l, logits_l):
            edges_next: List[EdgeQ]
            if len(edges) <= instance.beam_size:
                edges_next = edges
            else:
                # get next idxs
                next_idxs: List[int]
                if instance.temp == 0:
                    next_idxs = (np.argpartition(logits, -instance.beam_size)[-instance.beam_size:]).tolist()
                else:
                    # select next according to boltzmann
                    probs: List[float] = boltzmann(logits, instance.temp)
                    next_idxs = np.random.choice(len(edges), size=instance.beam_size, replace=False, p=probs).tolist()

                # randomly select index
                if instance.eps > 0:
                    replace_rand_idxs: List[int] = np.where(np.random.random(len(next_idxs)) < instance.eps)[0].tolist()
                    num_replace: int = len(replace_rand_idxs)
                    if num_replace > 0:
                        next_idxs_rand: List[int] = np.random.choice(len(edges), replace=False, size=num_replace).tolist()
                        for replace_idx, next_idx_rand in zip(replace_rand_idxs, next_idxs_rand):
                            next_idxs[replace_idx] = next_idx_rand
                        next_idxs = list(set(next_idxs))

                edges_next = [edges[idx] for idx in next_idxs]
            edges_next_l.append(edges_next)

        self.times.record_time("next_edges", time.time() - start_time)

        # get next nodes
        nodes_next_l, step_ret = self._get_next_nodes(instances, edges_next_l)

        # update inst
        start_time = time.time()
        for instance, nodes_next in zip(instances, nodes_next_l):
            instance.beam = nodes_next
            instance.itr += 1
        self.times.record_time("set_next", time.time() - start_time)

        if verbose:
            heuristics: List[float] = [node.heuristic for node in nodes_flat]
            path_costs: List[float] = [node.path_cost for node in nodes_flat]
            beam_sizes: List[int] = [len(instance.beam) for instance in instances]
            min_heur = float(np.min(heuristics))
            min_heur_pc = float(path_costs[np.argmin(heuristics)])
            max_heur = float(np.max(heuristics))
            max_heur_pc = float(path_costs[np.argmax(heuristics)])
            per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
            per_finished: float = 100.0 * float(np.mean([inst.finished() for inst in instances]))

            print(f"Itr: %i, Heur(PathCost)(Min/Max): "
                  f"%.2E(%.2f)/%.2E(%.2f), Beam sizes(Min/Max): {min(beam_sizes)}/{max(beam_sizes)}, %%has_soln: {per_has_soln}, "
                  f"%%finished: {per_finished}" % (self.itr, min_heur, min_heur_pc, max_heur, max_heur_pc))

        self.itr += 1

        return step_ret

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

    @abstractmethod
    def _get_edges_and_logits(self, instances: List[IBeam], nodes_by_inst: List[List[Node]]) -> Tuple[List[List[EdgeQ]], List[List[float]]]:
        pass

    @abstractmethod
    def _get_next_nodes(self, instances: List[IBeam], edges_l: List[List[EdgeQ]]) -> Tuple[List[List[Node]], List[StepRet]]:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(beam_size={self.beam_size_default}, temp={self.temp_default}, eps={self.eps_default})"


class InstanceEdgeBeam(InstanceEdge, InstanceBeam):
    pass


class InstanceNodeBeam(InstanceNode, InstanceBeam):
    pass


@pathfinding_factory.register_class("beam_p")
class BeamSearchPolicy(BeamSearch[Domain, InstanceEdgeBeam, EdgeQ], PathFindEdgeActsPolicy[Domain, InstanceEdgeBeam]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes_policy(states, goals)
        return self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _get_edges_and_logits(self, instances: List[InstanceEdgeBeam], nodes_by_inst: List[List[Node]]) -> Tuple[List[List[EdgeQ]], List[List[float]]]:
        # set act probs
        nodes_flat: List[Node] = misc_utils.flatten(nodes_by_inst)[0]
        self._set_node_act_probs(nodes_flat)

        # get edges and logits
        start_time = time.time()
        edges_by_inst: List[List[EdgeQ]] = []
        logits_by_inst: List[List[float]] = []

        for nodes in nodes_by_inst:
            edges: List[EdgeQ] = []
            logits: List[float] = []
            for node in nodes:
                assert node.act_probs is not None
                for action, logit in zip(node.act_probs[0], node.act_probs[1], strict=True):
                    edges.append(EdgeQ(node, action, 0.0))
                    logits.append(logit)

            edges_by_inst.append(edges)
            logits_by_inst.append(logits)
        self.times.record_time("edges_logits", time.time() - start_time)

        return edges_by_inst, logits_by_inst

    def _get_next_nodes(self, instances: List[InstanceEdgeBeam], edges_l: List[List[EdgeQ]]) -> Tuple[List[List[Node]], List[EdgeQ]]:
        edges_flat: List[EdgeQ] = misc_utils.flatten(edges_l)[0]
        return self.get_next_nodes(instances, edges_l), edges_flat


class BeamSearchHeurNode(BeamSearch[D, InstanceNodeBeam, Node], PathFindNodeHasHeur[D, InstanceNodeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceNodeBeam]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, compute_root_heur)
        return self._construct_instances(InstanceNodeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _get_edges_and_logits(self, instances: List[InstanceNodeBeam], nodes_by_inst: List[List[Node]]) -> Tuple[List[List[EdgeQ]], List[List[float]]]:
        # expand
        nodes_next_l: List[List[Node]] = self._expand_nodes(instances, nodes_by_inst)

        # get child heuristic values
        nodes_next_flat: List[Node] = misc_utils.flatten(nodes_next_l)[0]
        self._set_node_heurs(nodes_next_flat)

        # get edges and logits
        start_time = time.time()
        edges_by_inst: List[List[EdgeQ]] = []
        logits_by_inst: List[List[float]] = []

        for nodes in nodes_by_inst:
            edges: List[EdgeQ] = []
            logits: List[float] = []
            for node in nodes:
                for action, _ in node.edge_dict.items():
                    edges.append(EdgeQ(node, action, 0.0))

                assert node.is_solved is not None
                if node.is_solved:
                    for _ in node.edge_dict.items():
                        logits.append(0.0)
                else:
                    for action, (t_cost, child) in node.edge_dict.items():
                        logits.append(-(t_cost + child.heuristic))

            edges_by_inst.append(edges)
            logits_by_inst.append(logits)
        self.times.record_time("edges_logits", time.time() - start_time)

        return edges_by_inst, logits_by_inst

    def _get_next_nodes(self, instances: List[InstanceNodeBeam], edges_l: List[List[EdgeQ]]) -> Tuple[List[List[Node]], List[Node]]:
        start_time = time.time()
        edges_flat, split_idxs = misc_utils.flatten(edges_l)
        nodes_flat: List[Node] = [edge.node for edge in edges_flat]
        actions_flat: List[Action] = [edge.action for edge in edges_flat]

        nodes_next_flat: List[Node] = [node.edge_dict[action][1] for node, action in zip(nodes_flat, actions_flat)]
        nodes_next: List[List[Node]] = misc_utils.unflatten(nodes_next_flat, split_idxs)

        self.times.record_time("node_edge_next", time.time() - start_time)

        return nodes_next, nodes_flat


class BeamSearchHeurEdge(BeamSearch[D, InstanceEdgeBeam, EdgeQ], PathFindEdgeHasHeur[D, InstanceEdgeBeam], ABC):
    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       beam_size: Optional[int] = None, temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceEdgeBeam]:
        nodes_root: List[Node] = self._create_root_nodes_heur(states, goals, compute_root_heur)
        return self._construct_instances(InstanceEdgeBeam, nodes_root, inst_infos, beam_size, temp, eps)

    def _get_edges_and_logits(self, instances: List[InstanceEdgeBeam], nodes_by_inst: List[List[Node]]) -> Tuple[List[List[EdgeQ]], List[List[float]]]:
        # get qvalues
        nodes_qvalues_none: List[Node] = [x for x in misc_utils.flatten(nodes_by_inst)[0] if x.q_values is None]
        if len(nodes_qvalues_none) > 0:
            self._set_node_heurs(nodes_qvalues_none)

        # get edges
        start_time = time.time()
        edges_by_inst: List[List[EdgeQ]] = []
        logits_by_inst: List[List[float]] = []

        for nodes in nodes_by_inst:
            edges: List[EdgeQ] = []
            logits: List[float] = []
            for node in nodes:
                assert node.q_values is not None
                for action, q_val in zip(node.q_values[0], node.q_values[1]):
                    edges.append(EdgeQ(node, action, 0.0))
                    logits.append(-q_val)

            edges_by_inst.append(edges)
            logits_by_inst.append(logits)
        self.times.record_time("edges_logits", time.time() - start_time)

        return edges_by_inst, logits_by_inst

    def _get_next_nodes(self, instances: List[InstanceEdgeBeam], edges_l: List[List[EdgeQ]]) -> Tuple[List[List[Node]], List[EdgeQ]]:
        nodes_next_l: List[List[Node]] = self.get_next_nodes(instances, edges_l)
        nodes_next_flat: List[Node] = misc_utils.flatten(nodes_next_l)[0]
        self._set_node_heurs(nodes_next_flat)

        edges_flat: List[EdgeQ] = misc_utils.flatten(edges_l)[0]

        return nodes_next_l, edges_flat


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
