from abc import ABC
from typing import List, Any, Type, Optional, TypeVar, Dict
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Goal
from deepxube.base.pathfinding import Instance, Node, PathFindVHeur, PathFindVExpandEnum
from deepxube.pathfinding.pathfinding_utils import greedy_next_idx
from deepxube.factories.pathfinding_factory import pathfinding_factory
import numpy as np
import time


class InstanceGrPolV(Instance):
    def __init__(self, root_node: Node, temp: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_node: Node = self.root_node
        self.temp: float = temp
        self.eps: float = eps

    def record_goal(self) -> None:
        assert self.curr_node.is_solved is not None
        if self.curr_node.is_solved:
            if (self.goal_node is None) or (self.goal_node.path_cost > self.curr_node.path_cost):
                self.goal_node = self.curr_node

    def finished(self) -> bool:
        return self.has_soln()


D = TypeVar('D', bound=Domain)


class GreedyPolicyActAny(PathFindVHeur[D, InstanceGrPolV], ABC):
    def __init__(self, domain: D, temp: float = 1.0, eps: float = 0.0):
        super().__init__(domain)
        self.temp_default: float = temp
        self.eps_default: float = eps

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceGrPolV]:
        nodes_root: List[Node] = self._create_root_nodes(states, goals, compute_root_heur=compute_root_heur)
        temp_inst: float = temp if temp is not None else self.temp_default
        eps_inst: float = eps if eps is not None else self.eps_default
        if inst_infos is None:
            inst_infos = [None for _ in states]
        return [InstanceGrPolV(node_root, temp_inst, eps_inst, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]

    def step(self, verbose: bool = False) -> List[Node]:
        # get unsolved instances
        instances: List[InstanceGrPolV] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get curr nodes
        nodes_curr: List[Node] = [inst.curr_node for inst in instances]

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes_curr)
        for instance in instances:
            instance.record_goal()
        self.times.record_time("is_solved", time.time() - start_time)

        # expand
        self._expand_nodes(instances, [[node_curr] for node_curr in nodes_curr])

        # get q_vals
        start_time = time.time()
        q_vals_l: List[List[float]] = []
        nodes_next_l: List[List[Node]] = []
        for idx, instance in enumerate(instances):
            t_costs: List[float] = []
            nodes_next: List[Node] = []
            for t_cost, child in instance.curr_node.edge_dict.values():
                t_costs.append(t_cost)
                nodes_next.append(child)
            qvals: List[float] = [t_cost + node_next.heuristic for t_cost, node_next in zip(t_costs, nodes_next)]
            q_vals_l.append(qvals)
            nodes_next_l.append(nodes_next)
        self.times.record_time("q_vals", time.time() - start_time)

        # get next idxs
        start_time = time.time()
        temps: List[float] = [instance.temp for instance in instances]
        eps_l: List[float] = [instance.eps for instance in instances]
        next_idxs: List[int] = greedy_next_idx(q_vals_l, temps, eps_l)
        self.times.record_time("next_idx", time.time() - start_time)

        # update inst
        start_time = time.time()
        for instance, nodes_next, next_idx in zip(instances, nodes_next_l, next_idxs):
            instance.curr_node = nodes_next[next_idx]
            instance.itr += 1
        self.times.record_time("set_next", time.time() - start_time)

        if verbose:
            heuristics: List[float] = [node.heuristic for node in nodes_curr]
            path_costs: List[float] = [node.path_cost for node in nodes_curr]
            min_heur = float(np.min(heuristics))
            min_heur_pc = float(path_costs[np.argmin(heuristics)])
            max_heur = float(np.max(heuristics))
            max_heur_pc = float(path_costs[np.argmax(heuristics)])
            per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
            per_finished: float = 100.0 * float(np.mean([inst.finished() for inst in instances]))

            print(f"Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                  f"%.2E(%.2f)/%.2E(%.2f), %%has_soln: {per_has_soln}, "
                  f"%%finished: {per_finished}" % (self.itr, min_heur, min_heur_pc, max_heur, max_heur_pc))

        self.itr += 1

        return nodes_curr

    def __repr__(self) -> str:
        return f"{type(self).__name__}(temp={self.temp_default}, eps={self.eps_default})"


@pathfinding_factory.register_class("greedy_v")
class GreedyPolicy(GreedyPolicyActAny[ActsEnum], PathFindVExpandEnum[InstanceGrPolV]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_parser("greedy_v")
class GreedyVParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        assert len(args_str_l) == 2
        return {"temp": float(args_str_l[0]), "eps": float(args_str_l[1])}

    def help(self) -> str:
        return ("The temperature for Boltzmann exploration (0 for deterministic) and random action probability (eps). "
                "E.g. 'greedy_v.1.0_0.1'")
