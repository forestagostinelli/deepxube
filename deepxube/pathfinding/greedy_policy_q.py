from typing import List, Any, Type, Optional, TypeVar, Dict
from abc import ABC
from deepxube.base.factory import Parser
from deepxube.base.domain import Domain, ActsEnum, State, Action, Goal
from deepxube.base.pathfinding import InstanceQ, Node, PathFindQHeur, EdgeQ, PathFindQExpandEnum
from deepxube.pathfinding.utils.search import greedy_next_idx
from deepxube.factories.pathfinding_factory import pathfinding_factory
import time


class InstanceGrPolQ(InstanceQ):
    def __init__(self, root_node: Node, temp: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.curr_node: Node = self.root_node
        self.temp: float = temp
        assert (eps >= 0.0) and (eps <= 1.0)
        self.eps: float = eps

    def record_goal(self, node: Node) -> None:
        assert node.is_solved is not None
        if node.is_solved:
            if (self.goal_node is None) or (self.goal_node.path_cost > node.path_cost):
                self.goal_node = node

    def finished(self) -> bool:
        return self.has_soln()


D = TypeVar('D', bound=Domain)


class GreedyPolicyQActsAny(PathFindQHeur[D, InstanceGrPolQ], ABC):
    def __init__(self, domain: D, temp: float = 1.0, eps: float = 0.0):
        super().__init__(domain)
        self.temp_default: float = temp
        self.eps_default: float = eps

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None, compute_root_heur: bool = True,
                       temp: Optional[float] = None, eps: Optional[float] = None) -> List[InstanceGrPolQ]:
        """ Always computes the root heurisitc always (regardless of argument)

        """
        nodes_root: List[Node] = self._create_root_nodes(states, goals, compute_root_heur=True)
        temp_inst: float = temp if temp is not None else self.temp_default
        eps_inst: float = eps if eps is not None else self.eps_default
        if inst_infos is None:
            inst_infos = [None for _ in states]
        return [InstanceGrPolQ(node_root, temp_inst, eps_inst, inst_info) for node_root, inst_info in zip(nodes_root, inst_infos, strict=True)]

    def step(self, verbose: bool = False) -> List[EdgeQ]:
        # get unsolved instances
        instances: List[InstanceGrPolQ] = [instance for instance in self.instances if not instance.finished()]
        if len(instances) == 0:
            self.itr += 1  # TODO make more elegant
            return []

        # get curr nodes
        nodes: List[Node] = [inst.curr_node for inst in instances]

        # is solved
        start_time = time.time()
        self.set_is_solved(nodes)
        for instance, node in zip(instances, nodes):
            instance.record_goal(node)

        self.times.record_time("is_solved", time.time() - start_time)

        # get edge
        start_time = time.time()
        temps: List[float] = [instance.temp for instance in instances]
        eps_l: List[float] = [instance.eps for instance in instances]
        actions_l: List[List[Action]] = []
        q_vals_l: List[List[float]] = []
        for node in nodes:
            assert node.q_values is not None
            actions_l.append(node.q_values[0])
            q_vals_l.append(node.q_values[1])

        next_idxs: List[int] = greedy_next_idx(q_vals_l, temps, eps_l)

        edges: List[EdgeQ] = [EdgeQ(node, actions[next_idx], q_vals[next_idx]) for node, actions, q_vals, next_idx in
                              zip(nodes, actions_l, q_vals_l, next_idxs)]

        self.times.record_time("next_idx", time.time() - start_time)

        # next nodes
        nodes_next_l: List[List[Node]] = self.get_next_nodes(instances, [[edge] for edge in edges])
        nodes_next: List[Node] = [nodes_next_i[0] for nodes_next_i in nodes_next_l]

        # update inst
        start_time = time.time()
        for instance, node in zip(instances, nodes_next):
            instance.curr_node = node
            instance.itr += 1
        self.times.record_time("set_next", time.time() - start_time)

        self.itr += 1

        return edges

    def __repr__(self) -> str:
        return f"{type(self).__name__}(temp={self.temp_default}, eps={self.eps_default})"


@pathfinding_factory.register_class("greedy_q")
class GreedyPolicyQ(GreedyPolicyQActsAny[ActsEnum], PathFindQExpandEnum[InstanceGrPolQ]):
    @staticmethod
    def domain_type() -> Type[ActsEnum]:
        return ActsEnum


@pathfinding_factory.register_parser("greedy_q")
class GreedyQParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        assert len(args_str_l) == 2
        return {"temp": float(args_str_l[0]), "eps": float(args_str_l[1])}

    def help(self) -> str:
        return ("The temperature for Boltzmann exploration (0 for deterministic) and random action probability (eps). "
                "E.g. 'greedy_q.1.0_0.1'")
