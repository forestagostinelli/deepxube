from typing import List, Type

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfinding import Node, EdgeQ, InstanceEdge, InstanceNode
from deepxube.base.updater import UpdatePolicy, UpdateHeurV, UpdateHeurQ, UpdateSup, UpdateParser
from deepxube.factories.updater_factory import updater_factory
from deepxube.utils.timing_utils import Times
from deepxube.pathfinding.supervised import PathFindNodeSup, PathFindEdgeSup, PathFindEdgeSamp

import numpy as np


@updater_factory.register_class("up_sup_v")
class UpdateHeurVSup(UpdateHeurV[Domain, PathFindNodeSup], UpdateSup[Domain, PathFindNodeSup, InstanceNode]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_type() -> Type[PathFindNodeSup]:
        return PathFindNodeSup

    def _get_instance_data_norb(self, instances: List[InstanceNode], times: Times) -> List[NDArray]:
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.get_nodes_popped())

        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]

        ctgs_backup: List[float] = [node.heuristic for node in nodes_popped]
        inputs_np: List[NDArray] = self.get_heurv_nnet_par().process_inputs(states, goals).inputs_nnet
        return inputs_np + [np.array(ctgs_backup)]


@updater_factory.register_class("up_sup_q")
class UpdateHeurQSup(UpdateHeurQ[Domain, PathFindEdgeSup], UpdateSup[Domain, PathFindEdgeSup, InstanceEdge]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_type() -> Type[PathFindEdgeSup]:
        return PathFindEdgeSup

    def _get_instance_data_norb(self, instances: List[InstanceEdge], times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = [edge.action for edge in edges_popped]

        ctgs_backup: List[float] = [edge.q_val for edge in edges_popped]
        inputs_np: List[NDArray] = self.get_heurq_nnet_par().process_inputs(states, goals, [[action] for action in actions]).inputs_nnet
        return inputs_np + [np.array(ctgs_backup)]


@updater_factory.register_class("up_sup_p")
class UpdatePolicySup(UpdatePolicy[Domain, PathFindEdgeSamp, InstanceEdge], UpdateSup[Domain, PathFindEdgeSamp, InstanceEdge]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def pathfind_type() -> Type[PathFindEdgeSamp]:
        return PathFindEdgeSamp

    def _get_instance_data_norb(self, instances: List[InstanceEdge], times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = [edge.action for edge in edges_popped]

        inputs_np: List[NDArray] = self.get_policy_nnet_par().to_np_train(states, goals, actions)
        return inputs_np


@updater_factory.register_parser("up_sup_v")
class UpdateVSup(UpdateParser):
    pass


@updater_factory.register_parser("up_sup_q")
class UpdateQSup(UpdateParser):
    pass


@updater_factory.register_parser("up_sup_p")
class UpdatePSup(UpdateParser):
    pass
