from typing import List

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfinding import Node, EdgeQ, InstanceEdge, InstanceNode
from deepxube.base.updater import UpdatePolicy, UpdateHeurV, UpdateHeurQ, UpdateSup
from deepxube.utils.timing_utils import Times
from deepxube.pathfinding.supervised_v import PathFindVSup
from deepxube.pathfinding.supervised_q import PathFindQSup

import numpy as np


class UpdatePolicySup(UpdatePolicy[Domain, PathFindQSup, InstanceEdge], UpdateSup[Domain, PathFindQSup, InstanceEdge]):
    def _get_instance_data_norb(self, instances: List[InstanceEdge], times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = [edge.action for edge in edges_popped]

        inputs_np: List[NDArray] = self.get_policy_nnet_par().to_np_train(states, goals, actions)
        return inputs_np


class UpdateHeurVSup(UpdateHeurV[Domain, PathFindVSup], UpdateSup[Domain, PathFindVSup, InstanceNode]):
    def _get_instance_data_norb(self, instances: List[InstanceNode], times: Times) -> List[NDArray]:
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.get_nodes_popped())

        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]

        ctgs_backup: List[float] = [node.heuristic for node in nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals)
        return inputs_np + [np.array(ctgs_backup)]


class UpdateHeurQSup(UpdateHeurQ[Domain, PathFindQSup], UpdateSup[Domain, PathFindQSup, InstanceEdge]):
    def _get_instance_data_norb(self, instances: List[InstanceEdge], times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = [edge.action for edge in edges_popped]

        ctgs_backup: List[float] = [edge.q_val for edge in edges_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals, [[action] for action in actions])
        return inputs_np + [np.array(ctgs_backup)]
