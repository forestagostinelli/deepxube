from typing import List

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfinding import EdgeQ, InstanceEdge
from deepxube.base.updater import UpdateHeurQ, UpdateHeurSup
from deepxube.utils.timing_utils import Times
from deepxube.base.heuristic import HeurNNetParQ, HeurFnQ
from deepxube.pathfinding.supervised_q import PathFindQSup

import numpy as np


class UpdateHeurQSup(UpdateHeurQ[Domain, PathFindQSup], UpdateHeurSup[Domain, PathFindQSup, InstanceEdge, HeurNNetParQ, HeurFnQ]):
    def _step(self, pathfind: PathFindQSup, times: Times) -> None:
        edges_popped: List[EdgeQ] = pathfind.step()[1]
        assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"

    def _get_instance_data_norb(self, instances: List[InstanceEdge], times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        inputs_ctgs: List[NDArray] = self._get_inputs_ctgs(edges_popped)
        return inputs_ctgs

    def _get_inputs_ctgs(self, edges_popped: List[EdgeQ]) -> List[NDArray]:
        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = [edge.action for edge in edges_popped]

        ctgs_backup: List[float] = [edge.q_val for edge in edges_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals, [[action] for action in actions])
        return inputs_np + [np.array(ctgs_backup)]
