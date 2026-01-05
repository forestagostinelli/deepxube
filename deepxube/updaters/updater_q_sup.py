from typing import List

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfinding import EdgeQ, Instance
from deepxube.base.updater import UpdateHeurQ, UpdateHeurSup
from deepxube.utils.timing_utils import Times
from deepxube.base.heuristic import HeurNNetParQ, HeurFnQ
from deepxube.pathfinding.supervised_q import PathFindQSup

import numpy as np


class UpdateHeurQSup(UpdateHeurQ[Domain, PathFindQSup], UpdateHeurSup[Domain, PathFindQSup, HeurNNetParQ, HeurFnQ]):
    def _step(self, pathfind: PathFindQSup, times: Times) -> List[NDArray]:
        edges_popped: List[EdgeQ] = pathfind.step()
        assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"
        if not self.up_args.sync_main:
            self.edges_popped.extend(edges_popped)
            return []
        else:
            return self._get_inputs_ctgs(edges_popped)

    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        inputs_ctgs: List[NDArray] = self._get_inputs_ctgs(self.edges_popped)
        self.edges_popped = []
        return inputs_ctgs

    def _get_inputs_ctgs(self, edges_popped: List[EdgeQ]) -> List[NDArray]:
        states: List[State] = [edge.node.state for edge in edges_popped]
        goals: List[Goal] = [edge.node.goal for edge in edges_popped]
        actions: List[Action] = []
        for edge in edges_popped:
            assert edge.action is not None
            actions.append(edge.action)
        ctgs_backup: List[float] = [edge.q_val for edge in edges_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals, [[action] for action in actions])
        return inputs_np + [np.array(ctgs_backup)]
