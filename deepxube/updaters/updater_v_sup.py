from typing import List

from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal

from deepxube.base.pathfinding import Node, InstanceV
from deepxube.base.updater import UpdateHeurV, UpdateHeurSup
from deepxube.utils.timing_utils import Times
from deepxube.base.heuristic import HeurNNetParV, HeurFnV
from deepxube.pathfinding.supervised_v import PathFindVSup

import numpy as np


class UpdateHeurVSup(UpdateHeurV[Domain, PathFindVSup], UpdateHeurSup[Domain, PathFindVSup, InstanceV, HeurNNetParV, HeurFnV]):
    def _step(self, pathfind: PathFindVSup, times: Times) -> None:
        nodes_popped: List[Node] = pathfind.step()
        assert len(nodes_popped) == len(pathfind.instances), f"Values were {len(nodes_popped)} and {len(pathfind.instances)}"

    def _step_sync_main(self, pathfind: PathFindVSup, times: Times) -> List[NDArray]:
        raise NotImplementedError

    def _get_instance_data(self, instances: List[InstanceV], times: Times) -> List[NDArray]:
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.nodes_popped)

        inputs_ctgs: List[NDArray] = self._get_inputs_ctgs(nodes_popped)
        return inputs_ctgs

    def _get_inputs_ctgs(self, nodes_popped: List[Node]) -> List[NDArray]:
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        ctgs_backup: List[float] = [node.heuristic for node in nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals)
        return inputs_np + [np.array(ctgs_backup)]
