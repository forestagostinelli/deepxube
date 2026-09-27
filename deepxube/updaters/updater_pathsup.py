from abc import ABC
from typing import List, Type, Any

from deepxube.base.domain import Domain, NodesLabelable, State, Goal
from deepxube.base.pathfinding import PathFindSetHeurV, Node, Instance
from deepxube.base.pathfind_fns import UFNsHeurV
from deepxube.base.updater import UpdateHeurVPathFind, UpdatePathFindKeepGoal, UpdateRL, UFNsHV_T, InDataNode, UpdateParser
from deepxube.factories.updater_factory import updater_factory
from deepxube.utils.replay_buffer_utils import ReplayBufferVLab, ReplayVLab
from deepxube.utils.timing_utils import Times

import time


class UpdateHeurVPathSup(UpdateHeurVPathFind[Domain, PathFindSetHeurV, Instance, UFNsHV_T, ReplayBufferVLab, ReplayVLab],
                         UpdateRL[Domain, PathFindSetHeurV, Instance, UFNsHV_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurV]:
        return PathFindSetHeurV

    def _get_rb(self, max_size: int) -> ReplayBufferVLab:
        return ReplayBufferVLab(max_size)

    def _get_rb_data(self, popped: List[Node], times: Times) -> ReplayVLab:
        start_time = time.time()
        states: List[State] = [node.state for node in popped]
        goals: List[Goal] = [node.goal for node in popped]
        contexts: List[Any] = [node.context for node in popped]

        assert isinstance(self.domain, NodesLabelable)

        labels: List[float] = self.domain.label_nodes(states, goals, contexts)
        times.record_time("label", time.time() - start_time)

        return labels

    def _get_labels_rb(self, input_data: InDataNode, replay_data: ReplayVLab, times: Times) -> List[float]:
        return replay_data


class UpdateHeurVPathSupKeepGoalABC(UpdateHeurVPathSup[UFNsHV_T],
                                    UpdatePathFindKeepGoal[Domain, PathFindSetHeurV, Instance, UFNsHV_T, Node, InDataNode, ReplayBufferVLab, ReplayVLab], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _get_labels_no_rb(self, popped: List[Node], instances: List[Instance], times: Times) -> List[float]:
        return self._get_rb_data(popped, times)


@updater_factory.register_class("up_pathsup_v")
class UpdateHeurVPathSupKeepGoal(UpdateHeurVPathSupKeepGoalABC[UFNsHeurV]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurV]:
        return UFNsHeurV


@updater_factory.register_parser("up_pathsup_v")
class UpdateVPathSupParser(UpdateParser):
    pass
