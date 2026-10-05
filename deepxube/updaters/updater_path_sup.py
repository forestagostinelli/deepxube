from abc import ABC
from typing import List, Type, Any

from deepxube.base.domain import NodesLabelable, EdgesLabelable, State, Action, Goal
from deepxube.base.pathfinding import PathFindSetHeurV, PathFindSetHeurQ, Node, EdgeQ, Instance
from deepxube.base.pathfind_fns import UFNsHeurV, UFNsHeurQ
from deepxube.base.updater import (UpdateHeurVPathFind, UpdateHeurQPathFind, UpdatePathFindKeepGoal, UpdateLabel, UFNsHV_T, UFNsHQ_T, InDataNode, InDataEdge,
                                   UpdateParser)
from deepxube.factories.updater_factory import updater_factory
from deepxube.utils.replay_buffer_utils import ReplayBufferVLab, ReplayVLab, ReplayBufferQLab, ReplayQLab
from deepxube.utils.timing_utils import Times

import time


class UpdateHeurVPathSup(UpdateHeurVPathFind[NodesLabelable, PathFindSetHeurV, Instance, UFNsHV_T, ReplayBufferVLab, ReplayVLab],
                         UpdateLabel[NodesLabelable, PathFindSetHeurV, Instance, UFNsHV_T], ABC):
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

        labels: List[float] = self.domain.label_nodes(states, goals, contexts)
        times.record_time("label", time.time() - start_time)

        return labels

    def _get_labels_rb(self, input_data: InDataNode, replay_data: ReplayVLab, times: Times) -> List[float]:
        return replay_data


class UpdateHeurQPathSup(UpdateHeurQPathFind[EdgesLabelable, PathFindSetHeurQ, Instance, UFNsHQ_T, ReplayBufferQLab, ReplayQLab],
                         UpdateLabel[EdgesLabelable, PathFindSetHeurQ, Instance, UFNsHQ_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurQ]:
        return PathFindSetHeurQ

    def _get_rb(self, max_size: int) -> ReplayBufferQLab:
        return ReplayBufferQLab(max_size)

    def _get_rb_data(self, popped: List[EdgeQ], times: Times) -> ReplayQLab:
        start_time = time.time()
        nodes: List[Node] = [edge.node for edge in popped]

        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions: List[Action] = [edge.action for edge in popped]
        contexts: List[Any] = [node.context for node in nodes]

        labels: List[float] = self.domain.label_edges(states, goals, actions, contexts)
        times.record_time("label", time.time() - start_time)

        return labels

    def _get_labels_rb(self, input_data: InDataEdge, replay_data: ReplayQLab, times: Times) -> List[float]:
        return replay_data


class UpdateHeurVPathSupKeepGoalABC(UpdateHeurVPathSup[UFNsHV_T],
                                    UpdatePathFindKeepGoal[NodesLabelable, PathFindSetHeurV, Instance, UFNsHV_T, Node, InDataNode, ReplayBufferVLab,
                                    ReplayVLab], ABC):
    @staticmethod
    def domain_type() -> Type[NodesLabelable]:
        return NodesLabelable

    def _get_labels_no_rb(self, popped: List[Node], instances: List[Instance], times: Times) -> List[float]:
        return self._get_rb_data(popped, times)


class UpdateHeurQPathSupKeepGoalABC(UpdateHeurQPathSup[UFNsHQ_T],
                                    UpdatePathFindKeepGoal[EdgesLabelable, PathFindSetHeurQ, Instance, UFNsHQ_T, EdgeQ, InDataEdge, ReplayBufferQLab,
                                    ReplayQLab], ABC):
    @staticmethod
    def domain_type() -> Type[EdgesLabelable]:
        return EdgesLabelable

    def _get_labels_no_rb(self, popped: List[EdgeQ], instances: List[Instance], times: Times) -> List[float]:
        return self._get_rb_data(popped, times)


@updater_factory.register_class("path_sup_v")
class UpdateHeurVPathSupKeepGoal(UpdateHeurVPathSupKeepGoalABC[UFNsHeurV]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurV]:
        return UFNsHeurV


@updater_factory.register_class("path_sup_q")
class UpdateHeurQPathSupKeepGoal(UpdateHeurQPathSupKeepGoalABC[UFNsHeurQ]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQ]:
        return UFNsHeurQ


@updater_factory.register_parser("path_sup_v")
class UpdateVPathSupParser(UpdateParser):
    pass


@updater_factory.register_parser("path_sup_q")
class UpdateQPathSupParser(UpdateParser):
    pass
