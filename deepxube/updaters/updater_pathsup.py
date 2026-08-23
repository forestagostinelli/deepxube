from abc import ABC
from typing import List, Type, TypeVar, Any

from deepxube.base.domain import Domain, NodesLabelable, State, Goal
from deepxube.base.pathfinding import PFNsHV_T, PathFindSetHeurV, Node, Instance
from deepxube.base.pathfind_fns import PFNsHeurV, PFNsHeurVPolicy, UFNsHeurV, UFNsHeurVPolicy
from deepxube.base.updater import UpdateHasPolicy, UpdateHeurVPathFind, UpdatePathFindKeepGoal, UpdateRL, UFNsHV_T, InDataNode, UpdateParser
from deepxube.factories.updater_factory import updater_factory
from deepxube.utils.replay_buffer_utils import ReplayBufferVLab, ReplayVLab
from deepxube.utils.timing_utils import Times

import time


D_NL_T = TypeVar("D_NL_T", bound=NodesLabelable)


class UpdateHeurVPathSup(UpdateHeurVPathFind[D_NL_T, PFNsHV_T, PathFindSetHeurV, Instance, UFNsHV_T, ReplayBufferVLab, ReplayVLab],
                         UpdateRL[D_NL_T, PFNsHV_T, PathFindSetHeurV, Instance, UFNsHV_T], ABC):
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


class UpdateHeurVPathSupKeepGoalABC(UpdateHeurVPathSup[NodesLabelable, PFNsHV_T, UFNsHV_T],
                                    UpdatePathFindKeepGoal[NodesLabelable, PFNsHV_T, PathFindSetHeurV, Instance, UFNsHV_T, Node, InDataNode, ReplayBufferVLab,
                                    ReplayVLab], ABC):
    @staticmethod
    def domain_type() -> Type[NodesLabelable]:
        return NodesLabelable

    def _get_labels_no_rb(self, popped: List[Node], instances: List[Instance], times: Times) -> List[float]:
        return self._get_rb_data(popped, times)


@updater_factory.register_class("up_pathsup_v")
class UpdateHeurVPathSupKeepGoal(UpdateHeurVPathSupKeepGoalABC[PFNsHeurV, UFNsHeurV]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurV]:
        return PFNsHeurV

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurV]:
        return UFNsHeurV

    def _get_pathfind_functions(self) -> PFNsHeurV:
        return PFNsHeurV(self.get_heurv_fn())


@updater_factory.register_class("up_pathsup_v_p")
class UpdateHeurVRLKeepGoalPolicy(UpdateHeurVPathSupKeepGoalABC[PFNsHeurVPolicy, UFNsHeurVPolicy],
                                  UpdateHasPolicy[Domain, PFNsHeurVPolicy, PathFindSetHeurV, Instance, UFNsHeurVPolicy]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurVPolicy]:
        return PFNsHeurVPolicy

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurVPolicy]:
        return UFNsHeurVPolicy

    def _get_pathfind_functions(self) -> PFNsHeurVPolicy:
        return PFNsHeurVPolicy(self.get_heurv_fn(), self.get_policy_fn())


@updater_factory.register_parser("up_pathsup_v")
class UpdateVPathSupParser(UpdateParser):
    pass


@updater_factory.register_parser("up_pathsup_v_p")
class UpdateVPPathSupParser(UpdateParser):
    pass
