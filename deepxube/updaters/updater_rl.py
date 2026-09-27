from abc import ABC
from typing import Any, List, cast, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, State, Goal, Action
from deepxube.base.pathfinding import PathFindSetHeurV, PathFindSetHeurQ, PathFindSetPolicy, Node, EdgeQ, Instance
from deepxube.base.pathfind_fns import UFNsHeurV, UFNsHeurQ, UFNsPolicy
from deepxube.base.updater import (UpdateHeurVPathFind, UpdateHeurQPathFind, UpdatePolicyPathFind,
                                   UpdatePathFindHER, UpdatePathFindKeepGoal, UpdateRL, D, UpdateRLParser, UFNsHV_T, UFNsHQ_T, UFNsP_T, InDataNode, InDataEdge)
from deepxube.factories.updater_factory import updater_factory
from deepxube.utils.rl_utils import vi_backup
from deepxube.utils.replay_buffer_utils import ReplayBufferV, ReplayBufferQ, ReplayBufferP, ReplayV, ReplayQ, ReplayP
from deepxube.utils.timing_utils import Times

import time


class UpdateHeurVRL(UpdateHeurVPathFind[D, PathFindSetHeurV, Instance, UFNsHV_T, ReplayBufferV, ReplayV],
                    UpdateRL[D, PathFindSetHeurV, Instance, UFNsHV_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurV]:
        return PathFindSetHeurV

    def _get_rb(self, max_size: int) -> ReplayBufferV:
        return ReplayBufferV(max_size)

    def _get_rb_data(self, popped: List[Node], times: Times) -> ReplayV:
        start_time = time.time()
        is_solved_l: List[bool] = []
        for node in popped:
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)
        times.record_time("rb_data", time.time() - start_time)

        return is_solved_l

    def _get_labels_rb(self, input_data: InDataNode, replay_data: ReplayV, times: Times) -> List[float]:
        # expand states
        start_time = time.time()
        states: List[State] = input_data[0]
        goals: List[Goal] = input_data[1]
        contexts: List[Any] = input_data[2]

        states_exp, _, tcs_l = self.get_pathfind().expand_states(states, goals, contexts)
        times.record_time("vi_expand", time.time() - start_time, path=["replay"])

        # get vi backup
        start_time = time.time()
        ctgs_backup_l: List[float] = vi_backup(replay_data, goals, contexts, states_exp, tcs_l, self._get_targ_heurv_fn())
        times.record_time("vi_targ", time.time() - start_time, path=["replay"])

        return ctgs_backup_l


class UpdateHeurQRL(UpdateHeurQPathFind[D, PathFindSetHeurQ, Instance, UFNsHQ_T, ReplayBufferQ, ReplayQ],
                    UpdateRL[D, PathFindSetHeurQ, Instance, UFNsHQ_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurQ]:
        return PathFindSetHeurQ

    def _get_rb(self, max_size: int) -> ReplayBufferQ:
        return ReplayBufferQ(max_size)

    def _get_rb_data(self, popped: List[EdgeQ], times: Times) -> ReplayQ:
        start_time = time.time()
        nodes: List[Node] = [edge.node for edge in popped]

        is_solved_l: List[bool] = []
        tcs: List[float] = []
        states_next: List[State] = []
        for edge, node in zip(popped, nodes, strict=True):
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)
            tc, node_next = node.edge_dict[edge.action]
            tcs.append(tc)
            states_next.append(node_next.state)
        times.record_time("rb_data", time.time() - start_time)

        return is_solved_l, tcs, states_next

    def _get_labels_rb(self, input_data: InDataEdge, replay_data: ReplayQ, times: Times) -> List[float]:
        start_time = time.time()
        goals: List[Goal] = input_data[1]
        contexts: List[Any] = input_data[3]
        is_solved_l: List[bool] = replay_data[0]
        tcs: List[float] = replay_data[1]
        states_next: List[State] = replay_data[2]

        # min cost-to-go for next state
        actions_next: List[List[Action]] = self.get_pathfind().get_state_actions(states_next, goals, contexts)
        qvals_next_l: List[List[float]] = self._get_targ_heurq_fn()(states_next, goals, actions_next, contexts)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        times.record_time("q_learn_targ", time.time() - start_time, path=["replay"])

        return cast(List[float], ctg_backups.tolist())


class UpdatePolicyRL(UpdatePolicyPathFind[D, PathFindSetPolicy, Instance, UFNsP_T, ReplayBufferP, ReplayP],
                     UpdateRL[D, PathFindSetPolicy, Instance, UFNsP_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetPolicy]:
        return PathFindSetPolicy

    def _get_rb(self, max_size: int) -> ReplayBufferP:
        return ReplayBufferP(max_size)

    def _get_rb_data(self, popped: List[EdgeQ], times: Times) -> ReplayP:
        return None

    def _get_labels_rb(self, input_data: InDataEdge, replay_data: ReplayP, times: Times) -> List[float]:
        return []


class UpdateHeurVRLKeepGoalABC(UpdateHeurVRL[Domain, UFNsHV_T],
                               UpdatePathFindKeepGoal[Domain, PathFindSetHeurV, Instance, UFNsHV_T, Node, InDataNode, ReplayBufferV, ReplayV], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _get_labels_no_rb(self, popped: List[Node], instances: List[Instance], times: Times) -> List[float]:
        start_time = time.time()
        if not self.up_rl_args.lhbl:
            for node in popped:
                node.bellman_backup()
            if self.up_rl_args.ub_heur_solns:
                for node in popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        else:
            for instance in instances:
                instance.root_node.tree_backup()

        ctgs_backup: List[float] = [node.backup_val for node in popped]
        times.record_time("backup", time.time() - start_time)

        return ctgs_backup


class UpdateHeurQRLKeepGoalABC(UpdateHeurQRL[Domain, UFNsHQ_T],
                               UpdatePathFindKeepGoal[Domain, PathFindSetHeurQ, Instance, UFNsHQ_T, EdgeQ, InDataEdge, ReplayBufferQ, ReplayQ], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _get_labels_no_rb(self, popped: List[EdgeQ], instances: List[Instance], times: Times) -> List[float]:
        start_time = time.time()
        if not self.up_rl_args.lhbl:
            if self.up_rl_args.ub_heur_solns:
                for edge in popped:
                    assert edge.node.is_solved is not None
                    if edge.node.is_solved:
                        edge.node.upper_bound_parent_path(0.0)
        else:
            for instance in instances:
                instance.root_node.tree_backup()

        ctgs_backup: List[float] = []
        for edge in popped:
            node: Node = edge.node
            ctg_backup = node.backup_act(edge.action)
            node.backup_val = ctg_backup
            ctgs_backup.append(ctg_backup)

        times.record_time("backup", time.time() - start_time)

        return ctgs_backup


class UpdatePolicyRLKeepGoalABC(UpdatePolicyRL[Domain, UFNsP_T],
                                UpdatePathFindKeepGoal[Domain, PathFindSetPolicy, Instance, UFNsP_T, EdgeQ, InDataEdge, ReplayBufferP, ReplayP], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _get_labels_no_rb(self, popped: List[EdgeQ], instances: List[Instance], times: Times) -> List[float]:
        return []


class UpdateHeurVRLHERABC(UpdateHeurVRL[GoalSampleableFromState, UFNsHV_T],
                          UpdatePathFindHER[GoalSampleableFromState, PathFindSetHeurV, Instance, UFNsHV_T, Node, InDataNode, ReplayBufferV, ReplayV],
                          ABC):
    @staticmethod
    def domain_type() -> Type[GoalSampleableFromState]:
        return GoalSampleableFromState

    def _get_her_data(self, instances: List[Instance], goals_inst_her: List[Goal], times: Times) -> Tuple[InDataNode, ReplayV, int]:
        states, goals, contexts, is_solved_l = self._get_her_node_data(instances, goals_inst_her, times)
        return (states, goals, contexts), is_solved_l, len(states)


class UpdateHeurQRLHERABC(UpdateHeurQRL[GoalSampleableFromState, UFNsHQ_T],
                          UpdatePathFindHER[GoalSampleableFromState, PathFindSetHeurQ, Instance, UFNsHQ_T, EdgeQ, InDataEdge, ReplayBufferQ, ReplayQ],
                          ABC):
    @staticmethod
    def domain_type() -> Type[GoalSampleableFromState]:
        return GoalSampleableFromState

    def _get_her_data(self, instances: List[Instance], goals_inst_her: List[Goal], times: Times) -> Tuple[InDataEdge, ReplayQ, int]:
        states, goals, actions, contexts, is_solved_l, tcs, states_next = self._get_her_edge_data(instances, goals_inst_her, times)
        return (states, goals, actions, contexts), (is_solved_l, tcs, states_next), len(states)


class UpdatePolicyRLHERABC(UpdatePolicyRL[GoalSampleableFromState, UFNsP_T],
                           UpdatePathFindHER[GoalSampleableFromState, PathFindSetPolicy, Instance, UFNsP_T, EdgeQ, InDataEdge, ReplayBufferP, ReplayP],
                           ABC):
    @staticmethod
    def domain_type() -> Type[GoalSampleableFromState]:
        return GoalSampleableFromState

    def _get_her_data(self, instances: List[Instance], goals_inst_her: List[Goal], times: Times) -> Tuple[InDataEdge, ReplayP, int]:
        states, goals, actions, contexts, _, _, _ = self._get_her_edge_data(instances, goals_inst_her, times)
        return (states, goals, actions, contexts), None, len(states)


@updater_factory.register_class("up_rl_v")
class UpdateHeurVRLKeepGoal(UpdateHeurVRLKeepGoalABC[UFNsHeurV]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurV]:
        return UFNsHeurV


@updater_factory.register_class("up_her_v")
class UpdateHeurVRLHER(UpdateHeurVRLHERABC[UFNsHeurV]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurV]:
        return UFNsHeurV


@updater_factory.register_class("up_rl_q")
class UpdateHeurQRLKeepGoal(UpdateHeurQRLKeepGoalABC[UFNsHeurQ]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQ]:
        return UFNsHeurQ


@updater_factory.register_class("up_her_q")
class UpdateHeurQRLHER(UpdateHeurQRLHERABC[UFNsHeurQ]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQ]:
        return UFNsHeurQ


@updater_factory.register_class("up_rl_p")
class UpdatePolicyRLKeepGoal(UpdatePolicyRLKeepGoalABC[UFNsPolicy]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsPolicy]:
        return UFNsPolicy


@updater_factory.register_class("up_her_p")
class UpdatePolicyRLHER(UpdatePolicyRLHERABC[UFNsPolicy]):
    @staticmethod
    def updater_functions_type() -> Type[UFNsPolicy]:
        return UFNsPolicy


@updater_factory.register_parser("up_rl_v")
class UpdateVRL(UpdateRLParser):
    pass


@updater_factory.register_parser("up_her_v")
class UpdateVRLHER(UpdateRLParser):
    pass


@updater_factory.register_parser("up_rl_q")
class UpdateQRL(UpdateRLParser):
    pass


@updater_factory.register_parser("up_her_q")
class UpdateQRLHER(UpdateRLParser):
    pass


@updater_factory.register_parser("up_rl_p")
class UpdatePRL(UpdateRLParser):
    pass


@updater_factory.register_parser("up_her_p")
class UpdatePRLHER(UpdateRLParser):
    pass
