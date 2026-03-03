from abc import ABC
from typing import List, cast, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, State, Goal
from deepxube.base.pathfinding import FNsHV, FNsHeurV, FNsHeurVPolicy, PathFindSetHeurV, Node, InstanceNode
from deepxube.base.updater import UpdateHER, UpdateHasPolicy, UpdateHeurV, UpdateRL, UpArgs, D
from deepxube.factories.updater_factory import updater_factory
from deepxube.updaters.utils.replay_buffer_utils import ReplayBufferV
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times

import time


def _pathfind_v_step(pathfind: PathFindSetHeurV) -> List[Node]:
    # take a step
    nodes_popped: List[Node] = pathfind.step()[0]
    assert len(nodes_popped) == len(pathfind.instances), f"Values were {len(nodes_popped)} and {len(pathfind.instances)}"

    return nodes_popped


def _get_nodes_popped_data(nodes_popped: List[Node], times: Times) -> Tuple[List[State], List[Goal], List[bool]]:
    start_time = time.time()

    states: List[State] = [node.state for node in nodes_popped]
    goals: List[Goal] = [node.goal for node in nodes_popped]
    is_solved_l: List[bool] = []
    for node in nodes_popped:
        assert node.is_solved is not None
        is_solved_l.append(node.is_solved)

    times.record_time("nodes_popped", time.time() - start_time)

    return states, goals, is_solved_l


class UpdateHeurVRL(UpdateHeurV[D, FNsHV, PathFindSetHeurV], UpdateRL[D, FNsHV, PathFindSetHeurV, InstanceNode], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurV]:
        return PathFindSetHeurV

    def __init__(self, domain: D, pathfind_arg: str, up_args: UpArgs):
        super().__init__(domain, pathfind_arg, up_args)
        self.rb: ReplayBufferV = ReplayBufferV(0)

    def _step(self, pathfind: PathFindSetHeurV, times: Times) -> None:
        _pathfind_v_step(pathfind)

    def _value_iteration_target(self, goals: List[Goal], is_solved_l: List[bool], tcs_l: List[List[float]], states_exp: List[List[State]],
                                times: Times) -> List[float]:
        start_time = time.time()
        # get cost-to-go of expanded states
        states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
        goals_flat: List[Goal] = []
        for goal, state_exp in zip(goals, states_exp, strict=True):
            goals_flat.extend([goal] * len(state_exp))
        ctg_next: List[float] = self._get_targ_heur_fn()(states_exp_flat, goals_flat)

        # backup cost-to-go
        ctg_next_p_tc = np.concatenate(tcs_l, axis=0) + np.array(ctg_next)
        ctg_next_p_tc_l = np.split(ctg_next_p_tc, split_idxs)

        ctgs_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved_l)
        ctgs_backup_l: List[float] = cast(List[float], ctgs_backup.tolist())

        times.record_time("vi_targ", time.time() - start_time)

        return ctgs_backup_l

    def _inputs_ctgs_to_np(self, states: List[State], goals: List[Goal], ctgs_backup: List[float], times: Times) -> List[NDArray]:
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals)
        data_np: List[NDArray] = inputs_np + [np.array(ctgs_backup)]
        times.record_time("to_np", time.time() - start_time)

        return data_np

    def _init_replay_buffer(self, max_size: int) -> None:
        self.rb = ReplayBufferV(max_size)

    def _rb_add(self, states: List[State], goals: List[Goal], is_solved_l: List[bool], times: Times) -> None:
        start_time = time.time()
        self.rb.add(list(zip(states, goals, is_solved_l, strict=True)))
        times.record_time("rb_add", time.time() - start_time)

    def _sample_rb_vi_target(self, num: int, times: Times) -> Tuple[List[State], List[Goal], List[float]]:
        # sample from replay buffer
        start_time = time.time()
        states, goals, is_solved_l = self.rb.sample(num)
        times.record_time("rb_samp", time.time() - start_time)

        # expand states
        start_time = time.time()
        states_exp, _, tcs_l = self.get_pathfind().expand_states(states, goals)
        times.record_time("vi_expand", time.time() - start_time)

        # value iteration update
        ctgs_backup: List[float] = self._value_iteration_target(goals, is_solved_l, tcs_l, states_exp, times)

        return states, goals, ctgs_backup


class UpdateHeurVRLKeepGoalABC(UpdateHeurVRL[Domain, FNsHV], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _step_sync_main(self, pathfind: PathFindSetHeurV, times: Times) -> List[NDArray]:
        # take a step
        nodes_popped: List[Node] = _pathfind_v_step(pathfind)

        # get sync states/goals/is_solved
        states_sync, goals_sync, is_solved_l_sync = _get_nodes_popped_data(nodes_popped, times)

        # add to replay buffer
        self._rb_add(states_sync, goals_sync, is_solved_l_sync, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(nodes_popped), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)

    def _get_instance_data_norb(self, instances: List[InstanceNode], times: Times) -> List[NDArray]:
        # get popped node data
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.get_nodes_popped())

        # get backup
        start_time = time.time()
        if self.up_args.backup == 1:
            for node in nodes_popped:
                node.bellman_backup()
            if self.up_args.ub_heur_solns:
                for node in nodes_popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_args.backup == -1:
            for instance in instances:
                instance.root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_args.backup}")

        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        ctgs_backup: List[float] = [node.backup_val for node in nodes_popped]

        times.record_time("get_tr_data", time.time() - start_time)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)

    def _get_instance_data_rb(self, instances: List[InstanceNode], times: Times) -> List[NDArray]:
        # get popped node data
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.get_nodes_popped())
        states_popped, goals_popped, is_solved_l = _get_nodes_popped_data(nodes_popped, times)

        # add to replay buffer
        self._rb_add(states_popped, goals_popped, is_solved_l, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(nodes_popped), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)


class UpdateHeurVRLHERABC(UpdateHeurVRL[GoalSampleableFromState, FNsHV], UpdateHER[FNsHV, PathFindSetHeurV, InstanceNode], ABC):
    @staticmethod
    def domain_type() -> Type[GoalSampleableFromState]:
        return GoalSampleableFromState

    def _get_instance_data_rb(self, instances: List[InstanceNode], times: Times) -> List[NDArray]:
        # get goals according to HER
        instances, goals_inst_her = self._get_her_goals(instances, times)

        # get states and goals
        start_time = time.time()
        states_her: List[State] = []
        goals_her: List[Goal] = []
        for instance, goal_her in zip(instances, goals_inst_her, strict=True):
            states_inst: List[State] = [node.state for node in instance.get_nodes_popped()]
            states_her.extend(states_inst)
            goals_her.extend([goal_her] * len(states_inst))

        times.record_time("data_her", time.time() - start_time)

        # is solved
        start_time = time.time()
        is_solved_l_her: List[bool] = self.domain.is_solved(states_her, goals_her)
        times.record_time("is_solved_her", time.time() - start_time)

        # add to replay buffer
        self._rb_add(states_her, goals_her, is_solved_l_her, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(states_her), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)


@updater_factory.register_class("update_v_rl")
class UpdateHeurVRLKeepGoal(UpdateHeurVRLKeepGoalABC[FNsHeurV]):
    @staticmethod
    def functions_type() -> Type[FNsHeurV]:
        return FNsHeurV

    def _get_pathfind_functions(self) -> FNsHeurV:
        return FNsHeurV(self.get_heur_fn())


@updater_factory.register_class("update_v_rl_her")
class UpdateHeurVRLHER(UpdateHeurVRLHERABC[FNsHeurV]):
    @staticmethod
    def functions_type() -> Type[FNsHeurV]:
        return FNsHeurV

    def _get_pathfind_functions(self) -> FNsHeurV:
        return FNsHeurV(self.get_heur_fn())


@updater_factory.register_class("update_v_p_rl")
class UpdateHeurVRLKeepGoalPolicy(UpdateHeurVRLKeepGoalABC[FNsHeurVPolicy], UpdateHasPolicy[Domain, FNsHeurVPolicy, PathFindSetHeurV, InstanceNode]):
    @staticmethod
    def functions_type() -> Type[FNsHeurVPolicy]:
        return FNsHeurVPolicy

    def _get_pathfind_functions(self) -> FNsHeurVPolicy:
        return FNsHeurVPolicy(self.get_heur_fn(), self.get_policy_fn())


@updater_factory.register_class("update_v_p_rl_her")
class UpdateHeurVRLHERPolicy(UpdateHeurVRLHERABC[FNsHeurVPolicy], UpdateHasPolicy[Domain, FNsHeurVPolicy, PathFindSetHeurV, InstanceNode]):
    @staticmethod
    def functions_type() -> Type[FNsHeurVPolicy]:
        return FNsHeurVPolicy

    def _get_pathfind_functions(self) -> FNsHeurVPolicy:
        return FNsHeurVPolicy(self.get_heur_fn(), self.get_policy_fn())
