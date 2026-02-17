from abc import ABC
from typing import Dict, Any, List, cast, Tuple

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, State, Goal
from deepxube.base.heuristic import HeurNNetParV, HeurFnV
from deepxube.base.pathfinding import PathFindVHeur, Node, InstanceV
from deepxube.base.updater import UpdateHER, UpdateHeurV, UpdateHeurRL, UpArgs, UpHeurArgs, D
from deepxube.updaters.utils.replay_buffer_utils import ReplayBufferV
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times

import time


def _pathfind_v_step(pathfind: PathFindVHeur) -> List[Node]:
    # take a step
    nodes_popped: List[Node] = pathfind.step()
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

    times.record_time("inst_popped", time.time() - start_time)

    return states, goals, is_solved_l


class UpdateHeurVRL(UpdateHeurV[D, PathFindVHeur], UpdateHeurRL[D, PathFindVHeur, InstanceV, HeurNNetParV, HeurFnV], ABC):
    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args
        self.rb: ReplayBufferV = ReplayBufferV(0)

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_heur_args.__repr__()}"

    def _step(self, pathfind: PathFindVHeur, times: Times) -> None:
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


class UpdateHeurVRLKeepGoal(UpdateHeurVRL[Domain]):
    def _step_sync_main(self, pathfind: PathFindVHeur, times: Times) -> List[NDArray]:
        # take a step
        nodes_popped: List[Node] = _pathfind_v_step(pathfind)

        # get sync states/goals/is_solved
        states_sync, goals_sync, is_solved_l_sync = _get_nodes_popped_data(nodes_popped, times)

        # add to replay buffer
        self._rb_add(states_sync, goals_sync, is_solved_l_sync, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(nodes_popped), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)

    def _get_instance_data_norb(self, instances: List[InstanceV], times: Times) -> List[NDArray]:
        # get popped node data
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.nodes_popped)
        states, goals, is_solved_l = _get_nodes_popped_data(nodes_popped, times)

        # get backup
        start_time = time.time()
        if self.up_heur_args.backup == 1:
            for node in nodes_popped:
                node.bellman_backup()
            if self.up_heur_args.ub_heur_solns:
                for node, is_solved in zip(nodes_popped, is_solved_l, strict=True):
                    if is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                instance.root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")

        ctgs_backup: List[float] = [node.backup_val for node in nodes_popped]
        times.record_time("backup", time.time() - start_time)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)

    def _get_instance_data_rb(self, instances: List[InstanceV], times: Times) -> List[NDArray]:
        # get popped node data
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.nodes_popped)
        states_popped, goals_popped, is_solved_l = _get_nodes_popped_data(nodes_popped, times)

        # add to replay buffer
        self._rb_add(states_popped, goals_popped, is_solved_l, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(nodes_popped), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)


class UpdateHeurVRLHER(UpdateHeurVRL[GoalSampleableFromState], UpdateHER[PathFindVHeur, InstanceV]):
    def _get_instance_data_rb(self, instances: List[InstanceV], times: Times) -> List[NDArray]:
        # get goals according to HER
        instances, goals_inst_her = self._get_her_goals(instances, times)

        # get states and goals
        states_her: List[State] = []
        goals_her: List[Goal] = []
        for instance, goal_her in zip(instances, goals_inst_her, strict=True):
            states_inst: List[State] = [node.state for node in instance.nodes_popped]
            states_her.extend(states_inst)
            goals_her.extend([goal_her] * len(states_inst))

        # is solved
        start_time = time.time()
        is_solved_her_l: List[bool] = self.domain.is_solved(states_her, goals_her)
        times.record_time("is_solved_her", time.time() - start_time)

        # add to replay buffer
        self._rb_add(states_her, goals_her, is_solved_her_l, times)

        # rb value iteration update
        states, goals, ctgs_backup = self._sample_rb_vi_target(len(states_her), times)

        return self._inputs_ctgs_to_np(states, goals, ctgs_backup, times)
