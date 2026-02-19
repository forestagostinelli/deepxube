from abc import ABC
from typing import Dict, Any, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, Action, State, Goal
from deepxube.base.heuristic import HeurNNetParQ, HeurFnQ
from deepxube.base.pathfinding import PathFindQHeur, EdgeQ, InstanceQ, Node
from deepxube.base.updater import UpdateHER, UpdateHeurQ, UpdateHeurRL, D, UpArgs, UpHeurArgs
from deepxube.updaters.utils.replay_buffer_utils import ReplayBufferQ
from deepxube.utils.timing_utils import Times

import time


def _pathfind_q_step(pathfind: PathFindQHeur) -> List[EdgeQ]:
    edges_popped: List[EdgeQ] = pathfind.step()
    assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"

    return edges_popped


def _get_edge_popped_data(edges_popped: List[EdgeQ],
                          times: Times) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:

    start_time = time.time()
    nodes: List[Node] = [edge.node for edge in edges_popped]
    states: List[State] = [node.state for node in nodes]
    goals: List[Goal] = [node.goal for node in nodes]
    actions: List[Action] = [edge.action for edge in edges_popped]

    is_solved_l: List[bool] = []
    tcs: List[float] = []
    states_next: List[State] = []
    for edge, node in zip(edges_popped, nodes, strict=True):
        assert node.is_solved is not None
        is_solved_l.append(node.is_solved)
        tc, node_next = node.edge_dict[edge.action]
        tcs.append(tc)
        states_next.append(node_next.state)
    times.record_time("edge_data", time.time() - start_time)

    return states, goals, is_solved_l, actions, tcs, states_next


class UpdateHeurQRL(UpdateHeurQ[D, PathFindQHeur], UpdateHeurRL[D, PathFindQHeur, InstanceQ, HeurNNetParQ, HeurFnQ], ABC):
    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args
        self.rb: ReplayBufferQ = ReplayBufferQ(0)

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_heur_args.__repr__()}"

    def _step(self, pathfind: PathFindQHeur, times: Times) -> None:
        _pathfind_q_step(pathfind)

    def _get_qvals_targ(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        actions_next: List[List[Action]] = self.get_pathfind().get_state_actions(states, goals)
        qvals: List[List[float]] = self._get_targ_heur_fn()(states, goals, actions_next)

        return qvals

    def _q_learning_target(self, goals: List[Goal], is_solved_l: List[bool], tcs: List[float], states_next: List[State], times: Times) -> List[float]:
        start_time = time.time()
        # min cost-to-go for next state
        qvals_next_l: List[List[float]] = self._get_qvals_targ(states_next, goals)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        times.record_time("qlearn_targ", time.time() - start_time)

        return cast(List[float], ctg_backups.tolist())

    def _inputs_ctgs_to_np(self, states: List[State], goals: List[Goal], actions: List[Action], ctgs_backup: List[float], times: Times) -> List[NDArray]:
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]

    def _init_replay_buffer(self, max_size: int) -> None:
        self.rb = ReplayBufferQ(max_size)

    def _rb_add(self, states: List[State], goals: List[Goal], is_solved_l: List[bool], actions: List[Action], tcs: List[float], states_next: List[State],
                times: Times) -> None:
        start_time = time.time()
        self.rb.add(list(zip(states, goals, is_solved_l, actions, tcs, states_next, strict=True)))
        times.record_time("rb_add", time.time() - start_time)

    def _sample_rb_qlearn_target(self, num: int, times: Times) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        # sample from replay buffer
        start_time = time.time()
        states, goals, is_solved_l, actions, tcs, states_next = self.rb.sample(num)
        times.record_time("rb_samp", time.time() - start_time)

        # value iteration update
        ctgs_backup: List[float] = self._q_learning_target(goals, is_solved_l, tcs, states_next, times)

        return states, goals, actions, ctgs_backup


class UpdateHeurQRLKeepGoal(UpdateHeurQRL[Domain]):
    def _step_sync_main(self, pathfind: PathFindQHeur, times: Times) -> List[NDArray]:
        # take a step
        edges_popped: List[EdgeQ] = _pathfind_q_step(pathfind)

        # get sync states/goals/is_solved
        states_sync, goals_sync, is_solved_l_sync, actions_sync, tcs_sync, states_next_sync = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_sync, goals_sync, is_solved_l_sync, actions_sync, tcs_sync, states_next_sync, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)

    def _get_instance_data_norb(self, instances: List[InstanceQ], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.edges_popped)

        # backup
        start_time = time.time()
        if self.up_heur_args.backup == 1:
            if self.up_heur_args.ub_heur_solns:
                for edge in edges_popped:
                    assert edge.node.is_solved is not None
                    if edge.node.is_solved:
                        edge.node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                instance.root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        nodes: List[Node] = [edge.node for edge in edges_popped]
        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions: List[Action] = [edge.action for edge in edges_popped]

        ctgs_backup: List[float] = []
        for edge, node in zip(edges_popped, nodes):
            ctg_backup = node.backup_act(edge.action)
            node.backup_val = ctg_backup
            ctgs_backup.append(ctg_backup)

        times.record_time("get_tr_data", time.time() - start_time)

        # to_np
        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)

    def _get_instance_data_rb(self, instances: List[InstanceQ], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.edges_popped)
        states_p, goals_p, is_solved_l_p, actions_p, tcs_p, states_next_p = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_p, goals_p, is_solved_l_p, actions_p, tcs_p, states_next_p, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)


class UpdateHeurQRLHER(UpdateHeurQRL[GoalSampleableFromState], UpdateHER[PathFindQHeur, InstanceQ]):
    def _get_instance_data_rb(self, instances: List[InstanceQ], times: Times) -> List[NDArray]:
        # get goals according to HER
        instances, goals_inst_her = self._get_her_goals(instances, times)

        # get states and goals
        start_time = time.time()
        states_her: List[State] = []
        goals_her: List[Goal] = []
        actions_her: List[Action] = []
        tcs_her: List[float] = []
        states_next_her: List[State] = []
        for instance, goal_her in zip(instances, goals_inst_her, strict=True):
            nodes: List[Node] = [edge.node for edge in instance.edges_popped]
            states_inst: List[State] = [node.state for node in nodes]
            states_her.extend(states_inst)
            goals_her.extend([goal_her] * len(states_inst))
            actions_her.extend([edge.action for edge in instance.edges_popped])

            for edge, node in zip(instance.edges_popped, nodes, strict=True):
                tc, node_next = node.edge_dict[edge.action]
                tcs_her.append(tc)
                states_next_her.append(node_next.state)

        times.record_time("data_her", time.time() - start_time)

        # is solved
        start_time = time.time()
        is_solved_l_her: List[bool] = self.domain.is_solved(states_her, goals_her)
        times.record_time("is_solved_her", time.time() - start_time)

        # add to replay buffer
        self._rb_add(states_her, goals_her, is_solved_l_her, actions_her, tcs_her, states_next_her, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(states_her), times)

        # to_np
        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)
