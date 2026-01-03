import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, cast

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Action, State, Goal, ActsEnum
from deepxube.base.heuristic import HeurNNetParQ, HeurFnQ
from deepxube.base.pathfinding import PathFindQHeur, EdgeQ, Instance, Node
from deepxube.base.updater import UpdateHeurRL, D, UpArgs, UpHeurArgs
from deepxube.utils.timing_utils import Times


class UpdateHeurRLQ(UpdateHeurRL[D, PathFindQHeur, HeurNNetParQ, HeurFnQ], ABC):
    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
        self.edges_popped: List[EdgeQ] = []

    def get_heur_train_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.get_start_goal_pairs([0])
        actions: List[Action] = self.domain.get_state_action_rand(states)
        inputs_nnet: List[NDArray[Any]] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dtypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dtypes

    def _step(self, pathfind: PathFindQHeur, times: Times) -> List[NDArray]:
        # take a step
        edges_popped: List[EdgeQ] = pathfind.step()
        assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"

        if not self.up_args.sync_main:
            self.edges_popped.extend(edges_popped)
            return []
        else:
            start_time = time.time()
            states, goals, is_solved_l, actions, tcs, states_next = self._get_edge_data(edges_popped)
            ctgs_backup: List[float] = self._q_learning_backup_targ(goals, is_solved_l, tcs, states_next)
            times.record_time("backup_sync", time.time() - start_time)

            return self._inputs_ctgs_np(states, goals, actions, ctgs_backup, times)

    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        states, goals, actions, ctgs_backup = self._backup_edges(self.edges_popped, times)

        # to_np
        inputs_ctgs_np: List[NDArray] = self._inputs_ctgs_np(states, goals, actions, ctgs_backup, times)

        self.edges_popped = []
        return inputs_ctgs_np

    def _backup_edges(self, edges: List[EdgeQ], times: Times) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        start_time = time.time()
        edges_init, edges_real = _split_init_vs_real_edges(edges)
        times.record_time("split_edges", time.time() - start_time)

        # get backup of initial edge with random action
        # TODO this could be taking up a lot of GPU since includes more instances in parallel (i.e. both removed and current)
        start_time = time.time()
        states, goals, actions, ctgs_backup = self._backup_any_next_edge(edges_init)
        assert len(states) == len(goals) == len(actions) == len(ctgs_backup), \
            f"Values were {len(states)}, {len(goals)}, {len(actions)}, {len(ctgs_backup)}, "
        times.record_time("backup_init", time.time() - start_time)

        # get backup for real edges
        start_time = time.time()
        for edge_real in edges_real:
            node: Node = edge_real.node
            states.append(node.state)
            goals.append(node.goal)
            action: Optional[Action] = edge_real.action
            assert action is not None

            actions.append(action)
            ctgs_backup.append(node.backup_act(action))
        times.record_time("backup_real", time.time() - start_time)

        return states, goals, actions, ctgs_backup

    def _backup_any_next_edge(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        if len(edges) == 0:
            return [], [], [], []
        states, goals, is_solved_l, actions, tcs, states_next = self._edge_init_next_random(edges)
        ctgs_backup: List[float] = self._q_learning_backup_targ(goals, is_solved_l, tcs, states_next)

        return states, goals, actions, ctgs_backup

    def _get_edge_data(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:
        edges_init, edges_real = _split_init_vs_real_edges(edges)
        states, goals, is_solved_l, actions, tcs, states_next = self._edge_init_next_random(edges_init)
        for edge_real in edges_real:
            node: Node = edge_real.node
            states.append(node.state)
            goals.append(node.goal)
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)
            assert edge_real.action is not None
            actions.append(edge_real.action)
            tc, node_next = node.edge_dict[edge_real.action]
            tcs.append(tc)
            states_next.append(node_next.state)

        return states, goals, is_solved_l, actions, tcs, states_next

    def _edge_init_next_random(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:
        if len(edges) == 0:
            return [], [], [], [], [], []

        node_l: List[Node] = [edge.node for edge in edges]
        states: List[State] = [node.state for node in node_l]
        goals: List[Goal] = [node.goal for node in node_l]
        is_solved_l: List[bool] = []
        for node in node_l:
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)

        actions: List[Action] = self.domain.get_state_action_rand(states)

        states_next, tcs = self.domain.next_state(states, actions)
        assert len(states) == len(goals) == len(is_solved_l) == len(actions) == len(tcs) == len(states_next), \
            f"Values were {len(states)}, {len(goals)}, {len(is_solved_l)}, {len(actions)}, {len(tcs)}, {len(states_next)}"

        return states, goals, is_solved_l, actions, tcs, states_next

    @abstractmethod
    def _get_qvals_targ(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        pass

    def _q_learning_backup_targ(self, goals: List[Goal], is_solved_l: List[bool], tcs: List[float], states_next: List[State]) -> List[float]:
        # min cost-to-go for next state
        qvals_next_l: List[List[float]] = self._get_qvals_targ(states_next, goals)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        return cast(List[float], ctg_backups.tolist())

    def _inputs_ctgs_np(self, states: List[State], goals: List[Goal], actions: List[Action], ctgs_backup: List[float], times: Times) -> List[NDArray]:
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]


class UpdateHeurRLQEnum(UpdateHeurRLQ[ActsEnum]):
    def _get_qvals_targ(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        actions_next: List[List[Action]] = self.domain.get_state_actions(states)
        qvals: List[List[float]] = self._get_targ_heur_fn()(states, goals, actions_next)

        return qvals


def _split_init_vs_real_edges(edges: List[EdgeQ]) -> Tuple[List[EdgeQ], List[EdgeQ]]:
    edges_init: List[EdgeQ] = []
    edges_real: List[EdgeQ] = []
    for edge in edges:
        if edge.action is None:
            edges_init.append(edge)
        else:
            edges_real.append(edge)
    return edges_init, edges_real
