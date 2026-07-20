from abc import ABC
from typing import Any, List, Tuple, cast, Type

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, Action, State, Goal
from deepxube.base.pathfinding import PFNsHQ_T, PathFindSetHeurQ, EdgeQ, Instance, Node
from deepxube.base.pathfind_fns import PFNsHeurQ, PFNsHeurQPolicy, UFNsHeurQ, UFNsHeurQPolicy
from deepxube.base.updater import UpdateHER, UpdateHasPolicy, UpdateHeurQ, UpdateRL, D, UpdateRLParser, UFNsHQ_T
from deepxube.factories.updater_factory import updater_factory
from deepxube.updaters.utils.replay_buffer_utils import ReplayBufferQ
from deepxube.utils.timing_utils import Times

import time


def _pathfind_q_step(pathfind: PathFindSetHeurQ) -> List[EdgeQ]:
    edges_popped: List[EdgeQ] = pathfind.step()[1]
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


class UpdateHeurQRL(UpdateHeurQ[D, PFNsHQ_T, PathFindSetHeurQ, Instance, UFNsHQ_T], UpdateRL[D, PFNsHQ_T, PathFindSetHeurQ, Instance, UFNsHQ_T], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindSetHeurQ]:
        return PathFindSetHeurQ

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rb: ReplayBufferQ = ReplayBufferQ(0)

    def _step(self, pathfind: PathFindSetHeurQ, times: Times) -> None:
        _pathfind_q_step(pathfind)

    def _q_learning_target(self, goals: List[Goal], is_solved_l: List[bool], tcs: List[float], states_next: List[State]) -> List[float]:
        # min cost-to-go for next state
        actions_next: List[List[Action]] = self.get_pathfind().get_state_actions(states_next, goals)
        qvals_next_l: List[List[float]] = self._get_targ_heurq_fn()(states_next, goals, actions_next)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        return cast(List[float], ctg_backups.tolist())

    def _inputs_ctgs_to_np(self, states: List[State], goals: List[Goal], actions: List[Action], ctgs_backup: List[float], times: Times) -> List[NDArray]:
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heurq_nnet_par().process_inputs(states, goals, [[action] for action in actions]).inputs_nnet
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]

    def _init_replay_buffer(self, max_size: int) -> None:
        self.rb = ReplayBufferQ(max_size)

    def _rb_add(self, states: List[State], goals: List[Goal], is_solved_l: List[bool], actions: List[Action], tcs: List[float], states_next: List[State],
                times: Times) -> None:
        start_time = time.time()
        self.rb.add(list(zip(states, goals, is_solved_l, actions, tcs, states_next, strict=True)))
        times.record_time("add", time.time() - start_time, path=["replay"])

    def _sample_rb_qlearn_target(self, num: int, times: Times) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        # sample from replay buffer
        start_time = time.time()
        states, goals, is_solved_l, actions, tcs, states_next = self.rb.sample(num)
        times.record_time("samp", time.time() - start_time, path=["replay"])

        # value iteration update
        start_time = time.time()
        ctgs_backup: List[float] = self._q_learning_target(goals, is_solved_l, tcs, states_next)
        times.record_time("qlearn_targ", time.time() - start_time, path=["replay"])

        return states, goals, actions, ctgs_backup


class UpdateHeurQRLKeepGoalABC(UpdateHeurQRL[Domain, PFNsHQ_T, UFNsHQ_T], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _step_sync_main(self, pathfind: PathFindSetHeurQ, times: Times) -> List[NDArray]:
        # take a step
        edges_popped: List[EdgeQ] = _pathfind_q_step(pathfind)

        # get sync states/goals/is_solved
        states_sync, goals_sync, is_solved_l_sync, actions_sync, tcs_sync, states_next_sync = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_sync, goals_sync, is_solved_l_sync, actions_sync, tcs_sync, states_next_sync, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)

    def _get_instance_data_norb(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        # backup
        start_time = time.time()
        if not self.up_rl_args.lhbl:
            if self.up_rl_args.ub_heur_solns:
                for edge in edges_popped:
                    assert edge.node.is_solved is not None
                    if edge.node.is_solved:
                        edge.node.upper_bound_parent_path(0.0)
        else:
            for instance in instances:
                instance.root_node.tree_backup()

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

    def _get_instance_data_rb(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())
        states_p, goals_p, is_solved_l_p, actions_p, tcs_p, states_next_p = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_p, goals_p, is_solved_l_p, actions_p, tcs_p, states_next_p, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)


class UpdateHeurQRLHERABC(UpdateHeurQRL[GoalSampleableFromState, PFNsHQ_T, UFNsHQ_T], UpdateHER[PFNsHQ_T, PathFindSetHeurQ, Instance, UFNsHQ_T], ABC):
    @staticmethod
    def domain_type() -> Type[GoalSampleableFromState]:
        return GoalSampleableFromState

    def _get_instance_data_rb(self, instances: List[Instance], times: Times) -> List[NDArray]:
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
            nodes: List[Node] = [edge.node for edge in instance.get_edges_popped()]
            states_inst: List[State] = [node.state for node in nodes]
            states_her.extend(states_inst)
            goals_her.extend([goal_her] * len(states_inst))
            actions_her.extend([edge.action for edge in instance.get_edges_popped()])

            for edge, node in zip(instance.get_edges_popped(), nodes, strict=True):
                tc, node_next = node.edge_dict[edge.action]
                tcs_her.append(tc)
                states_next_her.append(node_next.state)

        times.record_time("data", time.time() - start_time, path=["HER"])

        # is solved
        start_time = time.time()
        is_solved_l_her: List[bool] = self.domain.is_solved(states_her, goals_her)
        times.record_time("is_solved", time.time() - start_time, path=["HER"])

        # add to replay buffer
        self._rb_add(states_her, goals_her, is_solved_l_her, actions_her, tcs_her, states_next_her, times)

        # rb q-learning update
        states, goals, actions, ctgs_backup = self._sample_rb_qlearn_target(len(states_her), times)

        # to_np
        return self._inputs_ctgs_to_np(states, goals, actions, ctgs_backup, times)


@updater_factory.register_class("up_rl_q")
class UpdateHeurQRLKeepGoal(UpdateHeurQRLKeepGoalABC[PFNsHeurQ, UFNsHeurQ]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQ]:
        return PFNsHeurQ

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQ]:
        return UFNsHeurQ

    def _get_pathfind_functions(self) -> PFNsHeurQ:
        return PFNsHeurQ(self.get_heurq_fn())


@updater_factory.register_class("up_her_q")
class UpdateHeurQRLHER(UpdateHeurQRLHERABC[PFNsHeurQ, UFNsHeurQ]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQ]:
        return PFNsHeurQ

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQ]:
        return UFNsHeurQ

    def _get_pathfind_functions(self) -> PFNsHeurQ:
        return PFNsHeurQ(self.get_heurq_fn())


@updater_factory.register_class("up_rl_q_p")
class UpdateHeurQRLKeepGoalPolicy(UpdateHeurQRLKeepGoalABC[PFNsHeurQPolicy, UFNsHeurQPolicy],
                                  UpdateHasPolicy[Domain, PFNsHeurQPolicy, PathFindSetHeurQ, Instance, UFNsHeurQPolicy]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQPolicy]:
        return PFNsHeurQPolicy

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQPolicy]:
        return UFNsHeurQPolicy

    def _get_pathfind_functions(self) -> PFNsHeurQPolicy:
        return PFNsHeurQPolicy(self.get_heurq_fn(), self.get_policy_fn())


@updater_factory.register_class("up_her_q_p")
class UpdateHeurQRLHERPolicy(UpdateHeurQRLHERABC[PFNsHeurQPolicy, UFNsHeurQPolicy],
                             UpdateHasPolicy[Domain, PFNsHeurQPolicy, PathFindSetHeurQ, Instance, UFNsHeurQPolicy]):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNsHeurQPolicy]:
        return PFNsHeurQPolicy

    @staticmethod
    def updater_functions_type() -> Type[UFNsHeurQPolicy]:
        return UFNsHeurQPolicy

    def _get_pathfind_functions(self) -> PFNsHeurQPolicy:
        return PFNsHeurQPolicy(self.get_heurq_fn(), self.get_policy_fn())


@updater_factory.register_parser("up_rl_q")
class UpdateVRL(UpdateRLParser):
    pass


@updater_factory.register_parser("up_her_q")
class UpdateVRLHER(UpdateRLParser):
    pass


@updater_factory.register_parser("up_rl_q_p")
class UpdateVPRL(UpdateRLParser):
    pass


@updater_factory.register_parser("up_her_q_p")
class UpdateVPRLHER(UpdateRLParser):
    pass
