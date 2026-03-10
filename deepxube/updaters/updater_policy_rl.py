from abc import ABC
from typing import List, Tuple, Type

from numpy.typing import NDArray

from deepxube.base.domain import Domain, GoalSampleableFromState, Action, State, Goal
from deepxube.base.heuristic import HeurNNetParV, HeurNNetParQ, HeurFnV, HeurFnQ
from deepxube.base.pathfinding import FNsP, FNsPolicy, FNsHeurVPolicy, FNsHeurQPolicy, PathFind, PathFindActsPolicy, EdgeQ, Node, Instance
from deepxube.base.updater import UpdateHER, UpdatePolicy, UpdateHasHeur, UpdateRL, D, UpArgs
from deepxube.factories.updater_factory import updater_factory
from deepxube.updaters.utils.replay_buffer_utils import ReplayBufferP
from deepxube.utils.timing_utils import Times

import numpy as np
import time


def _pathfind_step(pathfind: PathFind) -> List[EdgeQ]:
    edges_popped: List[EdgeQ] = pathfind.step()[1]
    assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"

    return edges_popped


def _get_edge_popped_data(edges_popped: List[EdgeQ], times: Times) -> Tuple[List[State], List[Goal], List[Action]]:
    start_time = time.time()
    nodes: List[Node] = [edge.node for edge in edges_popped]
    states: List[State] = [node.state for node in nodes]
    goals: List[Goal] = [node.goal for node in nodes]
    actions: List[Action] = [edge.action for edge in edges_popped]

    times.record_time("edge_data", time.time() - start_time)

    return states, goals, actions


class UpdatePolicyRL(UpdatePolicy[D, FNsP, PathFindActsPolicy, Instance], UpdateRL[D, FNsP, PathFindActsPolicy, Instance], ABC):
    @staticmethod
    def pathfind_type() -> Type[PathFindActsPolicy]:
        return PathFindActsPolicy

    def __init__(self, domain: D, pathfind_arg: str, up_args: UpArgs):
        super().__init__(domain, pathfind_arg, up_args)
        self.rb: ReplayBufferP = ReplayBufferP(0)

    def _step(self, pathfind: PathFindActsPolicy, times: Times) -> None:
        _pathfind_step(pathfind)

    def _inputs_ctgs_to_np(self, states: List[State], goals: List[Goal], actions: List[Action], times: Times) -> List[NDArray]:
        # sample random actions
        start_time = time.time()
        rand_idxs: List[int] = np.flatnonzero(np.random.random(len(states)) < self.up_args.policy_rand_prob).tolist()
        if len(rand_idxs) > 0:
            states_rand_acts: List[State] = [states[rand_idx] for rand_idx in rand_idxs]
            actions_rand: List[Action] = self.domain.sample_state_action(states_rand_acts)
            for rand_idx, action_rand in zip(rand_idxs, actions_rand):
                actions[rand_idx] = action_rand
        times.record_time("rand_acts", time.time() - start_time)

        # to_np
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_policy_nnet_par().to_np_train(states, goals, actions)
        times.record_time("to_np", time.time() - start_time)

        return inputs_np

    def _init_replay_buffer(self, max_size: int) -> None:
        self.rb = ReplayBufferP(max_size)

    def _rb_add(self, states: List[State], goals: List[Goal], actions: List[Action], times: Times) -> None:
        start_time = time.time()
        self.rb.add(list(zip(states, goals, actions, strict=True)))
        times.record_time("rb_add", time.time() - start_time)

    def _sample_rb(self, num: int, times: Times) -> Tuple[List[State], List[Goal], List[Action]]:
        # sample from replay buffer
        start_time = time.time()
        states, goals, actions = self.rb.sample(num)
        times.record_time("rb_samp", time.time() - start_time)

        return states, goals, actions


class UpdatePolicyRLKeepGoalABC(UpdatePolicyRL[Domain, FNsP], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def _step_sync_main(self, pathfind: PathFindActsPolicy, times: Times) -> List[NDArray]:
        # take a step
        edges_popped: List[EdgeQ] = _pathfind_step(pathfind)

        # get sync states/goals/is_solved
        states_sync, goals_sync, actions_sync = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_sync, goals_sync, actions_sync, times)

        # rb q-learning update
        states, goals, actions = self._sample_rb(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, times)

    def _get_instance_data_norb(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())

        start_time = time.time()
        nodes: List[Node] = [edge.node for edge in edges_popped]
        states: List[State] = [node.state for node in nodes]
        goals: List[Goal] = [node.goal for node in nodes]
        actions: List[Action] = [edge.action for edge in edges_popped]

        times.record_time("get_tr_data", time.time() - start_time)

        # to_np
        return self._inputs_ctgs_to_np(states, goals, actions, times)

    def _get_instance_data_rb(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get popped edge data
        edges_popped: List[EdgeQ] = []
        for instance in instances:
            edges_popped.extend(instance.get_edges_popped())
        states_p, goals_p, actions_p = _get_edge_popped_data(edges_popped, times)

        # add to replay buffer
        self._rb_add(states_p, goals_p, actions_p, times)

        # rb q-learning update
        states, goals, actions = self._sample_rb(len(edges_popped), times)

        return self._inputs_ctgs_to_np(states, goals, actions, times)


class UpdatePolicyRLHERABC(UpdatePolicyRL[GoalSampleableFromState, FNsP], UpdateHER[FNsP, PathFindActsPolicy, Instance], ABC):
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
        for instance, goal_her in zip(instances, goals_inst_her, strict=True):
            nodes: List[Node] = [edge.node for edge in instance.get_edges_popped()]
            states_inst: List[State] = [node.state for node in nodes]
            states_her.extend(states_inst)
            goals_her.extend([goal_her] * len(states_inst))
            actions_her.extend([edge.action for edge in instance.get_edges_popped()])

        times.record_time("data_her", time.time() - start_time)

        # add to replay buffer
        self._rb_add(states_her, goals_her, actions_her, times)

        # rb q-learning update
        states, goals, actions = self._sample_rb(len(states_her), times)

        # to_np
        return self._inputs_ctgs_to_np(states, goals, actions, times)


@updater_factory.register_class("update_p_rl")
class UpdatePolicyRLKeepGoal(UpdatePolicyRLKeepGoalABC[FNsPolicy]):
    @staticmethod
    def functions_type() -> Type[FNsPolicy]:
        return FNsPolicy

    def _get_pathfind_functions(self) -> FNsPolicy:
        return FNsPolicy(self.get_policy_fn())


@updater_factory.register_class("update_p_rl_her")
class UpdatePolicyRLHER(UpdatePolicyRLHERABC[FNsPolicy]):
    @staticmethod
    def functions_type() -> Type[FNsPolicy]:
        return FNsPolicy

    def _get_pathfind_functions(self) -> FNsPolicy:
        return FNsPolicy(self.get_policy_fn())


@updater_factory.register_class("update_p_v_rl")
class UpdatePolicyRLKeepGoalHeurV(UpdatePolicyRLKeepGoalABC[FNsHeurVPolicy],
                                  UpdateHasHeur[Domain, FNsHeurVPolicy, PathFindActsPolicy, Instance, HeurNNetParV, HeurFnV]):
    @staticmethod
    def functions_type() -> Type[FNsHeurVPolicy]:
        return FNsHeurVPolicy

    def _get_pathfind_functions(self) -> FNsHeurVPolicy:
        return FNsHeurVPolicy(self.get_heur_fn(), self.get_policy_fn())


@updater_factory.register_class("update_p_v_rl_her")
class UpdatePolicyRLHERHeurV(UpdatePolicyRLHERABC[FNsHeurVPolicy], UpdateHasHeur[Domain, FNsHeurVPolicy, PathFindActsPolicy, Instance, HeurNNetParV, HeurFnV]):
    @staticmethod
    def functions_type() -> Type[FNsHeurVPolicy]:
        return FNsHeurVPolicy

    def _get_pathfind_functions(self) -> FNsHeurVPolicy:
        return FNsHeurVPolicy(self.get_heur_fn(), self.get_policy_fn())


@updater_factory.register_class("update_p_q_rl")
class UpdatePolicyRLKeepGoalHeurQ(UpdatePolicyRLKeepGoalABC[FNsHeurQPolicy],
                                  UpdateHasHeur[Domain, FNsHeurQPolicy, PathFindActsPolicy, Instance, HeurNNetParQ, HeurFnQ]):
    @staticmethod
    def functions_type() -> Type[FNsHeurQPolicy]:
        return FNsHeurQPolicy

    def _get_pathfind_functions(self) -> FNsHeurQPolicy:
        return FNsHeurQPolicy(self.get_heur_fn(), self.get_policy_fn())


@updater_factory.register_class("update_p_q_rl_her")
class UpdatePolicyRLHERHeurQ(UpdatePolicyRLHERABC[FNsHeurQPolicy], UpdateHasHeur[Domain, FNsHeurQPolicy, PathFindActsPolicy, Instance, HeurNNetParQ, HeurFnQ]):
    @staticmethod
    def functions_type() -> Type[FNsHeurQPolicy]:
        return FNsHeurQPolicy

    def _get_pathfind_functions(self) -> FNsHeurQPolicy:
        return FNsHeurQPolicy(self.get_heur_fn(), self.get_policy_fn())
