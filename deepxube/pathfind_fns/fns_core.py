from typing import List, Type

from deepxube.utils import misc_utils
from deepxube.pytorch.nnet_utils import ProcessedInput
from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed
from deepxube.base.nnet_input import StateGoalActFixIn, StateGoalActIn
from dataclasses import dataclass
from deepxube.base.pathfind_fns import (PFNsHeurV, PFNsHeurQ, PFNsPolicy, PFNsHeurVPolicy, PFNsHeurQPolicy, HeurVNNetPar, HeurQNNetPar, PolicyNNetPar,
                                        UFNsHeurV, UFNsHeurQ, UFNsPolicy, UFNsHeurVPolicy, UFNsHeurQPolicy)
from deepxube.factories.pathfind_fns_factory import pathfind_fns_factory, deepxube_nnet_par_factory, updater_fns_factory

import numpy as np
from numpy.typing import NDArray


# pathfind functions

@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurVC(PFNsHeurV):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurQC(PFNsHeurQ):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsPolicyC(PFNsPolicy):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurVPolicyC(PFNsHeurVPolicy):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurQPolicyC(PFNsHeurQPolicy):
    pass


# nnet par functions

@deepxube_nnet_par_factory.register_class("heurv")
class HeurVNNetParC(HeurVNNetPar):
    pass


@dataclass(frozen=True)
class QOutFixCtx:
    states: List[State]


@deepxube_nnet_par_factory.register_class("heurq_fixout")
class HeurQNNetParFixOut(HeurQNNetPar[QOutFixCtx, ActsEnumFixed, StateGoalActFixIn]):
    @staticmethod
    def domain_type() -> Type[ActsEnumFixed]:
        return ActsEnumFixed

    @staticmethod
    def nnet_input_type() -> Type[StateGoalActFixIn]:
        return StateGoalActFixIn

    @staticmethod
    def _check_same_num_acts(actions_l: List[List[Action]]) -> None:
        assert len(set(len(actions) for actions in actions_l)) == 1, "num actions should be the same for all instances"

    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[QOutFixCtx]:
        self._check_same_num_acts(actions_l)
        return ProcessedInput(self._get_nnet_input().to_np(states, goals, actions_l), QOutFixCtx(states))

    def process_outputs(self, outs: List[NDArray], ctx: QOutFixCtx) -> List[List[float]]:
        q_vals_np: NDArray = outs[0]
        assert q_vals_np.shape[0] == len(ctx.states)

        q_vals_np = np.maximum(q_vals_np, 0)
        q_vals_l: List[List[float]] = [q_vals_np[state_idx].astype(np.float64).tolist() for state_idx in range(q_vals_np.shape[0])]
        return q_vals_l

    def _qfix(self) -> bool:
        return True

    def _out_dim(self) -> int:
        return self.domain.get_num_acts()


@dataclass(frozen=True)
class QInCtx:
    states_rep: List[State]
    split_idxs: List[int]


@deepxube_nnet_par_factory.register_class("heurq_in")
class HeurQNNetParIn(HeurQNNetPar[QInCtx, Domain, StateGoalActIn]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[StateGoalActIn]:
        return StateGoalActIn

    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[QInCtx]:
        actions_flat, split_idxs = misc_utils.flatten(actions_l)
        states_rep: List[State] = []
        goals_rep: List[Goal] = []
        for state, goal, actions in zip(states, goals, actions_l, strict=True):
            states_rep.extend([state] * len(actions))
            goals_rep.extend([goal] * len(actions))

        return ProcessedInput(self._get_nnet_input().to_np(states_rep, goals_rep, actions_flat), QInCtx(states_rep, split_idxs))

    def process_outputs(self, outs: List[NDArray], ctx: QInCtx) -> List[List[float]]:
        q_vals_np: NDArray = outs[0]

        assert q_vals_np.shape[0] == len(ctx.states_rep)
        q_vals_np = np.maximum(q_vals_np[:, 0], 0)

        q_vals_flat: List[float] = q_vals_np.astype(np.float64).tolist()
        q_vals_l: List[List[float]] = misc_utils.unflatten(q_vals_flat, ctx.split_idxs)
        return q_vals_l

    def _qfix(self) -> bool:
        return False

    def _out_dim(self) -> int:
        return 1


@deepxube_nnet_par_factory.register_class("policy")
class PolicyNNetParC(PolicyNNetPar):
    pass


@updater_fns_factory.register
@dataclass(frozen=True)
class UFNsHeurVC(UFNsHeurV):
    pass


@updater_fns_factory.register
@dataclass(frozen=True)
class UFNsHeurQC(UFNsHeurQ):
    pass


@updater_fns_factory.register
@dataclass(frozen=True)
class UPFNsPolicyC(UFNsPolicy):
    pass


@updater_fns_factory.register
@dataclass(frozen=True)
class UPFNsHeurVPolicyC(UFNsHeurVPolicy):
    pass


@updater_fns_factory.register
@dataclass(frozen=True)
class UPFNsHeurQPolicyC(UFNsHeurQPolicy):
    pass
