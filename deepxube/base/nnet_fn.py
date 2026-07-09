from dataclasses import dataclass
from typing import List, Union, runtime_checkable, Protocol, Tuple
from deepxube.base.domain import State, Action, Goal


@runtime_checkable
class HeurVFn(Protocol):
    """ Maps states and goals to cost-to-go """
    def __call__(self, states: List[State], goals: List[Goal]) -> List[float]:
        ...


@runtime_checkable
class HeurQFn(Protocol):
    """ Maps states, goals, and actions to transitions cost plus cost-to-go of resulting state """
    def __call__(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        ...


HeurFn = Union[HeurVFn, HeurQFn]


@runtime_checkable
class PolicyFn(Protocol):
    """ Samples actions and their corresponding log probabilities given states and goals """
    def __call__(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
        """ Map states and goals to sampled actions along with their probability (or log probability) densities

        """
        ...


@dataclass(frozen=True)
class FNsHeurV:
    heur_fn_v: HeurVFn


@dataclass(frozen=True)
class FNsHeurQ:
    heur_fn_q: HeurQFn


FNsHeur = Union[FNsHeurV, FNsHeurQ]


@dataclass(frozen=True)
class FNsPolicy:
    policy_fn: PolicyFn


@dataclass(frozen=True)
class FNsHeurVPolicy(FNsPolicy, FNsHeurV):
    pass


@dataclass(frozen=True)
class FNsHeurQPolicy(FNsPolicy, FNsHeurQ):
    pass
