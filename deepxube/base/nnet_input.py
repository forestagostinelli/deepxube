from typing import Any, List, Tuple, Generic, TypeVar
from abc import ABC, abstractmethod

from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed

from numpy.typing import NDArray


D = TypeVar('D', bound=Domain)
S = TypeVar('S', bound=State)
A = TypeVar('A', bound=Action)
G = TypeVar('G', bound=Goal)


class NNetInput(ABC, Generic[D]):
    def __init__(self, domain: D):
        self.domain: D = domain

    @abstractmethod
    def get_input_info(self) -> Any:
        pass

    @abstractmethod
    def to_np(self, *args: Any) -> List[NDArray]:
        pass


class FlatIn(NNetInput[D]):
    @abstractmethod
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        pass


class StateGoalIn(NNetInput[D], Generic[D, S, G]):
    @abstractmethod
    def to_np(self, states: List[S], goals: List[G]) -> List[NDArray]:
        pass


class StateGoalActFixIn(NNetInput[D], Generic[D, S, G, A]):
    @abstractmethod
    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray]:
        pass


class StateGoalActIn(NNetInput[D], Generic[D, S, G, A]):
    @abstractmethod
    def to_np(self, states: List[S], goals: List[G], actions: List[A]) -> List[NDArray]:
        pass


# Env mixins for inputs
class HasActsEnumFixedIn(ActsEnumFixed[S, A, G]):
    @abstractmethod
    def actions_to_indices(self, actions: List[A]) -> List[int]:
        pass


class HasFlatSGIn(Domain[S, A, G]):
    @abstractmethod
    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        pass

    @abstractmethod
    def to_np_flat_sg(self, states: List[S], goals: List[G]) -> List[NDArray]:
        pass


class HasFlatSGActsEnumFixedIn(HasFlatSGIn[S, A, G], HasActsEnumFixedIn[S, A, G], ABC):
    pass


class HasFlatSGAIn(Domain[S, A, G]):
    @abstractmethod
    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        pass

    @abstractmethod
    def to_np_flat_sga(self, states: List[S], goals: List[G], actions: List[A]) -> List[NDArray]:
        pass
