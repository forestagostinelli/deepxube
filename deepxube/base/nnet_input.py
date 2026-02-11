from typing import Any, List, Tuple, Generic, TypeVar, Type, ClassVar, Dict, Optional
from abc import ABC, abstractmethod

from deepxube.base.domain import Domain, State, Action, Goal

import numpy as np

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
        """
        :return: A list of dimensions of the arrays given to the neural network (pre one_hot), A list of depths for performing a one_hot representation on
        that corresponding input.
        If 1, then no one_hot is performed.
        """
        pass


class TwoDIn(NNetInput[D]):
    @abstractmethod
    def get_input_info(self) -> Tuple[List[int], Tuple[int, int], List[int], Optional[int]]:
        """
        :return: A list of channels of the arrays given to the neural network (pre one_hot), (height, width),
        a list of depths for performing a one_hot representation on that corresponding input, optional 1x1 conv channel out for qfix.
        The one_hot is applied to the channel dimension. If 1, then no one_hot is performed.
        """
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
class DynamicNNetInput(Domain[S, A, G], ABC):
    _nnet_input_register: ClassVar[Dict[str, Type[NNetInput]]] = dict()

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        # Create a fresh dict for THIS subclass
        cls._nnet_input_register = {}

    @classmethod
    def register_nnet_input(cls, nnet_input_t: Type[NNetInput], nnet_input_name: str) -> None:
        cls._nnet_input_register[nnet_input_name] = nnet_input_t

    @classmethod
    def get_dynamic_nnet_inputs(cls) -> Dict[str, Type[NNetInput]]:
        return cls._nnet_input_register.copy()


class HasFlatSGIn(DynamicNNetInput[S, A, G]):
    """ Has a flat representation for state/goal inputs

    """

    class FlatSGConcrete(FlatIn["HasFlatSGIn"], StateGoalIn["HasFlatSGIn", State, Goal]):
        def __init__(self, domain: "HasFlatSGIn"):
            super().__init__(domain)

        def get_input_info(self) -> Tuple[List[int], List[int]]:
            return self.domain.get_input_info_flat_sg()

        def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
            return self.domain.to_np_flat_sg(states, goals)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.register_nnet_input(cls.FlatSGConcrete, "flat_sg")

    @abstractmethod
    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        """
        :return: A list of dimensions of the arrays given to the neural network (pre one_hot), A list of depths for performing a one_hot representation on
        that corresponding input.
        If 1, then no one_hot is performed.
        """
        pass

    @abstractmethod
    def to_np_flat_sg(self, states: List[S], goals: List[G]) -> List[NDArray]:
        pass


class HasActsEnumFixedIn(Domain[S, A, G]):
    @abstractmethod
    def actions_to_indices(self, actions: List[A]) -> List[int]:
        pass


class HasFlatSGActsEnumFixedIn(HasFlatSGIn[S, A, G], HasActsEnumFixedIn[S, A, G], ABC):
    class FlatSGActFixConcrete(FlatIn["HasFlatSGActsEnumFixedIn"], StateGoalActFixIn["HasFlatSGActsEnumFixedIn", State, Goal, Action]):
        def __init__(self, domain: "HasFlatSGActsEnumFixedIn"):
            super().__init__(domain)

        def get_input_info(self) -> Tuple[List[int], List[int]]:
            return self.domain.get_input_info_flat_sg()

        def to_np(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
            num_actions: int = len(actions_l[0])
            actions_np: NDArray = np.zeros((len(actions_l), num_actions)).astype(int)
            for i, actions in enumerate(actions_l):
                actions_np[i] = np.array(self.domain.actions_to_indices(actions))

            return self.domain.to_np_flat_sg(states, goals) + [actions_np]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.register_nnet_input(cls.FlatSGActFixConcrete, "flat_sg_actfix")


class HasFlatSGAIn(DynamicNNetInput[S, A, G]):
    class FlatSGAConcrete(FlatIn["HasFlatSGAIn"], StateGoalActIn["HasFlatSGAIn", State, Goal, Action]):
        def __init__(self, domain: "HasFlatSGAIn"):
            super().__init__(domain)

        def get_input_info(self) -> Tuple[List[int], List[int]]:
            return self.domain.get_input_info_flat_sga()

        def to_np(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray]:
            return self.domain.to_np_flat_sga(states, goals, actions)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.register_nnet_input(cls.FlatSGAConcrete, "flat_sga")

    @abstractmethod
    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        pass

    @abstractmethod
    def to_np_flat_sga(self, states: List[S], goals: List[G], actions: List[A]) -> List[NDArray]:
        pass


class HasTwoDSGIn(DynamicNNetInput[S, A, G]):
    """ Has a 2d representation for state/goal inputs

    """

    class TwoDSGConcrete(TwoDIn["HasTwoDSGIn"], StateGoalIn["HasTwoDSGIn", State, Goal]):
        def __init__(self, domain: "HasTwoDSGIn"):
            super().__init__(domain)

        def get_input_info(self) -> Tuple[List[int], Tuple[int, int], List[int], Optional[int]]:
            return self.domain.get_input_info_2d_sg()

        def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
            return self.domain.to_np_2d_sg(states, goals)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.register_nnet_input(cls.TwoDSGConcrete, "2d_sg")

    @abstractmethod
    def get_input_info_2d_sg(self) -> Tuple[List[int], Tuple[int, int], List[int], Optional[int]]:
        """
        :return: A list of channels of the arrays given to the neural network (pre one_hot), (height, width),
        a list of depths for performing a one_hot representation on that corresponding input, optional 1x1 conv channel out for qfix.
        The one_hot is applied to the channel dimension. If 1, then no one_hot is performed.
        """
        pass

    @abstractmethod
    def to_np_2d_sg(self, states: List[S], goals: List[G]) -> List[NDArray]:
        """
        :param states: List of states
        :param goals: List of goals

        :return: list of arrays representing states and goals in (chan, height, width) format
        """


class HasTwoDSGActsEnumFixedIn(HasTwoDSGIn[S, A, G], HasActsEnumFixedIn[S, A, G], ABC):
    class TwoDSGActFixConcrete(TwoDIn["HasTwoDSGActsEnumFixedIn"], StateGoalActFixIn["HasTwoDSGActsEnumFixedIn", State, Goal, Action]):
        def __init__(self, domain: "HasTwoDSGActsEnumFixedIn"):
            super().__init__(domain)

        def get_input_info(self) -> Tuple[List[int], Tuple[int, int], List[int], Optional[int]]:
            return self.domain.get_input_info_2d_sg()

        def to_np(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
            num_actions: int = len(actions_l[0])
            actions_np: NDArray = np.zeros((len(actions_l), num_actions)).astype(int)
            for i, actions in enumerate(actions_l):
                actions_np[i] = np.array(self.domain.actions_to_indices(actions))

            return self.domain.to_np_2d_sg(states, goals) + [actions_np]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.register_nnet_input(cls.TwoDSGActFixConcrete, "2d_sg_actfix")
