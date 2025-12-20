from typing import Dict, Tuple, Type, Callable, List

from deepxube.base.nnet_input import (NNetInput, FlatIn, StateGoalIn, StateGoalActFixIn, StateGoalActIn, HasFlatSGIn,
                                      HasFlatSGActFixIn, HasFlatSGAIn)
from deepxube.base.domain import Domain, State, Goal, Action
from deepxube.factories.domain_factory import get_all_domain_names, get_domain_type


import numpy as np
from numpy.typing import NDArray


_nnet_input_registry: Dict[Tuple[str, str], Type[NNetInput]] = {}


def register_nnet_input(domain_name: str, nnet_input_name: str) -> Callable[[Type[NNetInput]], Type[NNetInput]]:
    def deco(cls: Type[NNetInput]) -> Type[NNetInput]:
        key: Tuple[str, str] = (domain_name, nnet_input_name)
        if key in _nnet_input_registry.keys():
            raise ValueError(f"{key!r} already registered for nnet inputs")
        _nnet_input_registry[key] = cls
        return cls
    return deco


def get_domain_nnet_input_keys(domain_name: str) -> List[Tuple[str, str]]:
    return [key for key in _nnet_input_registry.keys() if key[0] == domain_name]


def get_nnet_input_t(key: Tuple[str, str]) -> Type[NNetInput]:
    return _nnet_input_registry[key]


def register_nnet_input_dynamic() -> None:
    # register dynamically created nnet inputs
    for domain_name in get_all_domain_names():
        domain_t: Type[Domain] = get_domain_type(domain_name)
        if issubclass(domain_t, HasFlatSGIn):
            class FlatSGConcrete(FlatIn[HasFlatSGIn], StateGoalIn[HasFlatSGIn, State, Goal]):
                def __init__(self, domain: HasFlatSGIn):
                    super().__init__(domain)

                def get_input_info(self) -> Tuple[List[int], List[int]]:
                    return self.domain.get_input_info_flat_sg()

                def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
                    return self.domain.to_np_flat_sg(states, goals)

            register_nnet_input(domain_name, "flat_sg_dynamic")(FlatSGConcrete)

        if issubclass(domain_t, HasFlatSGActFixIn):
            class FlatSGActFixConcrete(FlatIn[HasFlatSGActFixIn],
                                       StateGoalActFixIn[HasFlatSGActFixIn, State, Goal, Action]):
                def __init__(self, domain: HasFlatSGActFixIn):
                    super().__init__(domain)

                def get_input_info(self) -> Tuple[List[int], List[int]]:
                    return self.domain.get_input_info_flat_sg()

                def to_np(self, states: List[State], goals: List[Goal],
                          actions_l: List[List[Action]]) -> List[NDArray]:
                    num_actions: int = len(actions_l[0])
                    actions_np: NDArray = np.zeros((len(actions_l), num_actions)).astype(int)
                    for i, actions in enumerate(actions_l):
                        actions_np[i] = np.array(self.domain.actions_to_indices(actions))

                    return self.domain.to_np_flat_sg(states, goals) + [actions_np]

            register_nnet_input(domain_name, "flat_sg_actfix_dynamic")(FlatSGActFixConcrete)

        if issubclass(domain_t, HasFlatSGAIn):
            class FlatSGAConcrete(FlatIn[HasFlatSGAIn], StateGoalActIn[HasFlatSGAIn, State, Goal, Action]):
                def __init__(self, domain: HasFlatSGAIn):
                    super().__init__(domain)

                def get_input_info(self) -> Tuple[List[int], List[int]]:
                    return self.domain.get_input_info_flat_sga()

                def to_np(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray]:
                    return self.domain.to_np_flat_sga(states, goals, actions)

            register_nnet_input(domain_name, "flat_sga_dynamic")(FlatSGAConcrete)
