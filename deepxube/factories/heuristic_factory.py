from abc import ABC
from typing import Dict, Tuple, Type, List, Any, Optional

import numpy as np

from deepxube.base.factory import Factory
from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed
from deepxube.base.nnet_input import NNetInput, StateGoalIn, StateGoalActFixIn, StateGoalActIn, PolicyNNetIn
from deepxube.base.heuristic import HeurNNet, PolicyNNet, HeurNNetPar, PolicyNNetPar, HeurNNetParV, HeurNNetParQIn, HeurNNetParQFixOut

from numpy.typing import NDArray

from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t


heuristic_factory: Factory[HeurNNet] = Factory[HeurNNet]("HeurNNet")
policy_factory: Factory[PolicyNNet] = Factory[PolicyNNet]("PolicyNNet")


def build_heur_nnet_par(domain: Domain, domain_name: str, nnet_name: str, nnet_kwargs: Dict[str, Any], heur_type: str) -> HeurNNetPar:
    nnet_input_t: Type[NNetInput] = heuristic_factory.get_type(nnet_name).nnet_input_type()
    nnet_input_domain_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)

    for nnet_input_domain_key in nnet_input_domain_keys:
        nnet_input_cls: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_key)
        if heur_type.upper() == "V":
            if issubclass(nnet_input_cls, StateGoalIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParVConcrete(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)
        elif heur_type.upper() == "QFIX":
            assert isinstance(domain, ActsEnumFixed)
            if issubclass(nnet_input_cls, StateGoalActFixIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParQFixOutConcrete(domain, nnet_input_domain_key, nnet_name, nnet_kwargs, domain.get_num_acts())
        elif heur_type.upper() == "QIN":
            if issubclass(nnet_input_cls, StateGoalActIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParQActInConcrete(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)
        else:
            raise ValueError(f"Unknown heur type {heur_type}")
    raise ValueError(f"Cannot build heur nnet for domain: {domain_name}, heur type {heur_type}, and "
                     f"nnet_input type {nnet_input_t}.\nNNet inputs checked: {nnet_input_domain_keys}")


def build_policy_nnet_par(domain: Domain, domain_name: str, nnet_name: str, nnet_kwargs: Dict[str, Any]) -> PolicyNNetPar:
    nnet_input_t: Type[NNetInput] = policy_factory.get_type(nnet_name).nnet_input_type()
    nnet_input_domain_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
    for nnet_input_domain_key in nnet_input_domain_keys:
        nnet_input_cls: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_key)
        if issubclass(nnet_input_cls, PolicyNNetIn) and issubclass(nnet_input_cls, nnet_input_t):
            return PolicyNNetParConcrete(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)

    raise ValueError(f"Cannot build policy nnet for domain: {domain_name}, and "
                     f"nnet_input type {nnet_input_t}.\nNNet inputs checked: {nnet_input_domain_keys}")


class HeurNNetParFacClass(HeurNNetPar, ABC):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], nnet_name: str, nnet_kwargs: Dict[str, Any], q_fix: bool, out_dim: int):
        self.domain: Domain = domain
        self.nnet_input_name: Tuple[str, str] = nnet_input_name
        self.nnet_input: Optional[NNetInput] = None
        self.nnet_name: str = nnet_name
        self.nnet_kwargs: Dict[str, Any] = nnet_kwargs
        self.q_fix: bool = q_fix
        self.out_dim: int = out_dim

    def get_nnet(self) -> HeurNNet:
        nnet_params: Dict = self.nnet_kwargs.copy()
        nnet_params['nnet_input'] = self._get_nnet_input()
        nnet_params['q_fix'] = self.q_fix
        nnet_params['out_dim'] = self.out_dim
        return heuristic_factory.build_class(self.nnet_name, nnet_params)

    def _get_nnet_input(self) -> NNetInput:
        if self.nnet_input is None:
            self.nnet_input = get_nnet_input_t(self.nnet_input_name)(domain=self.domain)
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__


class PolicyNNetParFacClass(PolicyNNetPar, ABC):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], nnet_name: str, nnet_kwargs: Dict[str, Any]):
        self.domain: Domain = domain
        self.nnet_input_name: Tuple[str, str] = nnet_input_name
        self.nnet_input: Optional[PolicyNNetIn] = None
        self.nnet_name: str = nnet_name
        self.nnet_kwargs: Dict[str, Any] = nnet_kwargs

    def get_nnet(self) -> PolicyNNet:
        nnet_params: Dict = self.nnet_kwargs.copy()
        nnet_params['nnet_input'] = self._get_nnet_input()
        return policy_factory.build_class(self.nnet_name, nnet_params)

    def _get_nnet_input(self) -> PolicyNNetIn:
        if self.nnet_input is None:
            nnet_input: NNetInput = get_nnet_input_t(self.nnet_input_name)(domain=self.domain)
            assert isinstance(nnet_input, PolicyNNetIn)
            self.nnet_input = nnet_input
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__


class HeurNNetParVConcrete(HeurNNetParV, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], nnet_name: str,
                 nnet_kwargs: Dict[str, Any]):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, nnet_name, nnet_kwargs, False, 1)

    def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals)

    def _get_nnet_input(self) -> StateGoalIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalIn)
        return nnet_input


class HeurNNetParQFixOutConcrete(HeurNNetParQFixOut, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], nnet_name: str, nnet_kwargs: Dict[str, Any], out_dim: int):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, nnet_name, nnet_kwargs, True, out_dim)

    def _to_np_fixed_acts(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions_l)

    def _get_nnet_input(self) -> StateGoalActFixIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActFixIn)
        return nnet_input


class HeurNNetParQActInConcrete(HeurNNetParQIn, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str],
                 nnet_name: str, nnet_kwargs: Dict[str, Any]):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, nnet_name, nnet_kwargs, False, 1)

    def _to_np_one_act(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def _get_nnet_input(self) -> StateGoalActIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActIn)
        return nnet_input


class PolicyNNetParConcrete(PolicyNNetParFacClass):
    def to_np_fn(self, states: List[State], goals: List[Goal]) -> List[NDArray[Any]]:
        return self._get_nnet_input().to_np_fn(states, goals)

    def to_np_train(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray[Any]]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def _nnet_out_to_actions(self, nnet_out: NDArray[np.float64]) -> List[Action]:
        return self._get_nnet_input().nnet_out_to_actions(nnet_out)
