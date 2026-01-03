from abc import ABC
from typing import Dict, Tuple, Type, List, Any, Optional

from deepxube.base.factory import Factory
from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed
from deepxube.base.nnet_input import NNetInput, StateGoalIn, StateGoalActFixIn, StateGoalActIn
from deepxube.base.heuristic import HeurNNet, HeurNNetPar, HeurNNetParV, HeurNNetParQIn, HeurNNetParQFixOut

from numpy.typing import NDArray

from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t


heuristic_factory: Factory[HeurNNet] = Factory[HeurNNet]("HeurNNet")


def build_heur_nnet_par(domain: Domain, domain_name: str, heur_nnet_mod_name: str,
                        heur_nnet_mod_kwargs: Dict[str, Any], heur_type: str) -> HeurNNetPar:
    nnet_input_t: Type[NNetInput] = heuristic_factory.get_type(heur_nnet_mod_name).nnet_input_type()
    nnet_input_domain_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)

    for nnet_input_domain_key in nnet_input_domain_keys:
        nnet_input_cls: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_key)
        if heur_type.upper() == "V":
            if issubclass(nnet_input_cls, StateGoalIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParVConcrete(domain, nnet_input_domain_key, heur_nnet_mod_name, heur_nnet_mod_kwargs)
        elif heur_type.upper() == "QFIX":
            assert isinstance(domain, ActsEnumFixed)
            if issubclass(nnet_input_cls, StateGoalActFixIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParQFixOutConcrete(domain, nnet_input_domain_key, heur_nnet_mod_name, heur_nnet_mod_kwargs,
                                                  domain.get_num_acts())
        elif heur_type.upper() == "QIN":
            if issubclass(nnet_input_cls, StateGoalActIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurNNetParQActInConcrete(domain, nnet_input_domain_key, heur_nnet_mod_name, heur_nnet_mod_kwargs)

        else:
            raise ValueError(f"Unknown heur type {heur_type}")
    raise ValueError(f"Cannot build heur nnet for domain: {domain_name}, heur type {heur_type}, and "
                     f"nnet_input type {nnet_input_t}.\nNNet inputs checked: {nnet_input_domain_keys}")


class HeurNNetParFacClass(HeurNNetPar, ABC):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], heur_nnet_name: str,
                 heur_nnet_kwargs: Dict[str, Any], q_fix: bool, out_dim: int):
        self.domain: Domain = domain
        self.nnet_input_name: Tuple[str, str] = nnet_input_name
        self.nnet_input: Optional[NNetInput] = None
        self.heur_nnet_name: str = heur_nnet_name
        self.heur_nnet_kwargs: Dict[str, Any] = heur_nnet_kwargs
        self.q_fix: bool = q_fix
        self.out_dim: int = out_dim

    def get_nnet(self) -> HeurNNet:
        heur_nnet_params: Dict = self.heur_nnet_kwargs.copy()
        heur_nnet_params['nnet_input'] = self._get_nnet_input()
        heur_nnet_params['q_fix'] = self.q_fix
        heur_nnet_params['out_dim'] = self.out_dim
        return heuristic_factory.build_class(self.heur_nnet_name, heur_nnet_params)

    def _get_nnet_input(self) -> NNetInput:
        if self.nnet_input is None:
            self.nnet_input = get_nnet_input_t(self.nnet_input_name)(domain=self.domain)
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__


class HeurNNetParVConcrete(HeurNNetParV, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], heur_nnet_name: str,
                 heur_nnet_kwargs: Dict[str, Any]):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, False, 1)

    def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals)

    def _get_nnet_input(self) -> StateGoalIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalIn)
        return nnet_input


class HeurNNetParQFixOutConcrete(HeurNNetParQFixOut, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str],
                 heur_nnet_name: str, heur_nnet_kwargs: Dict[str, Any], out_dim: int):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, True, out_dim)

    def _to_np_fixed_acts(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions_l)

    def _get_nnet_input(self) -> StateGoalActFixIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActFixIn)
        return nnet_input


class HeurNNetParQActInConcrete(HeurNNetParQIn, HeurNNetParFacClass):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str],
                 heur_nnet_name: str, heur_nnet_kwargs: Dict[str, Any]):
        HeurNNetParFacClass.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, False, 1)

    def _to_np_one_act(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def _get_nnet_input(self) -> StateGoalActIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActIn)
        return nnet_input
