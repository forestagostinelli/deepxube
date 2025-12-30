from abc import ABC
from typing import Dict, Tuple, Type, List, Any, Optional, Callable

from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed
from deepxube.base.nnet_input import NNetInput, StateGoalIn, StateGoalActFixIn, StateGoalActIn
from deepxube.base.heuristic import (HeurNNet, HeurNNetPar, HeurNNetParV, HeurNNetParQIn, HeurNNetParQFixOut,
                                     HeurNNetParser)

from numpy.typing import NDArray
import logging

from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t

_heur_nnet_registry: Dict[str, Type[HeurNNet]] = {}

_heur_nnet_parser_registry: Dict[str, Type[HeurNNetParser]] = {}


def register_heur_nnet(heur_nnet_name: str) -> Callable[[Type[HeurNNet]], Type[HeurNNet]]:
    def deco(cls: Type[HeurNNet]) -> Type[HeurNNet]:
        if heur_nnet_name in _heur_nnet_registry.keys():
            raise ValueError(f"{heur_nnet_name!r} already registered for heur_nnet")
        _heur_nnet_registry[heur_nnet_name] = cls
        return cls
    return deco


def register_heur_nnet_parser(heur_nnet_parser_name: str) -> Callable[[Type[HeurNNetParser]], Type[HeurNNetParser]]:
    def deco(cls: Type[HeurNNetParser]) -> Type[HeurNNetParser]:
        if heur_nnet_parser_name in _heur_nnet_parser_registry.keys():
            raise ValueError(f"{heur_nnet_parser_name!r} already registered for heur_nnet")
        _heur_nnet_parser_registry[heur_nnet_parser_name] = cls
        return cls
    return deco


def get_all_heur_nnet_names() -> List[str]:
    return list(_heur_nnet_registry.keys())


def get_heur_nnet_type(name: str) -> Type[HeurNNet]:
    try:
        return _heur_nnet_registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown heur_nnet {name!r}. Available: {sorted(_heur_nnet_registry)}"
        )


def get_heur_nnet_parser(heur_nnet_name: str) -> Optional[HeurNNetParser]:
    if heur_nnet_name in _heur_nnet_parser_registry.keys():
        cls_parser: Type[HeurNNetParser] = _heur_nnet_parser_registry[heur_nnet_name]
        parser: HeurNNetParser = cls_parser()
        return parser
    else:
        return None


def get_heur_nnet_kwargs(heur_nnet_name: str, args_str: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict()
    parser: Optional[HeurNNetParser] = get_heur_nnet_parser(heur_nnet_name)
    if (parser is not None) and (args_str is not None):
        try:
            kwargs = parser.parse(args_str)
        except Exception as e:
            logging.exception(f"Error occurred: {e}")
            raise ValueError(f"Error parsing {args_str} for heur_nnet {heur_nnet_name!r}. "
                             f"Help:\n{parser.help()}")
    else:
        assert args_str is None, f"No parser for heur_nnet {heur_nnet_name}, however, args given are {args_str}"
    return kwargs


def build_heur_nnet(heur_nnet_name: str, kwargs: Dict[str, Any]) -> HeurNNet:
    cls: Type[HeurNNet] = get_heur_nnet_type(heur_nnet_name)
    return cls(**kwargs)


def build_heur_nnet_par(domain: Domain, domain_name: str, heur_nnet_mod_name: str,
                        heur_nnet_mod_kwargs: Dict[str, Any], heur_type: str) -> HeurNNetPar:
    nnet_input_t: Type[NNetInput] = get_heur_nnet_type(heur_nnet_mod_name).nnet_input_type()
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


class HeurNNetParConcrete(HeurNNetPar, ABC):
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
        return build_heur_nnet(self.heur_nnet_name, heur_nnet_params)

    def _get_nnet_input(self) -> NNetInput:
        if self.nnet_input is None:
            self.nnet_input = get_nnet_input_t(self.nnet_input_name)(domain=self.domain)
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__


class HeurNNetParVConcrete(HeurNNetParV[State, Goal], HeurNNetParConcrete):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str], heur_nnet_name: str,
                 heur_nnet_kwargs: Dict[str, Any]):
        HeurNNetParConcrete.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, False, 1)

    def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals)

    def _get_nnet_input(self) -> StateGoalIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalIn)
        return nnet_input


class HeurNNetParQFixOutConcrete(HeurNNetParQFixOut[State, Action, Goal], HeurNNetParConcrete):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str],
                 heur_nnet_name: str, heur_nnet_kwargs: Dict[str, Any], out_dim: int):
        HeurNNetParConcrete.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, True, out_dim)

    def _to_np_fixed_acts(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions_l)

    def _get_nnet_input(self) -> StateGoalActFixIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActFixIn)
        return nnet_input


class HeurNNetParQActInConcrete(HeurNNetParQIn[State, Action, Goal], HeurNNetParConcrete):
    def __init__(self, domain: Domain, nnet_input_name: Tuple[str, str],
                 heur_nnet_name: str, heur_nnet_kwargs: Dict[str, Any]):
        HeurNNetParConcrete.__init__(self, domain, nnet_input_name, heur_nnet_name, heur_nnet_kwargs, False, 1)

    def _to_np_one_act(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def _get_nnet_input(self) -> StateGoalActIn:
        nnet_input: NNetInput = super()._get_nnet_input()
        assert isinstance(nnet_input, StateGoalActIn)
        return nnet_input
