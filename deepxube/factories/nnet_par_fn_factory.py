from typing import Dict, Any, Type, List, Tuple

from deepxube.base.domain import Domain, ActsEnumFixed
from deepxube.base.nnet_input import NNetInput, StateGoalIn, StateGoalActFixIn, StateGoalActIn, PolicyNNetIn
from deepxube.base.nnet_par_fn import HeurNNetPar, HeurVNNetPar, HeurQNNetParFixOut, HeurQNNetParIn, PolicyNNetPar
from deepxube.factories.heuristic_factory import deepxube_nnet_factory
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.utils.command_line_utils import get_name_args


def build_heur_nnet_par(domain: Domain, domain_name: str, nnet_name: str, nnet_kwargs: Dict[str, Any], heur_type: str) -> HeurNNetPar:
    nnet_input_t: Type[NNetInput] = deepxube_nnet_factory.get_type(nnet_name).nnet_input_type()
    nnet_input_domain_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)

    for nnet_input_domain_key in nnet_input_domain_keys:
        nnet_input_cls: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_key)
        if heur_type.upper() == "V":
            if issubclass(nnet_input_cls, StateGoalIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurVNNetPar(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)
        elif heur_type.upper() == "QFIX":
            assert isinstance(domain, ActsEnumFixed)
            if issubclass(nnet_input_cls, StateGoalActFixIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurQNNetParFixOut(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)
        elif heur_type.upper() == "QIN":
            if issubclass(nnet_input_cls, StateGoalActIn) and issubclass(nnet_input_cls, nnet_input_t):
                return HeurQNNetParIn(domain, nnet_input_domain_key, nnet_name, nnet_kwargs)
        else:
            raise ValueError(f"Unknown heur type {heur_type}")
    raise ValueError(f"Cannot build heur nnet for domain: {domain_name}, heur type {heur_type}, and "
                     f"nnet_input type {nnet_input_t}.\nNNet inputs checked: {nnet_input_domain_keys}")


def build_policy_nnet_par(domain: Domain, domain_name: str, nnet_name: str, nnet_kwargs: Dict[str, Any], num_samp: int) -> PolicyNNetPar:
    nnet_input_t: Type[NNetInput] = deepxube_nnet_factory.get_type(nnet_name).nnet_input_type()
    nnet_input_domain_keys: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
    for nnet_input_domain_key in nnet_input_domain_keys:
        nnet_input_cls: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_key)
        if issubclass(nnet_input_cls, PolicyNNetIn) and issubclass(nnet_input_cls, nnet_input_t):
            return PolicyNNetPar(domain, nnet_input_domain_key, nnet_name, nnet_kwargs, num_samp=num_samp)

    raise ValueError(f"Cannot build policy nnet for domain: {domain_name}, and "
                     f"nnet_input type {nnet_input_t}.\nNNet inputs checked: {nnet_input_domain_keys}")


def get_heur_nnet_par_from_arg(domain: Domain, domain_name: str, heur: str, heur_type: str) -> Tuple[HeurNNetPar, str]:
    nnet_name, nnet_args = get_name_args(heur)
    deepxube_nnet_factory.get_type(nnet_name)  # to ensure existence
    nnet_kwargs: Dict[str, Any] = deepxube_nnet_factory.get_kwargs(nnet_name, nnet_args)
    nnet_par: HeurNNetPar = build_heur_nnet_par(domain, domain_name, nnet_name, nnet_kwargs, heur_type)
    return nnet_par, nnet_name


def get_policy_nnet_par_from_arg(domain: Domain, domain_name: str, policy: str, num_samp: int) -> Tuple[PolicyNNetPar, str]:
    nnet_name, nnet_args = get_name_args(policy)
    deepxube_nnet_factory.get_type(nnet_name)  # to ensure existence
    nnet_kwargs: Dict[str, Any] = deepxube_nnet_factory.get_kwargs(nnet_name, nnet_args)
    nnet_par: PolicyNNetPar = build_policy_nnet_par(domain, domain_name, nnet_name, nnet_kwargs, num_samp)
    return nnet_par, nnet_name
