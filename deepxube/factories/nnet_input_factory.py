from typing import Dict, Type, Callable, List, Any

from deepxube.utils.command_line_utils import get_name_args
from deepxube.base.nnet_input import NNetInput, DynamicNNetInput
from deepxube.base.domain import Domain
from deepxube.base.factory import Factory, Parser
from deepxube.factories.domain_factory import domain_factory


_nnet_input_registry: Dict[str, Factory[NNetInput]] = {}


def register_nnet_input(domain_name: str, nnet_input_name: str) -> Callable[[Type[NNetInput]], Type[NNetInput]]:
    def deco(cls: Type[NNetInput]) -> Type[NNetInput]:
        if domain_name not in _nnet_input_registry.keys():
            _nnet_input_registry[domain_name] = Factory[NNetInput](f"{domain_name} NNetInput")
        _nnet_input_registry[domain_name].register_class(nnet_input_name)(cls)
        return cls
    return deco


def register_nnet_input_parser(domain_name: str, nnet_input_name: str) -> Callable[[Type[Parser]], Type[Parser]]:
    def deco(cls: Type[Parser]) -> Type[Parser]:
        if domain_name not in _nnet_input_registry.keys():
            _nnet_input_registry[domain_name] = Factory[NNetInput](f"{domain_name} NNetInput")
        _nnet_input_registry[domain_name].register_parser(nnet_input_name)(cls)
        return cls

    return deco


def get_domain_nnet_input_keys(domain_name: str) -> List[str]:
    return _nnet_input_registry[domain_name].get_all_class_names()


def get_nnet_input_t(domain_name: str, nnet_input_name: str) -> Type[NNetInput]:
    return _nnet_input_registry[domain_name].get_type(nnet_input_name)


def register_nnet_input_dynamic() -> None:
    for domain_name in domain_factory.get_all_class_names():
        domain_t: Type[Domain] = domain_factory.get_type(domain_name)
        if issubclass(domain_t, DynamicNNetInput):
            nnet_input_t_dict: Dict[str, Type[NNetInput]] = domain_t.get_dynamic_nnet_inputs()
            for nnet_input_name, nnet_input_t in nnet_input_t_dict.items():
                register_nnet_input(domain_name, f"{nnet_input_name}")(nnet_input_t)


def get_nnet_input_from_arg(domain: Domain, domain_name: str, nnet_input_name_args: str) -> NNetInput:
    nnet_input_name, nnet_input_args = get_name_args(nnet_input_name_args)
    nnet_input_factory: Factory[NNetInput] = _nnet_input_registry[domain_name]
    nnet_input_kwargs: Dict[str, Any] = nnet_input_factory.get_kwargs(nnet_input_name, nnet_input_args)
    nnet_input_kwargs["domain"] = domain
    return nnet_input_factory.build_class(nnet_input_name, nnet_input_kwargs)
