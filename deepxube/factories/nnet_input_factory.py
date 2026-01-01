from typing import Dict, Tuple, Type, Callable, List

from deepxube.base.nnet_input import NNetInput, DynamicNNetInput
from deepxube.base.domain import Domain
from deepxube.factories.domain_factory import get_all_domain_names, get_domain_type


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
    for domain_name in get_all_domain_names():
        domain_t: Type[Domain] = get_domain_type(domain_name)
        if issubclass(domain_t, DynamicNNetInput):
            nnet_input_t_dict: Dict[str, Type[NNetInput]] = domain_t.get_dynamic_nnet_inputs()
            for nnet_input_name, nnet_input_t in nnet_input_t_dict.items():
                register_nnet_input(domain_name, f"{nnet_input_name}")(nnet_input_t)
