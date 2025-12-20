from typing import Tuple, Optional, List, Dict, Any

from deepxube.base.domain import Domain
from deepxube.base.heuristic import HeurNNetPar

from deepxube.factories.domain_factory import build_domain, get_domain_kwargs
from deepxube.factories.heuristic_factory import build_heur_nnet, get_heur_module_kwargs


def get_name_args(name_args: str) -> Tuple[str, Optional[str]]:
    name_args_split: List[str] = name_args.split(".", 1)
    name: str = name_args_split[0]
    args: Optional[str]
    if len(name_args_split) == 1:
        args = None
    else:
        assert len(name_args_split) == 2
        args = name_args_split[1]
    return name, args


def get_domain_from_arg(domain: str) -> Tuple[Domain, str]:
    domain_name, domain_args = get_name_args(domain)
    domain_kwargs: Dict[str, Any] = get_domain_kwargs(domain_name, domain_args)
    return build_domain(domain_name, domain_kwargs), domain_name


def get_heur_nnet_from_arg(domain: Domain, domain_name: str, heur: str, heur_type: str) -> Tuple[HeurNNetPar, str]:
    heur_module_name, heur_module_args = get_name_args(heur)
    heur_module_kwargs: Dict[str, Any] = get_heur_module_kwargs(heur_module_name, heur_module_args)
    heur_nnet: HeurNNetPar = build_heur_nnet(domain, domain_name, heur_module_name, heur_module_kwargs, heur_type)
    return heur_nnet, heur_module_name
