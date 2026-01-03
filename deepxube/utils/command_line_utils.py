from typing import Tuple, Optional, List, Dict, Any

from deepxube.base.domain import Domain
from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.pathfinding import PathFind

from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.heuristic_factory import heuristic_factory, build_heur_nnet_par
from deepxube.factories.pathfinding_factory import pathfinding_factory


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
    domain_kwargs: Dict[str, Any] = domain_factory.get_kwargs(domain_name, domain_args)
    return domain_factory.build_class(domain_name, domain_kwargs), domain_name


def get_heur_nnet_par_from_arg(domain: Domain, domain_name: str, heur: str, heur_type: str) -> Tuple[HeurNNetPar, str]:
    heur_module_name, heur_module_args = get_name_args(heur)
    heuristic_factory.get_type(heur_module_name)  # to ensure existence
    heur_module_kwargs: Dict[str, Any] = heuristic_factory.get_kwargs(heur_module_name, heur_module_args)
    heur_nnet_par: HeurNNetPar = build_heur_nnet_par(domain, domain_name, heur_module_name, heur_module_kwargs, heur_type)
    return heur_nnet_par, heur_module_name


def get_pathfind_name_kwargs(pathfind: str) -> Tuple[str, Dict[str, Any]]:
    name, args_str = get_name_args(pathfind)
    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(name, args_str)
    return name, pathfind_kwargs


def get_pathfind_from_arg(pathfind: str, domain: Domain) -> Tuple[PathFind, str]:
    name, args_str = get_name_args(pathfind)
    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(name, args_str)
    pathfind_kwargs["domain"] = domain
    return pathfinding_factory.build_class(name, pathfind_kwargs), name
