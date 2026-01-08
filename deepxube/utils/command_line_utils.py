from typing import Tuple, Optional, List, Dict, Any, Type

from deepxube.base.domain import Domain
from deepxube.base.heuristic import HeurNNetPar
from deepxube.base.pathfinding import PathFind, PathFindHeur, PathFindVHeur, PathFindQHeur

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


def get_pathfind_from_arg(domain: Domain, heur_type: str, pathfind: str) -> Tuple[PathFind, str]:
    pathfind_name, args_str = get_name_args(pathfind)

    # check heur type
    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    if issubclass(pathfind_t, PathFindHeur):
        if issubclass(pathfind_t, PathFindVHeur):
            assert heur_type.upper() == "V", f"must use a V heur_type for pathfinding algorithm {pathfind_name, pathfind_t}"
        elif issubclass(pathfind_t, PathFindQHeur):
            assert heur_type.upper() in {"QFIX", "QIN"}, f"must use a QFix or QIn heur_types for pathfinding algorithm {pathfind_name, pathfind_t}"
        else:
            raise ValueError(f"Unknown subclass of PathFindHeur {pathfind_t}")

    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(pathfind_name, args_str)
    pathfind_kwargs["domain"] = domain
    return pathfinding_factory.build_class(pathfind_name, pathfind_kwargs), pathfind_name
