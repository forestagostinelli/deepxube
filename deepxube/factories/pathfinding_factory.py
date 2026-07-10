from typing import List, Type, Any, Tuple, Dict

from deepxube.nnet.nnet_utils import NNetCallable
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.pathfinding import PathFind
from deepxube.utils.command_line_utils import get_name_args

pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")


def get_domain_compat_pathfind_names(domain_t: Type[Domain]) -> List[str]:
    pathfind_names: List[str] = []
    for pathfind_name in pathfinding_factory.get_all_class_names():
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
        if issubclass(domain_t, pathfind_t.domain_type()):
            pathfind_names.append(pathfind_name)

    return pathfind_names


def get_pathfind_name_kwargs(pathfind: str) -> Tuple[str, Dict[str, Any]]:
    name, args_str = get_name_args(pathfind)
    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(name, args_str)
    return name, pathfind_kwargs


def get_pathfind_from_arg(domain: Domain, fns_dict: Dict[str, NNetCallable], pathfind_arg: str) -> Tuple[PathFind, str]:
    pathfind_name, args_str = get_name_args(pathfind_arg)
    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(pathfind_name, args_str)
    pathfind_kwargs["domain"] = domain
    pathfind_kwargs["fns_dict"] = fns_dict
    return pathfinding_factory.build_class(pathfind_name, pathfind_kwargs), pathfind_name
