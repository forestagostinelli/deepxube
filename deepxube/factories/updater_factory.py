from typing import Type, List, Tuple, Dict, Any

from deepxube.nnet.nnet_utils import NNetParRunner
from deepxube.utils.command_line_utils import get_name_args
from deepxube.base.domain import Domain
from deepxube.base.pathfinding import PathFind
from deepxube.base.updater import Update
from deepxube.base.factory import Factory

updater_factory: Factory[Update] = Factory[Update]("Update")


def get_domain_compat_updater_names(domain_t: Type[Domain]) -> List[str]:
    names: List[str] = []
    for name in updater_factory.get_all_class_names():
        class_t: Type[Update] = updater_factory.get_type(name)
        if issubclass(domain_t, class_t.domain_type()):
            names.append(name)

    return names


def get_pathfind_compat_updater_names(pathfind_t: Type[PathFind]) -> List[str]:
    names: List[str] = []
    for name in updater_factory.get_all_class_names():
        class_t: Type[Update] = updater_factory.get_type(name)
        if issubclass(pathfind_t, class_t.pathfind_type()) and (pathfind_t.functions_type() is class_t.functions_type()):
            names.append(name)

    return names


def get_updater_from_args(domain: Domain, pathfind_arg: str, nnet_par_run_dict: Dict[str, NNetParRunner], updater_arg: str) -> Tuple[Update, str]:
    updater_name, args_str = get_name_args(updater_arg)
    updater_kwargs: Dict[str, Any] = updater_factory.get_kwargs(updater_name, args_str)
    updater_kwargs["domain"] = domain
    updater_kwargs["pathfind_arg"] = pathfind_arg
    updater_kwargs["nnet_par_run_dict"] = nnet_par_run_dict
    return updater_factory.build_class(updater_name, updater_kwargs), updater_name
