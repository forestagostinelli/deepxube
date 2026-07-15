from typing import Type, List, Tuple, Dict, Any, Optional

from deepxube.utils.command_line_utils import get_name_args
from deepxube.base.domain import Domain
from deepxube.base.pathfind_fns import PFNs, UFNs
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
        if issubclass(pathfind_t, class_t.pathfind_type()) and (pathfind_t.pathfind_functions_type() is class_t.pathfind_functions_type()):
            names.append(name)

    return names


def get_updater_from_args(domain: Domain, pathfind: PathFind, pathfind_name_args: str, updater_fns: UFNs, updater_name_args: str) -> Tuple[Update, str]:
    updater_name_pre, args_str = get_name_args(updater_name_args)
    pathfind_t: Type[PathFind] = type(pathfind)
    pathfind_fns_t: Type[PFNs] = pathfind.pathfind_functions_type()
    updater_fns_t: Type[UFNs] = type(updater_fns)

    names: List[str] = updater_factory.get_all_class_names()
    compat_names: List[str] = []
    incompat_reasons: List[str] = []
    for name in names:
        if not name.startswith(updater_name_pre):
            continue

        incompat_reason: Optional[str] = updater_factory.get_type(name).get_incompat_reason(domain, pathfind_fns_t, pathfind_t, updater_fns_t)
        if incompat_reason is not None:
            incompat_reasons.append(incompat_reason + f" (Updater name: {name})")
        else:
            compat_names.append(name)

    if len(compat_names) == 0:
        incompat_reasons_str: str = '\n'.join(incompat_reasons)
        raise ValueError(f"Could not find any compatable Updater for Domain {domain}, Functions type {pathfind_fns_t}, PathFind {pathfind} for Updater "
                         f"name: {updater_name_pre}.\nIncompatibility reasons:\n{incompat_reasons_str}")

    # TODO if > 1 find more specific one in terms of function and pathfind

    if len(compat_names) > 1:
        raise ValueError(f"More then 1 compatable Updater for Domain {domain}, Functions type {pathfind_fns_t}, PathFind {pathfind} for Updater "
                         f"name: {updater_name_pre}: {compat_names}.")

    assert len(compat_names) == 1
    updater_name: str = compat_names[0]

    updater_kwargs: Dict[str, Any] = updater_factory.get_kwargs(updater_name, args_str)
    updater_kwargs["domain"] = domain
    updater_kwargs["pathfind_name_args"] = pathfind_name_args
    updater_kwargs["up_fns"] = updater_fns
    return updater_factory.build_class(updater_name, updater_kwargs), updater_name
