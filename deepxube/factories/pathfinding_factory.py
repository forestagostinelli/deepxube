from typing import List, Type, Any, Tuple, Dict, Optional
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.pathfind_fns import PFNs
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


def get_pathfind_from_arg(domain: Domain, pathfind_fns: PFNs, pathfind_name_args: str) -> Tuple[PathFind, str]:
    pathfind_name_pre, args_str = get_name_args(pathfind_name_args)

    pathfind_fns_t: Type[PFNs] = type(pathfind_fns)
    names: List[str] = pathfinding_factory.get_all_class_names()
    compat_names: List[str] = []
    incompat_reasons: List[str] = []
    for name in names:
        if not name.startswith(pathfind_name_pre):
            continue

        incompat_reason: Optional[str] = pathfinding_factory.get_type(name).get_incompat_reason(domain, pathfind_fns_t)
        if incompat_reason is not None:
            incompat_reasons.append(incompat_reason + f" (PathFind name: {name})")
        else:
            compat_names.append(name)

    if len(compat_names) == 0:
        incompat_reasons_str: str = '\n'.join(incompat_reasons)
        raise ValueError(f"Could not find any compatible PathFind for Domain {domain} and Functions {pathfind_fns} for PathFind name: {pathfind_name_pre}.\n"
                         f"Incompatibility reasons:\n{incompat_reasons_str}")

    # TODO if > 1 find more specific one in terms of function

    if len(compat_names) > 1:
        raise ValueError(f"More than 1 compatable PathFind class for Domain {domain} and Functions {pathfind_fns} for PathFind name: {pathfind_name_pre}: "
                         f"{compat_names}.")

    assert len(compat_names) == 1
    pathfind_name: str = compat_names[0]

    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(pathfind_name, args_str)
    pathfind_kwargs["domain"] = domain
    pathfind_kwargs["functions"] = pathfind_fns
    return pathfinding_factory.build_class(pathfind_name, pathfind_kwargs), pathfind_name
