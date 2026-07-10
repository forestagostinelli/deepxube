from typing import List, Type, Optional, Any, Tuple, Dict
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.pathfind_fns import PFNs
from deepxube.base.pathfinding import PathFind
from deepxube.base.pathfind_fns import HeurFn, HeurVFn, HeurQFn, PolicyFn, PFNsHeurV, PFNsHeurQ, PFNsPolicy, PFNsHeurVPolicy, PFNsHeurQPolicy
from deepxube.utils.command_line_utils import get_name_args

pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")


def get_pathfind_functions(pathfind_name: str, heur_fn: Optional[HeurFn], policy_fn: Optional[PolicyFn]) -> Any:
    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    functions_type: Any = pathfind_t.functions_type()
    if functions_type is PFNsHeurV:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurVFn)
        return PFNsHeurV(heur_fn)
    elif functions_type is PFNsHeurQ:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurQFn)
        return PFNsHeurQ(heur_fn)
    elif functions_type is PFNsPolicy:
        assert policy_fn is not None
        return PFNsPolicy(policy_fn)
    elif functions_type is PFNsHeurVPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurVFn)
        assert policy_fn is not None
        return PFNsHeurVPolicy(heur_fn, policy_fn)
    elif functions_type is PFNsHeurQPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurQFn)
        assert policy_fn is not None
        return PFNsHeurQPolicy(heur_fn, policy_fn)
    elif functions_type is Any:
        return None
    else:
        raise ValueError(f"Unknown Function type {functions_type}")


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
    names: List[str] = pathfinding_factory.get_all_class_names()
    compat_names: List[str] = []
    for name in names:
        if not name.startswith(pathfind_name_pre):
            continue

        if pathfinding_factory.get_type(name).is_compat(domain, type(pathfind_fns)):
            compat_names.append(name)

    if len(compat_names) == 0:
        raise ValueError(f"Could not find any compatable PathFind for Domain {domain} and Functions {PFNs} for PathFind name: {pathfind_name_pre}.")

    # TODO if > 1 find more specific one in terms of function

    if len(compat_names) > 1:
        raise ValueError(f"More than 1 compatable PathFind class for Domain {domain} and Functions {PFNs} for PathFind name: {pathfind_name_pre}: "
                         f"{compat_names}.")

    assert len(compat_names) == 1
    pathfind_name: str = compat_names[0]

    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(pathfind_name, args_str)
    pathfind_kwargs["domain"] = domain
    pathfind_kwargs["functions"] = pathfind_fns
    return pathfinding_factory.build_class(pathfind_name, pathfind_kwargs), pathfind_name
