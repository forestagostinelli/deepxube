from typing import List, Type, Optional, Any, Tuple, Dict
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.pathfinding import PathFind
from deepxube.base.nnet_fn import HeurFn, HeurVFn, HeurQFn, PolicyFn, FNsHeurV, FNsHeurQ, FNsPolicy, FNsHeurVPolicy, FNsHeurQPolicy
from deepxube.utils.command_line_utils import get_name_args

pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")


def get_pathfind_functions(pathfind_name: str, heur_fn: Optional[HeurFn], policy_fn: Optional[PolicyFn]) -> Any:
    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    functions_type: Any = pathfind_t.functions_type()
    if functions_type is FNsHeurV:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurVFn)
        return FNsHeurV(heur_fn)
    elif functions_type is FNsHeurQ:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurQFn)
        return FNsHeurQ(heur_fn)
    elif functions_type is FNsPolicy:
        assert policy_fn is not None
        return FNsPolicy(policy_fn)
    elif functions_type is FNsHeurVPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurVFn)
        assert policy_fn is not None
        return FNsHeurVPolicy(heur_fn, policy_fn)
    elif functions_type is FNsHeurQPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurQFn)
        assert policy_fn is not None
        return FNsHeurQPolicy(heur_fn, policy_fn)
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


def get_pathfind_from_arg(domain: Domain, functions: Any, pathfind_arg: str) -> Tuple[PathFind, str]:
    pathfind_name, args_str = get_name_args(pathfind_arg)
    pathfind_kwargs: Dict[str, Any] = pathfinding_factory.get_kwargs(pathfind_name, args_str)
    pathfind_kwargs["domain"] = domain
    pathfind_kwargs["functions"] = functions
    return pathfinding_factory.build_class(pathfind_name, pathfind_kwargs), pathfind_name
