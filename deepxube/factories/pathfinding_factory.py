from typing import List, Type, Optional, Any
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.heuristic import HeurFn, HeurFnV, HeurFnQ, PolicyFn
from deepxube.base.pathfinding import PathFind, FNsHeurV, FNsHeurQ, FNsHeurVPolicy, FNsHeurQPolicy


pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")


def get_pathfind_functions(pathfind_name: str, heur_fn: Optional[HeurFn], policy_fn: Optional[PolicyFn]) -> Any:
    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    functions_type: Any = pathfind_t.functions_type()
    if functions_type is FNsHeurV:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurFnV)
        return FNsHeurV(heur_fn)
    elif functions_type is FNsHeurQ:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurFnQ)
        return FNsHeurQ(heur_fn)
    elif functions_type is FNsHeurVPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurFnV)
        assert policy_fn is not None
        return FNsHeurVPolicy(heur_fn, policy_fn)
    elif functions_type is FNsHeurQPolicy:
        assert (heur_fn is not None) and isinstance(heur_fn, HeurFnQ)
        assert policy_fn is not None
        return FNsHeurQPolicy(heur_fn, policy_fn)
    else:
        raise ValueError(f"Unknown Function type {functions_type}")


def get_domain_compat_pathfind_names(domain_t: Type[Domain]) -> List[str]:
    pathfind_names: List[str] = []
    for pathfind_name in pathfinding_factory.get_all_class_names():
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
        if issubclass(domain_t, pathfind_t.domain_type()):
            pathfind_names.append(pathfind_name)

    return pathfind_names
