from typing import List, Type
from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.base.pathfinding import PathFind


pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")


def get_domain_compat_pathfind_names(domain_t: Type[Domain]) -> List[str]:
    pathfind_names: List[str] = []
    for pathfind_name in pathfinding_factory.get_all_class_names():
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
        if issubclass(domain_t, pathfind_t.domain_type()):
            pathfind_names.append(pathfind_name)

    return pathfind_names
