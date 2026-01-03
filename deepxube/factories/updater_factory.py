from typing import Dict, Any, Type
from deepxube.base.domain import Domain, ActsEnum
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ
from deepxube.base.pathfinding import PathFind, PathFindHeur
from deepxube.base.updater import UpdateHeurRL, UpdateHeurRLV, UpArgs, UpHeurArgs, UpdateHeurRLQEnum
from deepxube.factories.pathfinding_factory import pathfinding_factory


def get_updater(domain: Domain, heur_nnet_par: HeurNNetPar, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs,
                up_heur_args: UpHeurArgs) -> UpdateHeurRL:

    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    if issubclass(pathfind_t, PathFindHeur):
        assert isinstance(domain, ActsEnum), (f"No updaters avaialable when doing reinforcement learning updates for domains that are not a suclass of "
                                              f"{ActsEnum}")

        if isinstance(heur_nnet_par, HeurNNetParV):
            return UpdateHeurRLV(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args, heur_nnet_par)
        elif isinstance(heur_nnet_par, HeurNNetParQ):
            return UpdateHeurRLQEnum(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args, heur_nnet_par)
        else:
            raise ValueError(f"No update implementation for {heur_nnet_par}")
    else:
        raise NotImplementedError
