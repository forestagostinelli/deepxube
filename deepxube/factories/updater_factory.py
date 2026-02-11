from typing import Dict, Any, Type

from deepxube.base.domain import Domain, GoalSampleableFromState, ActsEnum
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ
from deepxube.base.pathfinding import PathFind, PathFindHeur, PathFindSup
from deepxube.base.updater import UpdateHeur, UpdateHeurRL, UpdateHeurSup, UpArgs, UpHeurArgs
from deepxube.updaters.updater_v_rl import UpdateHeurVRL
from deepxube.updaters.updater_q_rl import UpdateHeurQRLKeepGoal, UpdateHeurQRLHER
from deepxube.updaters.updater_v_sup import UpdateHeurVSup
from deepxube.updaters.updater_q_sup import UpdateHeurQSup
from deepxube.factories.pathfinding_factory import pathfinding_factory


def get_updater(domain: Domain, heur_nnet_par: HeurNNetPar, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs,
                up_heur_args: UpHeurArgs, her: bool) -> UpdateHeur:

    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    if issubclass(pathfind_t, PathFindHeur):
        assert isinstance(domain, ActsEnum), (f"No updaters avaialable when doing reinforcement learning updates for domains that are not a suclass of "
                                              f"{ActsEnum}")
        updater_rl: UpdateHeurRL
        if isinstance(heur_nnet_par, HeurNNetParV):
            updater_rl = UpdateHeurVRL(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
        elif isinstance(heur_nnet_par, HeurNNetParQ):
            if not her:
                updater_rl = UpdateHeurQRLKeepGoal(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
            else:
                assert isinstance(domain, GoalSampleableFromState)
                updater_rl = UpdateHeurQRLHER(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
        else:
            raise ValueError(f"No update implementation for {heur_nnet_par}")
        return updater_rl
    elif issubclass(pathfind_t, PathFindSup):
        assert her is False, "No hindsight experience replay (HER) for supervised learning"
        updater_sup: UpdateHeurSup
        if isinstance(heur_nnet_par, HeurNNetParV):
            updater_sup = UpdateHeurVSup(domain, pathfind_name, pathfind_kwargs, up_args)
        elif isinstance(heur_nnet_par, HeurNNetParQ):
            updater_sup = UpdateHeurQSup(domain, pathfind_name, pathfind_kwargs, up_args)
        else:
            raise ValueError(f"No update implementation for {heur_nnet_par}")
        return updater_sup
    else:
        raise ValueError(f"Unknown update method for {pathfind_t}")
