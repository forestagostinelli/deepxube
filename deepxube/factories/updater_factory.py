from typing import Dict, Any
from deepxube.base.domain import Domain
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ
from deepxube.base.updater import UpdateHeur, UpdateHeurV, UpdateHeurQ, UpArgs, UpHeurArgs


def get_updater(domain: Domain, heur_nnet_par: HeurNNetPar, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs,
                up_heur_args: UpHeurArgs) -> UpdateHeur:

    if isinstance(heur_nnet_par, HeurNNetParV):
        return UpdateHeurV(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args, heur_nnet_par)
    else:
        raise NotImplementedError
