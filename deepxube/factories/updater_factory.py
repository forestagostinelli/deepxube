
from deepxube.base.domain import Domain, ActsEnum
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ
from deepxube.base.updater import UpdateHeur, UpArgs, UpHeurArgs
from deepxube.updater.updaters import UpGraphSearchArgs, UpGreedyPolicyArgs
from deepxube.updater.updaters import (UpdateHeurRWSupV, UpdateHeurBWASEnum, UpdateHeurBWQSEnum, UpdateHeurGrPolVEnum,
                                       UpdateHeurGrPolQEnum)


def get_updater(domain: Domain, heur_nnet_par: HeurNNetPar, search_type: str, up_args: UpArgs, up_heur_args: UpHeurArgs,
                up_graphsch_args: UpGraphSearchArgs, up_greedy_args: UpGreedyPolicyArgs) -> UpdateHeur:
    search_type = search_type.upper()
    if search_type == "GRAPH":
        assert isinstance(domain, ActsEnum)
        if isinstance(heur_nnet_par, HeurNNetParV):
            return UpdateHeurBWASEnum(domain, up_args, up_heur_args, up_graphsch_args, heur_nnet_par)
        else:
            assert isinstance(heur_nnet_par, HeurNNetParQ)
            return UpdateHeurBWQSEnum(domain, up_args, up_heur_args, up_graphsch_args, heur_nnet_par)
    elif search_type == "GREEDY":
        assert isinstance(domain, ActsEnum)
        if isinstance(heur_nnet_par, HeurNNetParV):
            return UpdateHeurGrPolVEnum(domain, up_args, up_heur_args, up_greedy_args, heur_nnet_par)
        else:
            assert isinstance(heur_nnet_par, HeurNNetParQ)
            return UpdateHeurGrPolQEnum(domain, up_args, up_heur_args, up_greedy_args, heur_nnet_par)
    elif search_type == "SUP":
        assert isinstance(heur_nnet_par, HeurNNetParV)
        return UpdateHeurRWSupV(domain, up_args, up_heur_args, heur_nnet_par)
    else:
        raise ValueError(f"Unknown search type {search_type}")
