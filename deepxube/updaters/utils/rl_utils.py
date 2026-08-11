from typing import List, Any, cast
from numpy.typing import NDArray
import numpy as np

from deepxube.utils import misc_utils
from deepxube.base.domain import State, Goal
from deepxube.base.pathfind_fns import HeurVFn


def vi_backup(is_solved: List[bool], goals: List[Goal], contexts: List[Any], states_exp: List[List[State]], tcs_l: List[List[float]],
              heur_fn: HeurVFn) -> List[float]:
    # get flat and repeat goals/contexts for number of state_exp
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    goals_flat: List[Goal] = []
    contexts_flat: List[Any] = []
    for goal, context, state_exp in zip(goals, contexts, states_exp, strict=True):
        goals_flat.extend([goal] * len(state_exp))
        contexts_flat.extend([context] * len(state_exp))

    # get ctg of expanded states
    ctg_next: List[float] = heur_fn(states_exp_flat, goals_flat, contexts_flat)

    # backup cost-to-go
    ctg_next_p_tc: NDArray = np.concatenate(tcs_l, axis=0) + np.array(ctg_next)
    ctg_next_p_tc_l: List[NDArray] = np.split(ctg_next_p_tc, split_idxs)

    ctgs_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)
    ctgs_backup_l: List[float] = cast(List[float], ctgs_backup.tolist())

    return ctgs_backup_l
