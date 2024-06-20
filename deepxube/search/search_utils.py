from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from deepxube.environments.environment_abstract import Environment, State, Action, Goal
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
import time


def is_valid_soln(state: State, goal: Goal, soln: List[Action], env: Environment) -> bool:
    state_soln: State = state
    for action in soln:
        state_soln = env.next_state([state_soln], [action])[0][0]

    return env.is_solved([state_soln], [goal])[0]


def bellman(states: List[State], goals: List[Goal], heuristic_fn,
            env: Environment,
            times: Optional[Times] = None) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[List[State]],
                                                    List[bool]]:
    if times is None:
        times = Times()

    # expand states
    start_time = time.time()
    states_exp, _, tcs_l = env.expand(states)
    times.record_time("expand", time.time() - start_time)

    # get cost-to-go of expanded states
    start_time = time.time()
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)

    goals_flat = []
    for goal, state_exp in zip(goals, states_exp):
        goals_flat.extend([goal] * len(state_exp))

    ctg_next_flat: NDArray[np.float64] = heuristic_fn(states_exp_flat, goals_flat)
    times.record_time("heur", time.time() - start_time)

    # is solved
    start_time = time.time()
    is_solved = env.is_solved(states, goals)
    times.record_time("is_solved", time.time() - start_time)

    # backup
    start_time = time.time()
    tcs_flat: NDArray[np.float64] = np.hstack(tcs_l)
    ctg_next_p_tc_flat: NDArray[np.float64] = tcs_flat + ctg_next_flat
    ctg_next_p_tc_l: List[NDArray[np.float64]] = np.split(ctg_next_p_tc_flat, split_idxs)

    ctg_backup: NDArray[np.float64] = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)
    times.record_time("backup", time.time() - start_time)

    return ctg_backup, ctg_next_p_tc_l, states_exp, is_solved
