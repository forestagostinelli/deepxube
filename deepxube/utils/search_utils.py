from typing import List, Tuple, Optional
import numpy as np
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
import time


def is_valid_soln(state: State, goal: Goal, soln: List[int], env: Environment) -> bool:
    state_soln: State = state
    move: int
    for move in soln:
        state_soln = env.next_state([state_soln], [move])[0][0]

    return env.is_solved([state_soln], [goal])[0]


def bellman(states: List[State], goals: List[Goal], heuristic_fn,
            env: Environment, times: Optional[Times] = None) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    if times is None:
        times = Times()

    # expand states
    start_time = time.time()
    states_exp, tcs_l = env.expand(states)
    times.record_time("expand", time.time() - start_time)

    # get cost-to-go of expanded states
    start_time = time.time()
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)

    goals_flat = []
    for goal, state_exp in zip(goals, states_exp):
        goals_flat.extend([goal] * len(state_exp))

    ctg_next_flat: np.ndarray = heuristic_fn(states_exp_flat, goals_flat)
    times.record_time("heur", time.time() - start_time)

    # is solved
    start_time = time.time()
    is_solved = env.is_solved(states, goals)
    times.record_time("is_solved", time.time() - start_time)

    # backup
    start_time = time.time()
    tcs_flat = np.hstack(tcs_l)
    ctg_next_p_tc_flat = tcs_flat + ctg_next_flat
    ctg_next_p_tc_l = np.split(ctg_next_p_tc_flat, split_idxs)

    ctg_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)
    times.record_time("backup", time.time() - start_time)

    return ctg_backup, ctg_next_p_tc_l, states_exp


def q_step(states: List, goals: List[Goal], heuristic_fn, env: Environment) -> Tuple[np.array, np.ndarray]:
    # ctgs for each action
    ctg_acts = heuristic_fn(states, goals)

    # get actions
    actions: List[int] = list(np.argmin(ctg_acts, axis=1))

    # take action
    states_next: List[State]
    tcs: List[float]
    states_next, tcs = env.next_state(states, actions)

    # min cost-to-go for next state
    ctg_acts_next = heuristic_fn(states_next, goals)
    ctg_acts_next_max = ctg_acts_next.min(axis=1)

    # backup cost-to-go
    ctg_backups = np.array(tcs) + ctg_acts_next_max

    is_solved = env.is_solved(states, goals)
    ctg_backups = ctg_backups * np.logical_not(is_solved)

    return ctg_backups, ctg_acts
