from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from deepxube.environments.environment_abstract import Environment, State, Action, Goal
from deepxube.search.search_abstract import Instance
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
import time


def is_valid_soln(state: State, goal: Goal, soln: List[Action], env: Environment) -> bool:
    state_soln: State = state
    for action in soln:
        state_soln = env.next_state([state_soln], [action])[0][0]

    return env.is_solved([state_soln], [goal])[0]


@dataclass
class SearchPerf:
    def __init__(self):
        self.is_solved_l: List[bool] = []
        self.path_costs: List[float] = []
        self.search_itrs_l: List[int] = []

    def update_perf(self, instance: Instance):
        self.is_solved_l.append(instance.has_soln())
        if instance.has_soln():
            self.path_costs.append(instance.path_cost())
            self.search_itrs_l.append(instance.itr)

    def comb_perf(self, search_perf2: 'SearchPerf') -> 'SearchPerf':
        search_perf_new: SearchPerf = SearchPerf()
        search_perf_new.is_solved_l = self.is_solved_l + search_perf2.is_solved_l
        search_perf_new.path_costs = self.path_costs + search_perf2.path_costs
        search_perf_new.search_itrs_l = self.search_itrs_l + search_perf2.search_itrs_l

        return search_perf_new

    def stats(self) -> Tuple[float, float, float]:
        path_cost_ave: float = 0.0
        if len(self.path_costs) > 0:
            path_cost_ave: float = float(np.mean(self.path_costs))
        search_itrs_ave: float = float(np.mean(self.search_itrs_l))
        per_solved: float = 100.0 * float(np.mean(self.is_solved_l))

        return per_solved, path_cost_ave, search_itrs_ave

    def to_string(self) -> str:
        per_solved, path_cost_ave, search_itrs_ave = self.stats()
        return f"%solved: {per_solved:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f}"


def bellman(states: List[State], goals: List[Goal], heuristic_fn,
            env: Environment,
            times: Optional[Times] = None) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]],
                                                    List[List[State]], List[List[Action]], List[List[float]],
                                                    List[bool]]:
    if times is None:
        times = Times()

    # expand states
    start_time = time.time()
    states_exp, actions_exp, tcs_exp = env.expand(states)
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
    tcs_flat: NDArray[np.float64] = np.hstack(tcs_exp)
    ctg_next_p_tc_flat: NDArray[np.float64] = tcs_flat + ctg_next_flat
    ctg_next_p_tc_l: List[NDArray[np.float64]] = np.split(ctg_next_p_tc_flat, split_idxs)

    ctg_backup: NDArray[np.float64] = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)
    times.record_time("backup", time.time() - start_time)

    return ctg_backup, ctg_next_p_tc_l, states_exp, actions_exp, tcs_exp, is_solved
