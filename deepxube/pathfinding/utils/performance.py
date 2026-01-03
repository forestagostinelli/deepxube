from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Goal, Action, Domain
from deepxube.base.pathfinding import Instance


@dataclass
class PathFindPerf:
    def __init__(self) -> None:
        self.is_solved_l: List[bool] = []
        self.path_costs: List[float] = []
        self.search_itrs_l: List[int] = []
        self.ctgs: List[float] = []
        self.ctgs_bkup: List[float] = []

    def update_perf(self, instance: Instance) -> None:
        self.is_solved_l.append(instance.has_soln())
        self.ctgs.append(instance.root_node.heuristic)
        self.ctgs_bkup.append(instance.root_node.backup_val)
        if instance.has_soln():
            self.path_costs.append(instance.path_cost())
            self.search_itrs_l.append(instance.itr)

    def comb_perf(self, search_perf2: 'PathFindPerf') -> 'PathFindPerf':
        search_perf_new: PathFindPerf = PathFindPerf()
        search_perf_new.is_solved_l = self.is_solved_l + search_perf2.is_solved_l
        search_perf_new.path_costs = self.path_costs + search_perf2.path_costs
        search_perf_new.search_itrs_l = self.search_itrs_l + search_perf2.search_itrs_l
        search_perf_new.ctgs = self.ctgs + search_perf2.ctgs
        search_perf_new.ctgs_bkup = self.ctgs_bkup + search_perf2.ctgs_bkup

        return search_perf_new

    def per_solved(self) -> float:
        return 100.0 * float(np.mean(self.is_solved_l))

    def stats(self) -> Tuple[float, float, float]:
        path_cost_ave: float = 0.0
        search_itrs_ave: float = 0.0
        if len(self.path_costs) > 0:
            path_cost_ave = float(np.mean(self.path_costs))
            search_itrs_ave = float(np.mean(self.search_itrs_l))

        return self.per_solved(), path_cost_ave, search_itrs_ave

    def to_string(self) -> str:
        per_solved, path_cost_ave, search_itrs_ave = self.stats()
        return f"%solved: {per_solved:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f}"


def get_eq_weighted_perf(step_to_search_perf: Dict[int, PathFindPerf]) -> Tuple[float, float, float]:
    per_solved_l: List[float] = []
    path_cost_ave_l: List[float] = []
    search_itrs_ave_l: List[float] = []
    for search_perf in step_to_search_perf.values():
        per_solved_i, path_cost_ave_i, search_itrs_ave_i = search_perf.stats()
        per_solved_l.append(per_solved_i)
        if per_solved_i > 0.0:
            path_cost_ave_l.append(path_cost_ave_i)
            search_itrs_ave_l.append(search_itrs_ave_i)

    path_costs_ave: float = 0.0
    search_itrs_ave: float = 0.0
    if len(path_cost_ave_l) > 0:
        path_costs_ave = float(np.mean(path_cost_ave_l))
        search_itrs_ave = float(np.mean(search_itrs_ave_l))

    per_solved_ave: float = float(np.mean(per_solved_l))

    return per_solved_ave, path_costs_ave, search_itrs_ave


def print_pathfindperf(step_to_pathfindperf: Dict[int, PathFindPerf]) -> None:
    steps: List[int] = list(step_to_pathfindperf.keys())
    steps = sorted(steps)
    step_show_idxs: List[int] = list(np.unique(np.linspace(0, len(steps) - 1, 30, dtype=int)))
    for step_show_idx in step_show_idxs:
        step_show: int = steps[step_show_idx]
        pathfindperf: PathFindPerf = step_to_pathfindperf[step_show]

        is_solved: NDArray[np.bool_] = np.array(pathfindperf.is_solved_l)
        # ctgs: NDArray[np.float64] = np.array(pathfindperf.ctgs)
        ctgs_bkup: NDArray[np.float64] = np.array(pathfindperf.ctgs_bkup)

        # Get stats
        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_itrs: float = 0.0
        avg_path_costs: float = 0.0
        if per_solved > 0.0:
            avg_itrs = float(np.mean(pathfindperf.search_itrs_l))
            avg_path_costs = float(np.mean(pathfindperf.path_costs))

        # Print results
        print(f"Steps: %i, %%Solved: %.2f, avgItrs: {avg_itrs:.2f}, avgPathCosts: {avg_path_costs:.2f}, "
              f"CTG_Backup: %.2f(%.2f/%.2f/%.2f), "
              f"Num: {ctgs_bkup.shape[0]}" % (step_show, per_solved, float(np.mean(ctgs_bkup)),
                                              float(np.std(ctgs_bkup)), float(np.min(ctgs_bkup)),
                                              float(np.max(ctgs_bkup))))


def is_valid_soln(state: State, goal: Goal, soln: List[Action], env: Domain) -> bool:
    state_soln: State = state
    for action in soln:
        state_soln = env.next_state([state_soln], [action])[0][0]

    return env.is_solved([state_soln], [goal])[0]
