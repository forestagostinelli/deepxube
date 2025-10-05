from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from deepxube.base.environment import Environment, State, Action, Goal
from deepxube.base.pathfinding import Instance
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
import time


def is_valid_soln(state: State, goal: Goal, soln: List[Action], env: Environment) -> bool:
    state_soln: State = state
    for action in soln:
        state_soln = env.next_state([state_soln], [action])[0][0]

    return env.is_solved([state_soln], [goal])[0]


@dataclass
class PathFindPerf:
    def __init__(self):
        self.is_solved_l: List[bool] = []
        self.path_costs: List[float] = []
        self.search_itrs_l: List[int] = []
        self.ctgs: List[float] = []
        self.ctgs_bkup: List[float] = []

    def update_perf(self, instance: Instance):
        self.is_solved_l.append(instance.has_soln())
        self.ctgs.append(instance.root_node.heuristic)
        self.ctgs_bkup.append(instance.root_node.backup())
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
            path_cost_ave: float = float(np.mean(self.path_costs))
            search_itrs_ave: float = float(np.mean(self.search_itrs_l))

        return self.per_solved(), path_cost_ave, search_itrs_ave

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


def print_pathfindperf(step_to_pathfindperf: Dict[int, PathFindPerf]):
    steps: List[int] = list(step_to_pathfindperf.keys())
    steps = sorted(steps)
    step_show_idxs: List[int] = list(np.unique(np.linspace(0, len(steps) - 1, 30, dtype=int)))
    for step_show_idx in step_show_idxs:
        step_show: int = steps[step_show_idx]
        pathfindperf: PathFindPerf = step_to_pathfindperf[step_show]

        is_solved: NDArray[np.bool_] = np.array(pathfindperf.is_solved_l)
        ctgs: NDArray[np.float64] = np.array(pathfindperf.ctgs)
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
              f"CTG Mean(Std/Min/Max): %.2f(%.2f/%.2f/%.2f), CTG_Backup: %.2f("
              "%.2f/%.2f/%.2f)" % (
                  step_show, per_solved,
                  float(np.mean(ctgs)), float(np.std(ctgs)), float(np.min(ctgs)), float(np.max(ctgs)),
                  float(np.mean(ctgs_bkup)), float(np.std(ctgs_bkup)), float(np.min(ctgs_bkup)),
                  float(np.max(ctgs_bkup))))


"""
def search_runner(env: Environment, heur_nnet: HeurNNet, proc_id: int, search_method: str, max_solve_steps: int, data_q,
                  results_queue):
    heuristic_fn = heur_nnet.get_nnet_par_fn()
    states, goals, inst_gen_steps = data_q.get()

    # Solve with pathfinding
    search_method = search_method.upper()
    if search_method == "GREEDY":
        search: PathFindV = Greedy(env)
    elif search_method == "ASTAR":
        search: PathFindV = BWAS(env)
    else:
        raise ValueError(f"Unknown search method {search_method}")

    search.add_instances(states, goals, heuristic_fn, compute_init_heur=True)
    for _ in range(max_solve_steps):
        search.step(heuristic_fn)

    is_solved_all: NDArray[np.bool_] = np.array([instance.has_soln() for instance in search.instances])
    num_steps_all: NDArray[np.int_] = np.array([instance.itr for instance in search.instances])
    path_costs_all: NDArray[np.int_] = np.array([instance.path_cost() for instance in search.instances])

    # Get state cost-to-go
    state_ctg_all: NDArray[np.float64] = np.array([inst.root_node.heuristic for inst in search.instances])
    states_ctg_bkup_all: NDArray[np.float64] = np.array([inst.root_node.bellman_backup() for inst in search.instances])

    results_queue.put((proc_id, is_solved_all, num_steps_all, path_costs_all, state_ctg_all, states_ctg_bkup_all,
                       inst_gen_steps))


def search_test(env: Environment, states: List[State], goals: List[Goal], inst_gen_steps: List[int],
                num_procs: int, nnet_file: str, nnet_batch_size: Optional[int], device: torch.device, on_gpu: bool,
                search_method: str, max_solve_steps: int) -> Tuple[float, NDArray]:
    # start heur fns
    all_zeros: bool = False
    if len(nnet_file) == 0:
        all_zeros = True
    heur_fn_qs, heur_procs = start_nnet_fn_runners(env.get_v_nnet().__class__, num_procs, nnet_file, device, on_gpu,
                                                   all_zeros=all_zeros, clip_zero=False, batch_size=nnet_batch_size)

    # start pathfinding runners
    ctx = get_context("spawn")
    results_q: Queue = ctx.Queue()
    data_qs: List[Queue] = [ctx.Queue() for _ in heur_fn_qs]
    procs: List = []
    num_states_per_proc: List[int] = misc_utils.split_evenly(len(states), len(heur_fn_qs))
    start_idx: int = 0
    for proc_id, heur_fn_q in enumerate(heur_fn_qs):
        num_states_proc: int = num_states_per_proc[proc_id]
        if num_states_proc == 0:
            continue

        # start process
        end_idx: int = start_idx + num_states_proc
        data_q: Queue = data_qs[proc_id]
        proc = ctx.Process(target=search_runner, args=(env, heur_fn_q, proc_id, search_method, max_solve_steps, data_q,
                                                       results_q))
        proc.daemon = True
        proc.start()
        procs.append(proc)

        # put data
        states_proc = states[start_idx:end_idx]
        goals_proc = goals[start_idx:end_idx]
        inst_gen_steps_proc = inst_gen_steps[start_idx:end_idx]
        data_q.put((states_proc, goals_proc, inst_gen_steps_proc))
        start_idx = end_idx

    is_solved_l: List[List[bool]] = [[] for _ in heur_fn_qs]
    num_steps_l: List[List[int]] = [[] for _ in heur_fn_qs]
    path_costs_l: List[List[float]] = [[] for _ in heur_fn_qs]
    ctgs_l: List[List[float]] = [[] for _ in heur_fn_qs]
    ctgs_bkup_l: List[List[float]] = [[] for _ in heur_fn_qs]
    inst_gen_steps_l: List[List[int]] = [[] for _ in heur_fn_qs]

    for _ in heur_fn_qs:
        proc_id, is_solved_i, num_steps_i, path_costs_i, ctgs_i, ctgs_bkup_i, inst_gen_steps_i = results_q.get()
        is_solved_l[proc_id] = is_solved_i
        num_steps_l[proc_id] = num_steps_i
        path_costs_l[proc_id] = path_costs_i
        ctgs_l[proc_id] = ctgs_i
        ctgs_bkup_l[proc_id] = ctgs_bkup_i
        inst_gen_steps_l[proc_id] = inst_gen_steps_i
    is_solved_all = np.hstack(is_solved_l)
    num_steps_all = np.hstack(num_steps_l)
    path_costs_all = np.hstack(path_costs_l)
    ctgs_all = np.hstack(ctgs_l)
    ctgs_bkup_all = np.hstack(ctgs_bkup_l)
    inst_gen_steps_new = list(np.hstack(inst_gen_steps_l))
    assert inst_gen_steps_new == inst_gen_steps

    for proc in procs:
        proc.join()

    per_solved_all = 100 * float(sum(is_solved_all)) / float(len(is_solved_all))
    steps_show: List[int] = list(np.unique(np.linspace(0, max(inst_gen_steps), 30, dtype=int)))
    for back_step_test in np.sort(np.unique(steps_show)):
        # Get states
        step_idxs = np.where(np.array(inst_gen_steps) == back_step_test)[0]
        if len(step_idxs) == 0:
            continue

        is_solved: NDArray[np.bool_] = is_solved_all[step_idxs]
        num_steps: NDArray[np.int_] = num_steps_all[step_idxs]
        path_costs: NDArray[np.int_] = path_costs_all[step_idxs]
        ctgs: NDArray[np.float64] = ctgs_all[step_idxs]
        ctgs_bkup: NDArray[np.float64] = ctgs_bkup_all[step_idxs]

        # Get stats
        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_steps: float = 0.0
        avg_path_costs: float = 0.0
        if per_solved > 0.0:
            avg_steps = float(np.mean(num_steps[is_solved]))
            avg_path_costs = float(np.mean(path_costs[is_solved]))

        # Print results
        print(f"Steps: %i, %%Solved: %.2f, avgItrs: {avg_steps:.2f}, avgPathCosts: {avg_path_costs:.2f}, "
              f"CTG Mean(Std/Min/Max): %.2f(%.2f/%.2f/%.2f), CTG_Backup: %.2f("
              "%.2f/%.2f/%.2f)" % (
                  back_step_test, per_solved,
                  float(np.mean(ctgs)), float(np.std(ctgs)), float(np.min(ctgs)), float(np.max(ctgs)),
                  float(np.mean(ctgs_bkup)), float(np.std(ctgs_bkup)), float(np.min(ctgs_bkup)),
                  float(np.max(ctgs_bkup))))

    stop_nnet_runners(heur_procs, heur_fn_qs)

    return per_solved_all, is_solved_all
"""
