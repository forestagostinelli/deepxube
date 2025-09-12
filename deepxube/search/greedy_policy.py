from typing import List, Optional, Any
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.utils import misc_utils
from deepxube.nnet.nnet_utils import HeurFnQ, HeurFN_T
from deepxube.search.search_abstract import Node, Instance, Search
import numpy as np
from numpy.typing import NDArray
from deepxube.search.search_utils import bellman
from torch.multiprocessing import get_context, Queue
import random
import time


class InstanceGr(Instance):
    def __init__(self, state: State, goal: Goal, heuristic: float, is_solved: bool, inst_info: Any, eps: float):
        super().__init__(state, goal, heuristic, is_solved, inst_info)
        self.curr_node: Node = self.root_node
        self.eps = eps


class Greedy(Search[InstanceGr]):
    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: HeurFN_T,
                      inst_infos: Optional[List[Any]] = None, eps_l: Optional[List[float]] = None):
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None] * len(states)
        if eps_l is None:
            eps_l = [0.0] * len(states)

        assert len(states) == len(goals), "Number of states and goals should be the same"
        assert len(states) == len(eps_l), "Number of epsilon given should be the same as number of instances"
        assert len(states) == len(inst_infos), "Number of instance info given should be the same as number of instances"

        heuristics: NDArray = heur_fn(states, goals)
        is_solved_l: List[bool] = self.env.is_solved(states, goals)

        for state, goal, heuristic, is_solved, inst_info, eps_inst in zip(states, goals, heuristics, is_solved_l,
                                                                          inst_infos, eps_l):
            instance: InstanceGr = InstanceGr(state, goal, heuristic, is_solved, inst_info, eps_inst)
            self.instances.append(instance)
        self.times.record_time("add", time.time() - start_time)

    def step(self, heur_fn: HeurFN_T):
        # get unsolved instances
        instances: List[InstanceGr] = self._get_unsolved_instances()
        if len(instances) == 0:
            return None

        self.expand_nodes(instances, [[inst.curr_node] for inst in instances], heur_fn)

        # take action
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx, instance in enumerate(instances):
            # check solved
            if instance.curr_node.is_solved:
                instance.goal_node = instance.curr_node
            else:
                # get next state
                curr_node: Node = instance.curr_node
                assert curr_node.children is not None
                assert curr_node.t_costs is not None
                t_costs: List[float] = curr_node.t_costs
                children: List[Node] = curr_node.children
                tc_p_ctg_next: List[float] = [t_cost + child.heuristic for t_cost, child in zip(t_costs, children)]

                child_idx: int = int(np.argmin(tc_p_ctg_next))
                if rand_vals[idx] < instance.eps:
                    child_idx: int = random.choice(list(range(len(tc_p_ctg_next))))
                node_next: Node = children[child_idx]

                instance.curr_node = node_next
            instance.itr += 1
        self.times.record_time("get_next", time.time() - start_time)

        return None


def greedy_runner(env: Environment, heur_fn_q: HeurFnQ, proc_id: int,
                  max_solve_steps: int, data_q, results_queue):
    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    states, goals, inst_gen_steps = data_q.get()

    # Solve with GBFS
    greedy = Greedy(env)
    greedy.add_instances(states, goals, heuristic_fn, eps_l=None)
    for _ in range(max_solve_steps):
        greedy.step(heuristic_fn)

    is_solved_all: NDArray[np.bool_] = np.array([instance.has_soln() for instance in greedy.instances])
    num_steps_all: NDArray[np.int_] = np.array([instance.itr for instance in greedy.instances])
    path_costs_all: NDArray[np.int_] = np.array([instance.path_cost() for instance in greedy.instances])

    # Get state cost-to-go
    state_ctg_all: NDArray[np.float64] = heuristic_fn(states, goals)

    states_ctg_bkup_all: NDArray[np.float64] = bellman(states, goals, heuristic_fn, env)[0]

    results_queue.put((proc_id, is_solved_all, num_steps_all, path_costs_all, state_ctg_all, states_ctg_bkup_all,
                       inst_gen_steps))


def greedy_test(states: List[State], goals: List[Goal], inst_gen_steps: List[int], env: Environment,
                heur_fn_qs: List[HeurFnQ], max_solve_steps: Optional[int] = None) -> float:
    # initialize
    if max_solve_steps is None:
        max_solve_steps = max(max(inst_gen_steps), 1)

    ctx = get_context("spawn")
    data_q: Queue = ctx.Queue()
    results_q: Queue = ctx.Queue()
    procs: List = []
    num_states_per_proc: List[int] = misc_utils.split_evenly(len(states), len(heur_fn_qs))
    start_idx: int = 0
    for proc_id, heur_fn_q in enumerate(heur_fn_qs):
        num_states_proc: int = num_states_per_proc[proc_id]
        if num_states_proc == 0:
            continue
        # update_runner(num_states_proc, back_max, step_probs, update_batch_size,
        #              heur_fn_q, env.env_name, self.result_queue, solve_steps, update_method, eps_max)

        # start process
        end_idx: int = start_idx + num_states_proc
        proc = ctx.Process(target=greedy_runner, args=(env, heur_fn_q, proc_id, max_solve_steps, data_q, results_q))
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
    inst_gen_steps = list(np.hstack(inst_gen_steps_l))

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

    return per_solved_all
