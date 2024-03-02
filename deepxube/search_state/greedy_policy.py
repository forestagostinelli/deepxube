from typing import List, Tuple, Set, Callable, Optional
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
from deepxube.utils.nnet_utils import HeurFnQ
import numpy as np
from deepxube.utils import search_utils
from torch.multiprocessing import get_context
import random
import time


class Instance:

    def __init__(self, state: State, goal: Goal, eps: float):
        self.curr_state: State = state
        self.is_solved: bool = False
        self.goal: Goal = goal
        self.num_steps: int = 0
        self.trajs: List[Tuple[State, float]] = []
        self.seen_states: Set[State] = set()

        self.eps = eps

    def add_to_traj(self, state: State, cost_to_go: float):
        self.trajs.append((state, cost_to_go))
        self.seen_states.add(state)

    def next_state(self, state: State):
        self.curr_state = state
        self.num_steps += 1


class Greedy:
    def __init__(self, states: List[State], goals: List[Goal], env: Environment,
                 eps_l: Optional[List[float]] = None):
        self.curr_states: List[State] = states
        self.env: Environment = env

        if eps_l is None:
            eps_l = [0] * len(self.curr_states)

        self.instances: List[Instance] = []
        for state, goal, eps_inst in zip(states, goals, eps_l):
            instance: Instance = Instance(state, goal, eps_inst)
            self.instances.append(instance)

    def step(self, heuristic_fn: Callable, times: Optional[Times] = None, rand_seen: bool = False) -> None:
        if times is None:
            times = Times()

        # check which are solved
        start_time = time.time()
        self._record_solved()
        times.record_time("record_solved", time.time() - start_time)

        # take a step for unsolved states
        self._move(heuristic_fn, times, rand_seen)

    def get_trajs(self) -> List[List[Tuple[State, float]]]:
        trajs_all: List[List[Tuple[State, float]]] = []
        for instance in self.instances:
            trajs_all.append(instance.trajs)

        return trajs_all

    def get_is_solved(self) -> List[bool]:
        is_solved: List[bool] = [x.is_solved for x in self.instances]

        return is_solved

    def get_num_steps(self) -> List[int]:
        num_steps: List[int] = [x.num_steps for x in self.instances]

        return num_steps

    def _record_solved(self) -> None:
        # get unsolved instances
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return

        states: List[State] = [instance.curr_state for instance in instances]
        goals: List[Goal] = [instance.goal for instance in instances]

        is_solved: List[bool] = self.env.is_solved(states, goals)
        solved_idxs: List[int] = list(np.where(is_solved)[0])
        if len(solved_idxs) > 0:
            instances_solved: List[Instance] = [instances[idx] for idx in solved_idxs]
            states_solved: List[State] = [instance.curr_state for instance in instances_solved]

            for instance, state in zip(instances_solved, states_solved):
                instance.add_to_traj(state, 0.0)
                instance.is_solved = True

    def _move(self, heuristic_fn: Callable, times: Times, rand_seen: bool) -> None:
        # get unsolved instances
        start_time = time.time()
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return
        states: List[State] = [instance.curr_state for instance in instances]
        goals: List[Goal] = [instance.goal for instance in instances]
        times.record_time("get_unsolved", time.time() - start_time)

        ctg_backups, ctg_next_p_tcs, states_exp = search_utils.bellman(states, goals, heuristic_fn, self.env, times)

        # make move
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx in range(len(instances)):
            # add state to trajectory
            instance: Instance = instances[idx]
            state: State = states[idx]
            ctg_backup: float = float(ctg_backups[idx])

            instance.add_to_traj(state, ctg_backup)

            # get next state
            state_exp: List[State] = states_exp[idx]
            ctg_next_p_tc: np.ndarray = ctg_next_p_tcs[idx]

            state_next: State = state_exp[int(np.argmin(ctg_next_p_tc))]
            seen_state: bool = state_next in instance.seen_states
            if (rand_vals[idx] < instance.eps) or (seen_state and rand_seen):
                state_next: State = random.choice(state_exp)

            instance.next_state(state_next)
        times.record_time("get_next", time.time() - start_time)

    def _get_unsolved_instances(self) -> List[Instance]:
        instances_unsolved: List[Instance] = [instance for instance in self.instances if not instance.is_solved]
        return instances_unsolved


def greedy_runner(env: Environment, states: List[State], goals: List[Goal], heur_fn_q: HeurFnQ, proc_id: int,
                  max_solve_steps: int, results_queue):
    heuristic_fn = heur_fn_q.get_heuristic_fn(env)

    # Solve with GBFS
    greedy = Greedy(states, goals, env, eps_l=None)
    for _ in range(max_solve_steps):
        greedy.step(heuristic_fn, rand_seen=False)

    is_solved_all: np.ndarray = np.array(greedy.get_is_solved())
    num_steps_all: np.ndarray = np.array(greedy.get_num_steps())

    # Get state cost-to-go
    state_ctg_all: np.ndarray = heuristic_fn(states, goals)

    results_queue.put((proc_id, is_solved_all, num_steps_all, state_ctg_all))


def greedy_test(states: List[State], goals: List[Goal], state_steps_l: List[int], env: Environment,
                heur_fn_qs: List[HeurFnQ], max_solve_steps: Optional[int] = None) -> float:
    # initialize
    state_back_steps: np.ndarray = np.array(state_steps_l)
    if max_solve_steps is None:
        max_solve_steps = max(np.max(state_back_steps), 1)

    ctx = get_context("spawn")
    results_q: ctx.Queue = ctx.Queue()
    procs: List[ctx.Process] = []
    num_states_per_proc: List[int] = misc_utils.split_evenly(len(states), len(heur_fn_qs))
    start_idx: int = 0
    for proc_id, heur_fn_q in enumerate(heur_fn_qs):
        num_states_proc: int = num_states_per_proc[proc_id]
        if num_states_proc == 0:
            continue
        # update_runner(num_states_proc, back_max, step_probs, update_batch_size,
        #              heur_fn_q, env.env_name, self.result_queue, solve_steps, update_method, eps_max)

        end_idx: int = start_idx + num_states_proc
        states_proc = states[start_idx:end_idx]
        goals_proc = goals[start_idx:end_idx]
        proc = ctx.Process(target=greedy_runner, args=(env, states_proc, goals_proc, heur_fn_q, proc_id,
                                                       max_solve_steps, results_q))
        proc.daemon = True
        proc.start()
        procs.append(proc)
        start_idx = end_idx

    is_solved_l: List[List[bool]] = [[] for _ in heur_fn_qs]
    num_steps_l: List[List[int]] = [[] for _ in heur_fn_qs]
    ctgs_l: List[List[float]] = [[] for _ in heur_fn_qs]

    for _ in heur_fn_qs:
        proc_id, is_solved_i, num_steps_i, ctgs_i = results_q.get()
        is_solved_l[proc_id] = is_solved_i
        num_steps_l[proc_id] = num_steps_i
        ctgs_l[proc_id] = ctgs_i
    is_solved_all = np.hstack(is_solved_l)
    num_steps_all = np.hstack(num_steps_l)
    ctgs_all = np.hstack(ctgs_l)

    for proc in procs:
        proc.join()

    per_solved_all = 100 * float(sum(is_solved_all)) / float(len(is_solved_all))
    steps_show: List[int] = list(np.unique(np.linspace(0, max(state_back_steps), 30, dtype=int)))
    for back_step_test in np.sort(np.unique(steps_show)):
        # Get states
        step_idxs = np.where(state_back_steps == back_step_test)[0]
        if len(step_idxs) == 0:
            continue

        is_solved: np.ndarray = is_solved_all[step_idxs]
        num_steps: np.ndarray = num_steps_all[step_idxs]
        ctgs: np.ndarray = ctgs_all[step_idxs]

        # Get stats
        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_solve_steps = 0.0
        if per_solved > 0.0:
            avg_solve_steps = np.mean(num_steps[is_solved])

        # Print results
        print("Back Steps: %i, %%Solved: %.2f, avgSolveSteps: %.2f, CTG Mean(Std/Min/Max): %.2f("
              "%.2f/%.2f/%.2f)" % (
                  back_step_test, per_solved, avg_solve_steps, float(np.mean(ctgs)),
                  float(np.std(ctgs)), np.min(ctgs),
                  np.max(ctgs)))

    return per_solved_all