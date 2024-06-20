from typing import List, Tuple, Set, Callable, Optional
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
from deepxube.nnet.nnet_utils import HeurFnQ, HeurFN_T
import numpy as np
from numpy.typing import NDArray
from deepxube.search.search_utils import bellman
from torch.multiprocessing import get_context, Queue
import random
import time


class Instance:

    def __init__(self, state: State, goal: Goal, eps: float):
        self.curr_state: State = state
        self.is_solved: bool = False
        self.goal: Goal = goal
        self.step_num: int = 0
        self.traj: List[Tuple[State, float]] = []
        self.seen_states: Set[State] = set()

        self.eps = eps

    def add_to_traj(self, state: State, cost_to_go: float):
        self.traj.append((state, cost_to_go))
        self.seen_states.add(state)

    def next_state(self, state: State):
        self.curr_state = state
        self.step_num += 1


class Greedy:
    def __init__(self, env: Environment):
        self.env: Environment = env
        self.instances: List[Instance] = []

    def add_instances(self, states: List[State], goals: List[Goal], eps_l: Optional[List[float]]):
        if eps_l is None:
            eps_l = [0] * len(states)

        assert len(states) == len(goals), "Number of states and goals should be the same"
        assert len(goals) == len(eps_l), "Number of epsilon given should be the same as number of instances"

        for state, goal, eps_inst in zip(states, goals, eps_l):
            instance: Instance = Instance(state, goal, eps_inst)
            self.instances.append(instance)

    def step(self, heuristic_fn: HeurFN_T, times: Optional[Times] = None,
             rand_seen: bool = False) -> Tuple[List[State], List[Goal], NDArray[np.float64]]:
        if times is None:
            times = Times()

        # get unsolved instances
        start_time = time.time()
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return [], [], np.zeros(0)

        states: List[State] = [instance.curr_state for instance in instances]
        goals: List[Goal] = [instance.goal for instance in instances]
        times.record_time("get_unsolved", time.time() - start_time)

        # bellman
        ctg_backups, ctg_next_p_tcs, states_exp, is_solved = bellman(states, goals, heuristic_fn, self.env, times)

        # take action
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx in range(len(instances)):
            # add state to trajectory
            instance: Instance = instances[idx]
            state: State = states[idx]
            ctg_backup: float = float(ctg_backups[idx])

            instance.add_to_traj(state, ctg_backup)

            # get next state
            if not is_solved[idx]:
                state_exp: List[State] = states_exp[idx]
                ctg_next_p_tc: NDArray[np.float64] = ctg_next_p_tcs[idx]

                state_next: State = state_exp[int(np.argmin(ctg_next_p_tc))]
                seen_state: bool = state_next in instance.seen_states
                if (rand_vals[idx] < instance.eps) or (seen_state and rand_seen):
                    state_next = random.choice(state_exp)
                instance.next_state(state_next)
        times.record_time("get_next", time.time() - start_time)

        # check which are solved
        start_time = time.time()
        states_next: List[State] = [instance.curr_state for instance in instances]
        goals = [instance.goal for instance in instances]
        is_solved_next: List[bool] = self.env.is_solved(states_next, goals)
        solved_idxs: List[int] = list(np.where(is_solved_next)[0])
        for solved_idx in solved_idxs:
            instances[solved_idx].is_solved = True
        times.record_time("record_solved", time.time() - start_time)

        return states, goals, ctg_backups

    def remove_instances(self, test_rem: Callable[[Instance], bool]) -> List[Instance]:
        """ Remove instances

        :param test_rem: A Callable that takes an instance as input and returns true if the instance should be removed
        :return: List of removed instances
        """
        instances_remove: List[Instance] = []
        instances_keep: List[Instance] = []
        for instance in self.instances:
            if test_rem(instance):
                instances_remove.append(instance)
            else:
                instances_keep.append(instance)

        self.instances = instances_keep

        return instances_remove

    def _get_unsolved_instances(self) -> List[Instance]:
        instances_unsolved: List[Instance] = [instance for instance in self.instances if not instance.is_solved]
        return instances_unsolved


def greedy_runner(env: Environment, heur_fn_q: HeurFnQ, proc_id: int,
                  max_solve_steps: int, data_q, results_queue):
    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    states, goals, inst_gen_steps = data_q.get()

    # Solve with GBFS
    greedy = Greedy(env)
    greedy.add_instances(states, goals, eps_l=None)
    for _ in range(max_solve_steps):
        greedy.step(heuristic_fn, rand_seen=True)

    is_solved_all: NDArray[np.bool_] = np.array([instance.is_solved for instance in greedy.instances])
    num_steps_all: NDArray[np.int_] = np.array([instance.step_num for instance in greedy.instances])

    # Get state cost-to-go
    state_ctg_all: NDArray[np.float64] = heuristic_fn(states, goals)

    results_queue.put((proc_id, is_solved_all, num_steps_all, state_ctg_all, inst_gen_steps))


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
    ctgs_l: List[List[float]] = [[] for _ in heur_fn_qs]
    inst_gen_steps_l: List[List[int]] = [[] for _ in heur_fn_qs]

    for _ in heur_fn_qs:
        proc_id, is_solved_i, num_steps_i, ctgs_i, inst_gen_steps_i = results_q.get()
        is_solved_l[proc_id] = is_solved_i
        num_steps_l[proc_id] = num_steps_i
        ctgs_l[proc_id] = ctgs_i
        inst_gen_steps_l[proc_id] = inst_gen_steps_i
    is_solved_all = np.hstack(is_solved_l)
    num_steps_all = np.hstack(num_steps_l)
    ctgs_all = np.hstack(ctgs_l)
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
        ctgs: NDArray[np.float64] = ctgs_all[step_idxs]

        # Get stats
        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_solve_steps: float = 0.0
        if per_solved > 0.0:
            avg_solve_steps = float(np.mean(num_steps[is_solved]))

        # Print results
        print("Back Steps: %i, %%Solved: %.2f, avgSolveSteps: %.2f, CTG Mean(Std/Min/Max): %.2f("
              "%.2f/%.2f/%.2f)" % (
                  back_step_test, per_solved, avg_solve_steps, float(np.mean(ctgs)),
                  float(np.std(ctgs)), float(np.min(ctgs)), float(np.max(ctgs))))

    return per_solved_all
