from typing import List, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from deepxube.utils import misc_utils
from deepxube.search.search_utils import bellman
from deepxube.nnet.nnet_utils import HeurFnQ
from deepxube.utils.timing_utils import Times
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.search.greedy_policy import Greedy
from torch.multiprocessing import get_context, Queue
from multiprocessing.process import BaseProcess
import random
import time


def bellman_step_targ(env: Environment, states: List[State], goals: List[Goal], heur_fn, eps_l: List[float],
                      times: Times) -> Tuple[List[State], NDArray, List[bool]]:
    ctg_backups, ctg_next_p_tcs, states_exp, is_solved = bellman(states, goals, heur_fn, env, times)

    # take action
    start_time = time.time()
    rand_vals = np.random.random(len(states))
    states_next: List[State] = []
    for idx in range(len(states)):
        # get next state
        state_exp: List[State] = states_exp[idx]
        ctg_next_p_tc: NDArray[np.float64] = ctg_next_p_tcs[idx]

        state_next: State = state_exp[int(np.argmin(ctg_next_p_tc))]
        if rand_vals[idx] < eps_l[idx]:
            state_next = random.choice(state_exp)
        states_next.append(state_next)
    times.record_time("get_next", time.time() - start_time)

    return states_next, ctg_backups, is_solved


def greedy_update(states: List[State], goals: List[Goal], env: Environment, num_steps: int, heuristic_fn,
                  eps_max: float, times: Times):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    greedy = Greedy(env)
    greedy.add_instances(states, goals, eps_l=eps)
    for _ in range(num_steps):
        greedy.step(heuristic_fn, times=times, rand_seen=True)

    trajs: List[List[Tuple[State, float]]] = [instance.traj for instance in greedy.instances]

    trajs_flat: List[Tuple[State, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    states_update: List[State] = []
    cost_to_go_update_l: List[float] = []
    for traj in trajs_flat:
        states_update.append(traj[0])
        cost_to_go_update_l.append(traj[1])

    goals_update: List[Goal] = []
    for goal, traj_i in zip(goals, trajs):
        goals_update.extend([goal] * len(traj_i))

    cost_to_go_update = np.array(cost_to_go_update_l)

    is_solved: NDArray[np.bool_] = np.array([instance.is_solved for instance in greedy.instances])

    return states_update, goals_update, cost_to_go_update, is_solved


def update_runner(num_states: int, step_max: int, step_probs: List[float], update_batch_size: int, heur_fn_q: HeurFnQ,
                  env: Environment, result_queue, solve_steps: int, update_method: str, eps_max: float):
    times: Times = Times()

    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    num_inst_gen: int = min(int(np.ceil(num_states / solve_steps)), update_batch_size)
    steps_gen: NDArray[np.int_] = np.random.choice(step_max + 1, size=num_inst_gen, p=step_probs)
    curr_steps: NDArray = np.zeros(len(steps_gen))
    eps_l: List[float] = list(np.random.rand(num_inst_gen) * eps_max)

    times_states: Times = Times()
    states, goals = env.get_start_goal_pairs(list(steps_gen), times=times_states)
    times.add_times(times_states, ["get_states"])

    num_states_curr: int = 0
    while num_states_curr < num_states:
        # generate states
        idxs_gen: NDArray = np.where(curr_steps == step_max)[0]
        num_gen: int = idxs_gen.shape[0]
        if num_gen > 0:
            times_states: Times = Times()
            states_gen, goals_gen = env.get_start_goal_pairs(list(steps_gen[idxs_gen]), times=times_states)
            times.add_times(times_states, ["get_states"])
            for idx, idx_gen in enumerate(list(idxs_gen)):
                states[idx_gen] = states_gen[idx]
                goals[idx_gen] = goals_gen[idx]
            curr_steps[idxs_gen] = 0

        # step
        if update_method.upper() == "GREEDY":
            states_next, ctgs, is_solved_l = bellman_step_targ(env, states, goals, heuristic_fn, eps_l, times)
        else:
            raise ValueError("Unknown update method %s" % update_method)

        # remove leftover
        leftover: int = (len(states) + num_states_curr) - num_states
        if leftover > 0:
            num_keep: int = len(states) - leftover
            keep_idxs = np.random.choice(len(states), size=num_keep)

            states = [states[keep_idx] for keep_idx in keep_idxs]
            goals = [goals[keep_idx] for keep_idx in keep_idxs]
            ctgs = ctgs[keep_idxs]
            is_solved_l = [is_solved_l[keep_idx] for keep_idx in keep_idxs]
            states_next = [states_next[keep_idx] for keep_idx in keep_idxs]

        # put to queue
        start_time = time.time()
        states_goals_nnet = env.states_goals_to_nnet_input(states, goals)
        times.record_time("states_goals_to_nnet", time.time() - start_time)

        result_queue.put((states_goals_nnet, ctgs, np.array(is_solved_l), times))

        num_states_curr += len(states)
        states = states_next
        curr_steps = curr_steps + 1
        curr_steps[np.array(is_solved_l)] = step_max


    result_queue.put((None, heur_fn_q.proc_id))


class Updater:
    def __init__(self, env: Environment, num_states: int, back_max: int, step_probs: List[float],
                 heur_fn_qs: List[HeurFnQ], solve_steps: int, update_method: str, update_batch_size: int = 1000,
                 eps_max: float = 0.0):
        super().__init__()
        ctx = get_context("spawn")
        num_procs = len(heur_fn_qs)

        # initialize queues
        self.result_queue: Queue = ctx.Queue()

        # num states per process
        self.num_states: int = num_states
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        # initialize processes
        self.procs: List[BaseProcess] = []
        for proc_id, heur_fn_q in enumerate(heur_fn_qs):
            num_states_proc: int = num_states_per_proc[proc_id]
            if num_states_proc == 0:
                continue
            # update_runner(num_states_proc, back_max, step_probs, update_batch_size,
            #              heur_fn_q, env, self.result_queue, solve_steps, update_method, eps_max)

            proc = ctx.Process(target=update_runner, args=(num_states_proc, back_max, step_probs, update_batch_size,
                                                           heur_fn_q, env, self.result_queue, solve_steps,
                                                           update_method, eps_max))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self) -> Tuple[List[NDArray[Any]], NDArray[np.float64], NDArray[np.bool_]]:
        states_goals_update_nnet: List[NDArray[Any]]
        cost_to_go_update: NDArray[np.float64]
        is_solved: NDArray[np.bool_]
        states_goals_update_nnet, cost_to_go_update, is_solved = self._update()

        output_update = np.expand_dims(cost_to_go_update, 1)

        return states_goals_update_nnet, output_update, is_solved

    def _update(self) -> Tuple[List[NDArray[Any]], NDArray[np.float64], NDArray[np.bool_]]:
        # process results
        states_goals_update_nnet_l: List[List[NDArray[Any]]] = []
        cost_to_go_update_l: List[NDArray[np.float64]] = []
        is_solved_l: List[NDArray[np.bool_]] = []

        none_count: int = 0
        result_count: int = 0
        display_counts: NDArray[np.int_] = np.linspace(1, self.num_states, 10, dtype=int)

        start_time = time.time()
        times: Times = Times()

        num_states_curr: int = 0
        not_done = set(range(len(self.procs)))
        while none_count < len(self.procs):
            # print(num_states, result_count, self.num_batches, none_count, len(self.procs), not_done)
            result = self.result_queue.get()
            if result[0] is None:
                none_count += 1
                not_done.remove(result[1])
                continue
            result_count += 1

            states_goals_nnet_q: List[NDArray[Any]]
            states_goals_nnet_q, cost_to_go_q, is_solved_q, times_q = result

            states_goals_update_nnet_l.append(states_goals_nnet_q)

            cost_to_go_update_l.append(cost_to_go_q)
            is_solved_l.append(is_solved_q)
            num_states_curr += cost_to_go_q.shape[0]
            times.add_times(times_q)

            if num_states_curr >= min(display_counts):
                print(f"Generated {num_states_curr}/{self.num_states} states "
                      f"(%.2f%%) (Total time: %.2f)" % (100 * num_states_curr / self.num_states,
                                                        time.time() - start_time))
                print(times.get_time_str())
                display_counts = display_counts[num_states_curr < display_counts]

        states_goals_update_nnet: List[NDArray[Any]] = []
        for np_idx in range(len(states_goals_update_nnet_l[0])):
            states_goals_nnet_idx: NDArray[Any] = np.concatenate([x[np_idx] for x in states_goals_update_nnet_l],
                                                                 axis=0)
            states_goals_update_nnet.append(states_goals_nnet_idx)

        cost_to_go_update: NDArray[np.float64] = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: NDArray[np.bool_] = np.concatenate(is_solved_l, axis=0)

        for proc in self.procs:
            proc.join()

        return states_goals_update_nnet, cost_to_go_update, is_solved
