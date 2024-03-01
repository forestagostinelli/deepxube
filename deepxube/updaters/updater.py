from typing import List, Tuple
import numpy as np
from deepxube.utils import misc_utils
from deepxube.utils.nnet_utils import HeurFnQ
from deepxube.utils.timing_utils import Times
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.search_state.greedy_policy import Greedy
from torch.multiprocessing import get_context
import time


def greedy_update(states: List[State], goals: List[Goal], env: Environment, num_steps: int, heuristic_fn,
                  eps_max: float, times: Times):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    greedy = Greedy(states, goals, env, eps_l=eps)
    for _ in range(num_steps):
        greedy.step(heuristic_fn, times=times, rand_seen=True)

    trajs: List[List[Tuple[State, float]]] = greedy.get_trajs()

    trajs_flat: List[Tuple[State, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    states_update: List = []
    cost_to_go_update_l: List[float] = []
    for traj in trajs_flat:
        states_update.append(traj[0])
        cost_to_go_update_l.append(traj[1])

    goals_update: List[Goal] = []
    for goal, traj_i in zip(goals, trajs):
        goals_update.extend([goal] * len(traj_i))

    cost_to_go_update = np.array(cost_to_go_update_l)

    is_solved: np.ndarray = np.array(greedy.get_is_solved())

    return states_update, goals_update, cost_to_go_update, is_solved


def update_runner(num_states: int, step_max: int, step_probs: List[float], update_batch_size: int, heur_fn_q: HeurFnQ,
                  env: Environment, result_queue, solve_steps: int, update_method: str, eps_max: float):
    heuristic_fn = heur_fn_q.get_heuristic_fn(env)
    states_per_inst: float = solve_steps
    num_states_curr: int = 0

    states_per_inst_l: List[float] = []
    while num_states_curr < num_states:
        times: Times = Times()
        num_states_i_pred: int = int(np.ceil(update_batch_size * states_per_inst))
        if (num_states_curr + num_states_i_pred) > num_states:
            num_inst_i = int(np.ceil((num_states - num_states_curr) / states_per_inst))
        else:
            num_inst_i = update_batch_size
        num_steps_l: np.array = np.random.choice(step_max + 1, size=num_inst_i, p=step_probs)

        times_states: Times = Times()
        states, states_goal_set = env.get_start_goal_pairs(list(num_steps_l), times=times_states)
        times.add_times(times_states, ["get_states"])

        if update_method.upper() == "GREEDY":
            times_greedy: Times = Times()
            states_update, goals_update, cost_to_go_update, is_solved = greedy_update(states, states_goal_set, env,
                                                                                      solve_steps, heuristic_fn,
                                                                                      eps_max, times_greedy)
            times.add_times(times_greedy, ["greedy"])
        else:
            raise ValueError("Unknown update method %s" % update_method)

        states_per_inst_l.append(len(states_update) / len(states))
        states_per_inst = np.mean(states_per_inst_l)

        leftover: int = (len(states_update) + num_states_curr) - num_states
        if leftover > 0:
            num_keep: int = len(states_update) - leftover
            keep_idxs = np.random.choice(len(states_update), size=num_keep)

            states_update = [states_update[keep_idx] for keep_idx in keep_idxs]
            goals_update = [goals_update[keep_idx] for keep_idx in keep_idxs]
            cost_to_go_update = cost_to_go_update[keep_idxs]

        start_time = time.time()
        states_update_nnet = env.states_to_nnet_input(states_update)
        times.record_time("states_to_nnet", time.time() - start_time)

        start_time = time.time()
        goals_update_nnet = env.goals_to_nnet_input(goals_update)
        times.record_time("goals_to_nnet", time.time() - start_time)

        result_queue.put((states_update_nnet, goals_update_nnet, cost_to_go_update, is_solved, times))

        num_states_curr += len(states_update)

    result_queue.put((None, heur_fn_q.proc_id))


class Updater:
    def __init__(self, env: Environment, num_states: int, back_max: int, step_probs: List[float],
                 heur_fn_qs: List[HeurFnQ], solve_steps: int, update_method: str, update_batch_size: int = 1000,
                 eps_max: float = 0.0):
        super().__init__()
        ctx = get_context("spawn")
        num_procs = len(heur_fn_qs)

        # initialize queues
        self.result_queue: ctx.Queue = ctx.Queue()

        # num states per process
        self.num_states: int = num_states
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        # initialize processes
        self.procs: List[ctx.Process] = []
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

    def update(self):
        states_update_nnet: List[np.ndarray]
        states_update_goal_nnet: List[np.ndarray]
        cost_to_go_update: np.ndarray
        is_solved: np.ndarray
        states_update_nnet, states_update_goal_nnet, cost_to_go_update, is_solved = self._update()

        output_update = np.expand_dims(cost_to_go_update, 1)

        return states_update_nnet, states_update_goal_nnet, output_update, is_solved

    def _update(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        # process results
        states_update_nnet_l: List[List[np.ndarray]] = []
        states_update_goal_nnet_l: List[List[np.ndarray]] = []
        cost_to_go_update_l: List = []
        is_solved_l: List = []

        none_count: int = 0
        result_count: int = 0
        display_counts: np.array = np.linspace(1, self.num_states, 10, dtype=int)

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

            states_nnet_q: List[np.ndarray]
            states_goal_nnet_q: List[np.ndarray]
            states_nnet_q, states_goal_nnet_q, cost_to_go_q, is_solved_q, times_q = result

            states_update_nnet_l.append(states_nnet_q)
            states_update_goal_nnet_l.append(states_goal_nnet_q)

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

        states_update_nnet: List[np.ndarray] = []
        for np_idx in range(len(states_update_nnet_l[0])):
            states_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_nnet_l], axis=0)
            states_update_nnet.append(states_nnet_idx)

        states_update_goal_nnet: List[np.ndarray] = []
        for np_idx in range(len(states_update_goal_nnet_l[0])):
            states_goal_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_goal_nnet_l], axis=0)
            states_update_goal_nnet.append(states_goal_nnet_idx)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l, axis=0)

        for proc in self.procs:
            proc.join()

        return states_update_nnet, states_update_goal_nnet, cost_to_go_update, is_solved
