from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.multiprocessing import Queue, get_context

from deepxube.base.domain import Domain, ActsEnum, StartGoalWalkable, State, Goal, Action
from deepxube.base.heuristic import HeurNNetPar, HeurNNetParV, HeurNNetParQ
from deepxube.nnet.nnet_utils import NNetCallable
from deepxube.nnet import nnet_utils
from deepxube.utils.misc_utils import flatten
from deepxube.utils.timing_utils import Times
import numpy as np

import time


def data_runner(queue1: Queue, queue2: Queue) -> None:
    while True:
        the = queue1.get()
        if the is None:
            break
        queue2.put(the)


def test_env(env: Domain, num_states: int, step_max: int) -> Tuple[List[State], List[Goal], List[Action]]:
    # get data
    start_time = time.time()
    sg_times: Times = Times()
    states, goals = env.sample_start_goal_pairs(list(np.random.randint(step_max + 1, size=num_states)), times=sg_times)
    assert len(states) == len(goals), f"state({len(states)}) and goal({len(goals)}) pairs not same length"

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print(sg_times.get_time_str(decplace=16))
    print("Generated %i start/goal states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # get state action
    start_time = time.time()
    actions: List[Action] = env.sample_state_action(states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Got %i random actions in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # next state
    start_time = time.time()
    env.next_state(states, actions)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Got %i next states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # is_solved
    start_time = time.time()
    is_solved_l: List[bool] = env.is_solved(states, goals)
    per_solved: float = 100.0 * float(np.mean(is_solved_l))
    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print(f"Computed is_solved for {len(states)} states ({per_solved:.2f}% solved) in {elapsed_time} seconds "
          f"({states_per_sec:.2f}/second)")

    # next state
    start_time = time.time()
    states_next: List[State] = env.sample_next_state(states)[0]

    elapsed_time = time.time() - start_time
    states_per_sec = len(states_next) / elapsed_time
    print("Got %i random next states in %s seconds (%.2f/second)" % (len(states_next), elapsed_time, states_per_sec))

    # multiprocessing
    print("")
    start_time = time.time()
    ctx = get_context("spawn")
    queue1: Queue = ctx.Queue()
    queue2: Queue = ctx.Queue()
    proc = ctx.Process(target=data_runner, args=(queue1, queue2))
    proc.daemon = True
    proc.start()
    print("Process start time: %.2f" % (time.time() - start_time))

    start_time = time.time()
    queue1.put(env)
    print("Environment send time: %s" % (time.time() - start_time))

    start_time = time.time()
    queue2.get()
    print("Environment get time: %s" % (time.time() - start_time))

    start_time = time.time()
    queue1.put(None)
    proc.join()
    print("Process join time: %.2f" % (time.time() - start_time))

    return states, goals, actions


def test_envstartgoalrw(env: StartGoalWalkable, num_states: int) -> None:
    # generate start/goal states
    start_time = time.time()
    states: List[State] = env.sample_start_states(num_states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Generated %i start states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))


def test_envenumerableacts(env: ActsEnum, states: List[State]) -> None:
    torch.set_num_threads(1)

    # expand
    start_time = time.time()
    states_exp, _, tcs = env.expand(states)
    ave_next_states: float = float(np.mean([len(x) for x in states_exp]))
    ave_tc: float = float(np.mean(flatten(tcs)[0]))

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print(f"Expanded %i states, mean #next/tc: ({ave_next_states:.2f}/{ave_tc:.2f}), "
          f"in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))


def init_nnet(heur_nnet: HeurNNetPar) -> Tuple[nn.Module, torch.device]:
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    nnet: nn.Module = heur_nnet.get_nnet()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    return nnet, device


def heur_fn_out(heur_nnet: HeurNNetPar, heur_fn: NNetCallable, states: List[State], goals: List[Goal],
                actions: List[Action]) -> None:
    if isinstance(heur_nnet, HeurNNetParV):
        heur_fn(states, goals)
    elif isinstance(heur_nnet, HeurNNetParQ):
        heur_fn(states, goals, [[action] for action in actions])
    else:
        raise ValueError(f"Unknown heur fn class {heur_fn}")


def test_heur_nnet_par(heur_nnet_par: HeurNNetPar, states: List[State], goals: List[Goal], actions: List[Action]) -> None:
    # nnet format
    start_time = time.time()
    if isinstance(heur_nnet_par, HeurNNetParV):
        heur_nnet_par.to_np(states, goals)
    elif isinstance(heur_nnet_par, HeurNNetParQ):
        heur_nnet_par.to_np(states, goals, [[action] for action in actions])
    else:
        raise ValueError(f"Unknown heur nnet class {heur_nnet_par}")
    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Converted %i states and goals to nnet format in "
          "%s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # initialize nnet
    nnet, device = init_nnet(heur_nnet_par)
    print("")
    heur_fn: NNetCallable = heur_nnet_par.get_nnet_fn(nnet, None, device, None)
    heur_fn_out(heur_nnet_par, heur_fn, states, goals, actions)

    # nnet heuristic
    start_time = time.time()
    heur_fn_out(heur_nnet_par, heur_fn, states, goals, actions)

    nnet_time = time.time() - start_time
    states_per_sec = len(states) / nnet_time
    print("Computed heuristic for %i states in %s seconds (%.2f/second)" % (len(states), nnet_time, states_per_sec))


def time_test(domain: Domain, heur_nnet_par: Optional[HeurNNetPar], num_states: int, step_max: int) -> None:
    states, goals, actions = test_env(domain, num_states, step_max)
    if isinstance(domain, StartGoalWalkable):
        test_envstartgoalrw(domain, num_states)
    if isinstance(domain, ActsEnum):
        test_envenumerableacts(domain, states)

    if heur_nnet_par is not None:
        test_heur_nnet_par(heur_nnet_par, states, goals, actions)
