from typing import List, Tuple

import torch
import torch.nn as nn
from torch.multiprocessing import Queue, get_context

from deepxube.base.env import Env, EnvEnumerableActs, EnvStartGoalRW, State, Goal, Action
from deepxube.base.heuristic import NNetPar, HeurNNetV, NNetCallable, HeurNNetQ
from deepxube.nnet import nnet_utils
from deepxube.utils.misc_utils import flatten
from deepxube.pathfinding.q.bwqs import BWQSEnum, InstArgsBWQS
import numpy as np

import time


def data_runner(queue1: Queue, queue2: Queue):
    while True:
        the = queue1.get()
        if the is None:
            break
        queue2.put(the)


def test_env(env: Env, num_states: int, step_max: int) -> Tuple[List[State], List[Goal], List[Action]]:
    # get data
    start_time = time.time()
    states, goals = env.get_start_goal_pairs(list(np.random.randint(step_max + 1, size=num_states)))

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Generated %i start/goal states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # get state action
    start_time = time.time()
    actions: List[Action] = env.get_state_action_rand(states)

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
    env.next_state_rand(states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Got %i random next states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

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

    queue1.put(env)

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


def test_envstartgoalrw(env: EnvStartGoalRW, num_states: int):
    # generate start/goal states
    start_time = time.time()
    states: List[State] = env.get_start_states(num_states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Generated %i start states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))


def test_envenumerableacts(env: EnvEnumerableActs, states: List[State]):
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


def init_nnet(heur_nnet: NNetPar) -> Tuple[nn.Module, torch.device]:
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    nnet: nn.Module = heur_nnet.get_nnet()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    return nnet, device


def heur_fn_out(heur_nnet: NNetPar, heur_fn: NNetCallable, states: List[State], goals: List[Goal], actions: List[Action]):
    if isinstance(heur_nnet, HeurNNetV):
        heur_fn(states, goals)
    elif isinstance(heur_nnet, HeurNNetQ):
        heur_fn(states, goals, [[action] for action in actions])
    else:
        raise ValueError(f"Unknown heur fn class {heur_fn}")


def test_heur_nnet(heur_nnet: NNetPar, states: List[State], goals: List[Goal], actions: List[Action]):
    # nnet format
    start_time = time.time()
    if isinstance(heur_nnet, HeurNNetV):
        heur_nnet.to_np(states, goals)
    elif isinstance(heur_nnet, HeurNNetQ):
        heur_nnet.to_np(states, goals, [[action] for action in actions])
    else:
        raise ValueError(f"Unknown heur nnet class {heur_nnet}")
    elapsed_time = time.time() - start_time
    states_per_sec = len(states) / elapsed_time
    print("Converted %i states and goals to nnet format in "
          "%s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # initialize nnet
    nnet, device = init_nnet(heur_nnet)
    print("")
    heur_fn: NNetCallable = heur_nnet.get_nnet_fn(nnet, None, device)
    heur_fn_out(heur_nnet, heur_fn, states, goals, actions)

    # nnet heuristic
    start_time = time.time()
    heur_fn_out(heur_nnet, heur_fn, states, goals, actions)

    nnet_time = time.time() - start_time
    states_per_sec = len(states) / nnet_time
    print("Computed heuristic for %i states in %s seconds (%.2f/second)" % (len(states), nnet_time, states_per_sec))


def test(env: Env, heur_nnet: NNetPar, num_states: int, step_max: int):
    states, goals, actions = test_env(env, num_states, step_max)
    if isinstance(env, EnvStartGoalRW):
        test_envstartgoalrw(env, num_states)
    if isinstance(env, EnvEnumerableActs):
        test_envenumerableacts(env, states)

    test_heur_nnet(heur_nnet, states, goals, actions)
    nnet, device = init_nnet(heur_nnet)
    heur_fn = heur_nnet.get_nnet_fn(nnet, None, device)
    search: BWQSEnum = BWQSEnum(env, heur_fn)
    nnet.eval()
    search.add_instances([states[0]], [goals[0]], [InstArgsBWQS()])
    instance = search.instances[0]
    while any([not instance.finished() for instance in search.instances]):
        node_q_acts = search.step()
        for node_q_act in node_q_acts:
            node_q = node_q_act.node
            print(node_q.state)
            print(instance.itr, node_q.is_solved, env.is_solved([node_q.state], [goals[0]]), node_q.path_cost,
                  instance.lb, instance.ub)
        breakpoint()
