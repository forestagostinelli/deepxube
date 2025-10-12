from typing import List, Dict, Tuple, Any, Generic, TypeVar, Optional
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess

import numpy as np
import torch
from numpy.typing import NDArray

from deepxube.nnet.nnet_utils import NNetParInfo
from deepxube.base.env import Env, State, Goal, Action, EnvEnumerableActs
from deepxube.base.heuristic import NNetPar, NNetCallable, HeurFnV, HeurFnQ, HeurNNetV, HeurNNetQ
from deepxube.base.pathfinding import PathFind, PathFindV, PathFindQ, Instance, InstArgs, NodeV, NodeQ, NodeQAct
from deepxube.nnet import nnet_utils
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, print_pathfindperf
from deepxube.training.train_utils import ReplayBuffer
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd
from deepxube.utils.misc_utils import split_evenly_w_max
from deepxube.utils.timing_utils import Times

from torch.utils.tensorboard import SummaryWriter


@dataclass
class UpHeurArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data
    balance.

    :param up_itrs: How many iterations to wait for updating target network
    :param up_gen_itrs: How many iterations worth of data to generate per udpate
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_search_itrs: Maximum number of pathfinding iterationos to take from generated problem instances
    :param up_batch_size: Maximum number of searches to do at a time. Helps manage memory.
    Decrease if memory is running out during updater.
    :param up_nnet_batch_size: Batch size of each nnet used for each process updater. Make smaller if running out
    of memory.
    Increasing this number could make the heuristic function more robust to depression regions.
    """
    up_itrs: int
    up_gen_itrs: int
    up_procs: int
    up_search_itrs: int
    up_batch_size: int
    up_nnet_batch_size: int


def get_data_from_procs(num_gen: int, from_q: Queue, to_q: Queue, procs: List[BaseProcess],
                        rb: ReplayBuffer, start_time_gen: float, writer: SummaryWriter,
                        train_itr: int) -> Dict[int, PathFindPerf]:
    # getting data from processes
    times_up: Times = Times()
    display_counts: NDArray[np.int_] = np.linspace(0, num_gen, 10, dtype=int)
    num_gen_curr: int = 0
    ctgs_min: float = np.inf
    ctgs_max: float = -np.inf
    ctgs_mean: float = 0
    while num_gen_curr < num_gen:
        inputs_nnet_shm_get_l, ctgs_shm_get_l = from_q.get()
        start_time = time.time()
        for inputs_nnet_shm_get, ctgs_shm_get in zip(inputs_nnet_shm_get_l, ctgs_shm_get_l, strict=True):
            # put to rb
            inputs_nnet_get: List[NDArray] = []
            for inputs_idx in range(len(inputs_nnet_shm_get)):
                inputs_nnet_get.append(inputs_nnet_shm_get[inputs_idx].array.copy())
            ctgs_get: NDArray = ctgs_shm_get.array.copy()
            rb.add(inputs_nnet_get + [ctgs_get])

            # status tracking
            num_gen_curr += ctgs_get.shape[0]
            ctgs_min = min(ctgs_get.min(), ctgs_min)
            ctgs_max = max(ctgs_get.max(), ctgs_max)
            ctgs_mean = ctgs_get.sum()

            # unlink shared mem
            for arr_shm in inputs_nnet_shm_get + [ctgs_shm_get]:
                arr_shm.close()
                arr_shm.unlink()
        times_up.record_time("rb", time.time() - start_time)
        if num_gen_curr >= min(display_counts):
            print(f"{num_gen_curr}/{num_gen} instances (%.2f%%) "
                  f"(Tot time: %.2f)" % (100 * num_gen_curr / num_gen, time.time() - start_time_gen))
            display_counts = display_counts[num_gen_curr < display_counts]
    ctgs_mean = ctgs_mean/float(num_gen_curr)

    # sending stop signal
    for _ in procs:
        to_q.put(None)

    # get summary from processes
    step_to_pathperf: Dict[int, PathFindPerf] = dict()
    for _ in procs:
        times_up_i, step_to_pathperf_i = from_q.get()
        times_up.add_times(times_up_i)
        for step_num_perf, pathperf in step_to_pathperf_i.items():
            if step_num_perf not in step_to_pathperf.keys():
                step_to_pathperf[step_num_perf] = PathFindPerf()
            step_to_pathperf[step_num_perf] = step_to_pathperf[step_num_perf].comb_perf(pathperf)

    # summary
    print(f"Generated {format(num_gen_curr, ',')} training instances, "
          f"Replay buffer size: {format(rb.size(), ',')}")
    print(f"Cost-to-go (mean/min/max): {ctgs_mean:.2f}/{ctgs_min:.2f}/{ctgs_max:.2f}")
    print(f"Times - {times_up.get_time_str()}")

    writer.add_scalar("ctgs (mean)", ctgs_mean, train_itr)
    writer.add_scalar("ctgs (min)", ctgs_min, train_itr)
    writer.add_scalar("ctgs (max)", ctgs_max, train_itr)

    return step_to_pathperf


def _put_from_q(inputs_nnet_l: List[List[NDArray]], ctgs_backup_l: List[NDArray], from_q: Queue, times: Times):
    start_time = time.time()

    inputs_nnet_shm_l: List[List[SharedNDArray]] = []
    ctgs_backup_shm_l: List[SharedNDArray] = []
    for inputs_nnet, ctgs_backup in zip(inputs_nnet_l, ctgs_backup_l, strict=True):
        inputs_nnet_shm_l.append([np_to_shnd(inputs_nnet_i) for inputs_nnet_i in inputs_nnet])
        ctgs_backup_shm_l.append(np_to_shnd(ctgs_backup))

    from_q.put((inputs_nnet_shm_l, ctgs_backup_shm_l))

    for inputs_nnet_shm, ctgs_backup_shm in zip(inputs_nnet_shm_l, ctgs_backup_shm_l, strict=True):
        for arr_shm in inputs_nnet_shm + [ctgs_backup_shm]:
            arr_shm.close()

    times.record_time("put", time.time() - start_time)


def _update_perf(insts_rem: List[Instance], step_to_pathperf: Dict[int, PathFindPerf]):
    for inst_rem in insts_rem:
        step_num_inst: int = int(inst_rem.inst_info[0])
        if step_num_inst not in step_to_pathperf.keys():
            step_to_pathperf[step_num_inst] = PathFindPerf()
        step_to_pathperf[step_num_inst].update_perf(inst_rem)


E = TypeVar('E', bound=Env)
HNet = TypeVar('HNet', bound=NNetPar)
H = TypeVar('H', bound=NNetCallable)
P = TypeVar('P', bound=PathFind)


class Update(ABC, Generic[E, HNet, H, P]):
    @staticmethod
    def _send_work_to_q(up_args: UpHeurArgs, num_gen: int, ctx) -> Queue:
        num_searches: int = num_gen // up_args.up_search_itrs
        print(f"Generating {format(num_gen, ',')} training instances with {format(num_searches, ',')} searches")

        assert num_gen % up_args.up_search_itrs == 0, (f"Number of instances to generate per for this updater "
                                                       f"{num_gen} is not divisible by the max number of "
                                                       f"pathfinding iterations to take during the "
                                                       f"updater ({up_args.up_search_itrs})")
        to_q: Queue = ctx.Queue()
        num_to_send_per: List[int] = split_evenly_w_max(num_searches, up_args.up_procs, up_args.up_batch_size)
        for num_to_send_per_i in num_to_send_per:
            if num_to_send_per_i > 0:
                to_q.put(num_to_send_per_i)

        return to_q

    @staticmethod
    def get_heur_fn_runners(heur_nnet: HNet, heur_file: str, up_args: UpHeurArgs, device: torch.device, on_gpu: bool,
                            all_zeros: bool) -> Tuple[List[NNetParInfo], List[BaseProcess]]:
        nnet_par_infos, nnet_procs = nnet_utils.start_nnet_fn_runners(heur_nnet.get_nnet, up_args.up_procs,
                                                                      heur_file, device, on_gpu, all_zeros=all_zeros,
                                                                      clip_zero=True,
                                                                      batch_size=up_args.up_nnet_batch_size)
        return nnet_par_infos, nnet_procs

    @staticmethod
    def print_update_summary(step_to_search_perf: Dict[int, PathFindPerf], writer: SummaryWriter, train_itr: int):
        per_solved_l: List[float] = []
        path_cost_ave_l: List[float] = []
        search_itrs_ave_l: List[float] = []
        for search_perf in step_to_search_perf.values():
            per_solved_i, path_cost_ave_i, search_itrs_ave_i = search_perf.stats()
            per_solved_l.append(per_solved_i)
            if per_solved_i > 0.0:
                path_cost_ave_l.append(path_cost_ave_i)
                search_itrs_ave_l.append(search_itrs_ave_i)
        per_solved_ave: float = float(np.mean(per_solved_l))
        path_costs_ave: float = float(np.mean(path_cost_ave_l))
        search_itrs_ave: float = float(np.mean(search_itrs_ave_l))
        print(f"%solved: {per_solved_ave:.2f}, path_costs: {path_costs_ave:.3f}, "
              f"search_itrs: {search_itrs_ave:.3f} (equally weighted across step numbers)")
        writer.add_scalar("solved (update)", per_solved_ave, train_itr)
        writer.add_scalar("path_cost (update)", path_costs_ave, train_itr)
        writer.add_scalar("search_itrs (update)", search_itrs_ave, train_itr)

        print_pathfindperf(step_to_search_perf)

    @staticmethod
    @abstractmethod
    def get_input_shapes_dtypes(env: E, heur_nnet: HNet) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        pass

    @classmethod
    @abstractmethod
    def get_update_data(cls, env: E, heur_nnet: HNet, heur_file: str, all_zeros: bool, up_args: UpHeurArgs,
                        step_max: int, step_probs: NDArray, num_gen: int, rb: ReplayBuffer, device: torch.device,
                        on_gpu: bool, writer: SummaryWriter, train_itr: int) -> Dict[int, PathFindPerf]:
        pass

    def __init__(self, env: E, heur_nnet: HNet, heur_nnet_par_info: NNetParInfo, up_args: UpHeurArgs):
        self.env: E = env
        self.heur_nnet: HNet = heur_nnet
        self.heur_nnet_par_info: NNetParInfo = heur_nnet_par_info
        self.heur_fn: Optional[H] = None
        self.up_args: UpHeurArgs = up_args

    @abstractmethod
    def initialize_fns(self):
        pass

    @abstractmethod
    def get_pathfind(self) -> P:
        pass

    @abstractmethod
    def step_get_in_out_np(self, pathfind: P, times: Times) -> Tuple[List[NDArray], List[float]]:
        pass

    def update_runner(self, gen_step_max: int, to_q: Queue, from_q: Queue, step_probs: NDArray):
        times: Times = Times()

        self.initialize_fns()
        step_to_pathperf: Dict[int, PathFindPerf] = dict()
        while True:
            batch_size = to_q.get()
            if batch_size is None:
                break

            pathfind: P = self.get_pathfind()

            insts_rem: List[Instance] = []
            inputs_nnet_l: List[List[NDArray]] = []
            ctgs_backup_l: List[NDArray] = []
            for _ in range(self.up_args.up_search_itrs):
                # add instances
                self._add_instances(pathfind, insts_rem, gen_step_max, batch_size, step_probs, times)
                assert len(pathfind.instances) == batch_size, f"Values were {len(pathfind.instances)} and {batch_size}"

                # step and to_np
                inputs_nnet, ctgs_backup = self.step_get_in_out_np(pathfind, times)
                assert len(ctgs_backup) == batch_size, f"Values were {len(ctgs_backup)} and {batch_size}"

                # put
                inputs_nnet_l.append(inputs_nnet)
                ctgs_backup_l.append(np.array(ctgs_backup))

                # remove instances
                insts_rem = pathfind.remove_finished_instances(self.up_args.up_search_itrs)

                # pathfinding performance
                _update_perf(insts_rem, step_to_pathperf)

            _put_from_q(inputs_nnet_l, ctgs_backup_l, from_q, times)
            times.add_times(pathfind.times, path=["pathfinding"])

        from_q.put((times, step_to_pathperf))

    def _add_instances(self, pathfind: P, insts_rem: List[Instance], gen_step_max: int, batch_size: int,
                       step_probs: NDArray, times: Times):
        if (len(pathfind.instances) == 0) or (len(insts_rem) > 0):
            times_states: Times = Times()
            # get steps generate
            start_time = time.time()
            steps_gen: List[int]
            if len(pathfind.instances) == 0:
                steps_gen = list(np.random.choice(gen_step_max + 1, size=batch_size, p=step_probs))
            else:
                steps_gen = [int(inst.inst_info[0]) for inst in insts_rem]
            times_states.record_time("steps_gen", time.time() - start_time)

            # generate states
            states_gen, goals_gen = self.env.get_start_goal_pairs(steps_gen, times=times_states)
            times.add_times(times_states, ["get_states"])

            # get instance information and kwargs
            start_time = time.time()
            inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
            times.record_time("inst_info", time.time() - start_time)

            # add instances
            pathfind.add_instances(states_gen, goals_gen, self._get_inst_args(len(states_gen)), inst_infos=inst_infos,
                                   compute_init_heur=True)

    @abstractmethod
    def _get_inst_args(self, num: int) -> List[InstArgs]:
        pass


class UpdateHeur(Update[E, HNet, H, P], ABC):
    def initialize_fns(self):
        self.heur_fn = self.heur_nnet.get_nnet_par_fn(self.heur_nnet_par_info)

    @classmethod
    def get_update_data(cls, env: E, heur_nnet: HNet, heur_file: str, all_zeros: bool, up_args: UpHeurArgs,
                        step_max: int, step_probs: NDArray, num_gen: int, rb: ReplayBuffer, device: torch.device,
                        on_gpu: bool, writer: SummaryWriter, train_itr: int) -> Dict[int, PathFindPerf]:
        start_time_gen = time.time()
        # put work information on to_q
        ctx = get_context("spawn")
        to_q: Queue = cls._send_work_to_q(up_args, num_gen, ctx)

        # parallel heuristic functions
        nnet_par_infos, nnet_procs = cls.get_heur_fn_runners(heur_nnet, heur_file, up_args, device, on_gpu, all_zeros)

        # start updater procs
        updaters: List[UpdateHeur] = [cls(env, heur_nnet, nnet_par_info, up_args) for nnet_par_info in nnet_par_infos]

        from_q: Queue = ctx.Queue()
        procs: List[BaseProcess] = []
        for updater in updaters:
            proc: BaseProcess = ctx.Process(target=updater.update_runner, args=(step_max, to_q, from_q, step_probs))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        # getting data from procs
        step_to_pathperf = get_data_from_procs(num_gen, from_q, to_q, procs, rb, start_time_gen, writer, train_itr)

        # clean up clean up everybody do your share
        nnet_utils.stop_nnet_runners(nnet_procs, nnet_par_infos)
        for proc in procs:
            proc.join()

        return step_to_pathperf


PV = TypeVar('PV', bound=PathFindV)


class UpdateHeurV(UpdateHeur[EnvEnumerableActs, HeurNNetV[State, Goal], HeurFnV[State, Goal], PV], ABC):
    @staticmethod
    def get_input_shapes_dtypes(env: EnvEnumerableActs, heur_nnet: HeurNNetV) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = env.get_start_goal_pairs([0])
        inputs_nnet: List[NDArray[Any]] = heur_nnet.to_np(states, goals)

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dypes

    def step_get_in_out_np(self, pathfind: PV, times: Times) -> Tuple[List[NDArray], List[float]]:
        # take a step
        nodes_popped: List[NodeV] = pathfind.step()

        # to np
        start_time = time.time()
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        ctgs_backup: List[float] = [node.backup() for node in nodes_popped]
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        inputs_np: List[NDArray] = self.heur_nnet.to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)
        return inputs_np, ctgs_backup


PQ = TypeVar('PQ', bound=PathFindQ)


class UpdateHeurQ(UpdateHeur[E, HeurNNetQ[State, Action, Goal], HeurFnQ[State, Goal, Action], PQ], ABC):
    @staticmethod
    def get_input_shapes_dtypes(env: E, heur_nnet: HeurNNetQ) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = env.get_start_goal_pairs([0])
        actions: List[Action] = env.get_state_action_rand(states)
        inputs_nnet: List[NDArray[Any]] = heur_nnet.to_np(states, goals, [[action] for action in actions])

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dypes

    @abstractmethod
    def get_qvals(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        pass

    def q_learning_backup(self, states: List[State], goals: List[Goal], actions: List[Action],
                          is_solved_l: List[bool]) -> Tuple[List[State], List[float]]:
        states_next, tcs = self.env.next_state(states, actions)

        # min cost-to-go for next state
        qvals_next_l: List[List[float]] = self.get_qvals(states_next, goals)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        return states_next, ctg_backups.tolist()

    def step_get_in_out_np(self, pathfind: PQ, times: Times) -> Tuple[List[NDArray], List[float]]:
        # take a step
        nodeqacts: List[NodeQAct] = pathfind.step()
        assert len(nodeqacts) == len(pathfind.instances), f"Values were {len(nodeqacts)} and {pathfind.instances}"

        # get backup for node_q_acts with actions that are not none
        states: List[State] = []
        goals: List[Goal] = []
        actions: List[Action] = []
        ctgs_backup: List[float] = []
        node_q_l_up: List[NodeQ] = []
        for nodeq_act in nodeqacts:
            action: Optional[Action] = nodeq_act.action
            node_q: NodeQ = nodeq_act.node
            if action is not None:
                ctg_backup: float = node_q.backup_act(action)
                states.append(node_q.state)
                goals.append(node_q.goal)
                actions.append(action)
                ctgs_backup.append(ctg_backup)
            else:
                node_q_l_up.append(node_q)

        # get backup of initial node_q_act with random action
        start_time = time.time()
        states_up, goals_up, actions_up, ctgs_backup_up = self.update_any_action(node_q_l_up)
        states.extend(states_up)
        goals.extend(goals_up)
        actions.extend(actions_up)
        ctgs_backup.extend(ctgs_backup_up)
        assert len(states) == len(goals) == len(actions) == len(ctgs_backup), \
            f"Values were {len(states)}, {len(goals)}, {len(actions)}, {len(ctgs_backup)}, "
        times.record_time("backup_1st", time.time() - start_time)

        # to_np
        start_time = time.time()
        inputs_np: List[NDArray] = self.heur_nnet.to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)
        return inputs_np, ctgs_backup

    def update_any_action(self, node_q_l: List[NodeQ]) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        if len(node_q_l) == 0:
            return [], [], [], []
        states: List[State] = []
        goals: List[Goal] = []
        is_solved_l: List[bool] = []
        for node_q in node_q_l:
            states.append(node_q.state)
            goals.append(node_q.goal)

            # act_probs: List[float] = boltzmann([-q_value for q_value in node_q.q_values], self.temp)
            # act_idx: int = int(np.random.multinomial(1, act_probs, size=1).argmax())
            # actions.append(random.choice(node_q.actions))
            is_solved_l.append(node_q.is_solved)

        actions: List[Action] = self.env.get_state_action_rand(states)

        # do q-learning backup
        ctgs_backup: List[float] = self.q_learning_backup(states, goals, actions, is_solved_l)[1]

        return states, goals, actions, ctgs_backup


class UpdateHeurQEnum(UpdateHeurQ[EnvEnumerableActs, PQ], ABC):
    def get_qvals(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        assert self.heur_fn is not None
        actions_next: List[List[Action]] = self.env.get_state_actions(states)
        qvals: List[List[float]] = self.heur_fn(states, goals, actions_next)

        return qvals
