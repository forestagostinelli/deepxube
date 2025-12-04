from typing import List, Dict, Tuple, Any, Generic, TypeVar, Optional, cast
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.process import BaseProcess
from multiprocessing.context import SpawnContext  # noqa

import numpy as np
import torch
from numpy.typing import NDArray

from deepxube.nnet.nnet_utils import NNetParInfo, NNetCallable, NNetPar
from deepxube.base.env import Env, State, Goal, Action, EnvEnumerableActs
from deepxube.base.heuristic import HeurNNet, HeurFnV, HeurFnQ, HeurNNetV, HeurNNetQ
from deepxube.base.pathfinding import PathFind, PathFindV, PathFindQ, Instance, Node, NodeV, NodeQ, NodeQAct
from deepxube.nnet import nnet_utils
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, print_pathfindperf
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd
from deepxube.utils.misc_utils import split_evenly_w_max
from deepxube.utils.timing_utils import Times

import copy
from torch.multiprocessing import get_context


@dataclass
class UpArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data
    balance.

    :param step_max: Maximum number of steps to take when generating data
    :param up_itrs: How many iterations to wait for updating target network
    :param up_gen_itrs: How many iterations worth of data to generate per udpate
    :param up_procs: Number of parallel workers used to compute updated cost-to-go values
    :param up_search_itrs: Maximum number of pathfinding iterationos to take from generated problem instances
    :param up_batch_size: Maximum number of searches to do at a time. Helps manage memory.
    Decrease if memory is running out during updater.
    :param up_nnet_batch_size: Batch size of each nnet used for each process updater. Make smaller if running out
    of memory.
    :param up_v: True if update is verbose.
    Increasing this number could make the heuristic function more robust to depression regions.
    """
    step_max: int
    up_itrs: int
    up_gen_itrs: int
    up_procs: int
    up_search_itrs: int
    up_batch_size: int
    up_nnet_batch_size: int
    up_v: bool = False


def _put_from_q(data_l: List[List[NDArray]], from_q: Queue, times: Times) -> None:
    start_time = time.time()

    data_shm_l: List[List[SharedNDArray]] = []
    for data in data_l:
        data_shm_l.append([np_to_shnd(data_i) for data_i in data])

    from_q.put(data_shm_l)

    for data_shm in data_shm_l:
        for arr_shm in data_shm:
            arr_shm.close()

    times.record_time("put", time.time() - start_time)


E = TypeVar('E', bound=Env)
N = TypeVar('N', bound=Node)
Inst = TypeVar('Inst', bound=Instance)
P = TypeVar('P', bound=PathFind)


class Update(ABC, Generic[E, N, Inst, P]):
    @staticmethod
    def get_nnet_fn_runners(nnet_par: NNetPar, nnet_file: str, up_args: UpArgs, device: torch.device,
                            on_gpu: bool) -> Tuple[List[NNetParInfo], List[BaseProcess]]:
        nnet_par_infos, nnet_procs = nnet_utils.start_nnet_fn_runners(nnet_par.get_nnet, up_args.up_procs, nnet_file,
                                                                      device, on_gpu,
                                                                      batch_size=up_args.up_nnet_batch_size)
        return nnet_par_infos, nnet_procs

    @staticmethod
    def _update_perf(insts: List[Inst], step_to_pathperf: Dict[int, PathFindPerf]) -> None:
        for inst in insts:
            step_num_inst: int = int(inst.inst_info[0])
            if step_num_inst not in step_to_pathperf.keys():
                step_to_pathperf[step_num_inst] = PathFindPerf()
            step_to_pathperf[step_num_inst].update_perf(inst)

    @staticmethod
    def _send_work_to_q(up_args: UpArgs, num_gen: int, ctx: SpawnContext, train_batch_size: int, verbose: bool) -> Queue:
        num_searches: int = num_gen // up_args.up_search_itrs
        if verbose:
            print(f"Generating {format(num_gen, ',')} training instances with {format(num_searches, ',')} searches")

        assert num_gen % up_args.up_search_itrs == 0, (f"Number of instances to generate per for this updater "
                                                       f"{num_gen} is not divisible by the max number of "
                                                       f"pathfinding iterations to take during the "
                                                       f"updater ({up_args.up_search_itrs})")
        to_q: Queue = ctx.Queue()
        num_to_send_per: List[int] = split_evenly_w_max(num_searches, up_args.up_procs,
                                                        min(up_args.up_batch_size, train_batch_size))
        for num_to_send_per_i in num_to_send_per:
            if num_to_send_per_i > 0:
                to_q.put(num_to_send_per_i)

        return to_q

    def __init__(self, env: E, up_args: UpArgs):
        self.env: E = env
        self.up_args: UpArgs = up_args
        self.targ_update_num: Optional[int] = None
        self.nnet_par_dict: Dict[str, NNetPar] = dict()
        self.nnet_file_dict: Dict[str, str] = dict()
        for nnet_name, nnet_file, nnet_par in env.get_nnet_pars():
            self.add_nnet_par(nnet_name, nnet_par)
            self.set_nnet_file(nnet_name, nnet_file)
        self.nnet_par_info_dict: Dict[str, NNetParInfo] = dict()
        self.nnet_fn_dict: Dict[str, NNetCallable] = dict()

        # update info
        self.nnet_runner_dict: Dict[str, Tuple[List[NNetParInfo], List[BaseProcess]]] = dict()
        self.procs: List[BaseProcess] = []
        self.to_q: Optional[Queue] = None
        self.from_q: Optional[Queue] = None
        self.num_generated: int = 0
        self.to_main_q: Optional[Queue] = None
        self.from_main_q: Optional[Queue] = None
        self.main_q_idx: int = 0

    def get_nnet_fn_runner_dict(self, device: torch.device,
                                on_gpu: bool) -> Dict[str, Tuple[List[NNetParInfo], List[BaseProcess]]]:
        nnet_runner_dict: Dict[str, Tuple[List[NNetParInfo], List[BaseProcess]]] = dict()
        for nnet_name, nnet_par in self.nnet_par_dict.items():
            nnet_file: str = self.nnet_file_dict[nnet_name]
            nnet_par_infos, nnet_procs = self.get_nnet_fn_runners(nnet_par, nnet_file, self.up_args, device, on_gpu)
            nnet_runner_dict[nnet_name] = (nnet_par_infos, nnet_procs)
        return nnet_runner_dict

    def set_nnet_par_info(self, nnet_name: str, nnet_par_info: NNetParInfo) -> None:
        assert nnet_name in self.nnet_par_dict.keys(), f"{nnet_name} not in dict"
        assert nnet_name in self.nnet_file_dict.keys(), f"{nnet_name} not in dict"
        assert nnet_name not in self.nnet_par_info_dict.keys(), f"{nnet_name} already in dict"
        self.nnet_par_info_dict[nnet_name] = nnet_par_info

    def add_nnet_par(self, nnet_name: str, nnet_par: NNetPar) -> None:
        assert nnet_name not in self.nnet_par_dict.keys(), f"{nnet_name} already in dict"
        self.nnet_par_dict[nnet_name] = nnet_par

    def set_nnet_file(self, nnet_name: str, nnet_file: str) -> None:
        assert nnet_name in self.nnet_par_dict.keys(), f"{nnet_name} should already be in dict, but it is not"
        self.nnet_file_dict[nnet_name] = nnet_file

    def set_main_qs(self, to_main_q: Queue, from_main_q: Queue, main_q_idx: int) -> None:
        self.to_main_q = to_main_q
        self.from_main_q = from_main_q
        self.main_q_idx = main_q_idx

    def start_update(self, step_probs: List[int], num_gen: int, device: torch.device, on_gpu: bool,
                     targ_update_num: int, train_batch_size: int) -> Tuple[Queue, List[Queue]]:
        self.set_targ_update_num(targ_update_num)

        # start updater procs
        # TODO implement safer copy?
        updaters: List[Update] = [copy.deepcopy(self) for _ in range(self.up_args.up_procs)]

        # parallel heuristic functions
        ctx = get_context("spawn")
        self.nnet_runner_dict = self.get_nnet_fn_runner_dict(device, on_gpu)
        for proc_itr, updater in enumerate(updaters):
            for nnet_name in self.nnet_runner_dict.keys():
                updater.set_nnet_par_info(nnet_name, self.nnet_runner_dict[nnet_name][0][proc_itr])

        # put work information on to_q
        self.to_q = self._send_work_to_q(self.up_args, num_gen, ctx, train_batch_size, self.up_args.up_v)

        self.from_q = ctx.Queue()
        self.procs = []
        to_main_q: Queue = ctx.Queue()
        from_main_qs: List[Queue] = []
        for updater_idx, updater in enumerate(updaters):
            from_main_q: Queue = ctx.Queue(1)
            from_main_qs.append(from_main_q)
            proc: BaseProcess = ctx.Process(target=updater.update_runner, args=(self.to_q, self.from_q, step_probs,
                                                                                to_main_q, from_main_q, updater_idx))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

        return to_main_q, from_main_qs

    def get_update_data(self) -> List[List[NDArray]]:
        assert self.from_q is not None
        data_l: List[List[NDArray]] = []
        data_get_l: List[List[SharedNDArray]] = self.from_q.get()
        for data_get in data_get_l:
            # to np
            data_get_np: List[NDArray] = []
            for data_get_i in data_get:
                data_get_np.append(data_get_i.array.copy())
            data_l.append(data_get_np)

            # status tracking
            self.num_generated += data_get_np[0].shape[0]

            # unlink shared mem
            for arr_shm in data_get:
                arr_shm.close()
                arr_shm.unlink()

        return data_l

    def end_update(self) -> Dict[int, PathFindPerf]:
        assert (self.to_q is not None) and (self.from_q is not None)
        # sending stop signal
        for _ in self.procs:
            self.to_q.put(None)

        # get summary from processes
        step_to_pathperf: Dict[int, PathFindPerf] = dict()
        times_up: Times = Times()
        for _ in self.procs:
            times_up_i, step_to_pathperf_i = self.from_q.get()
            times_up.add_times(times_up_i)
            for step_num_perf, pathperf in step_to_pathperf_i.items():
                if step_num_perf not in step_to_pathperf.keys():
                    step_to_pathperf[step_num_perf] = PathFindPerf()
                step_to_pathperf[step_num_perf] = step_to_pathperf[step_num_perf].comb_perf(pathperf)

        # print
        if self.up_args.up_v:
            print(f"Generated {format(self.num_generated, ',')} training instances")
            print(f"Times - {times_up.get_time_str()}")
            print_pathfindperf(step_to_pathperf)

        # clean up clean up everybody do your share
        for nnet_par_infos, nnet_procs in self.nnet_runner_dict.values():
            nnet_utils.stop_nnet_runners(nnet_procs, nnet_par_infos)
        for proc in self.procs:
            proc.join()

        self.nnet_runner_dict = dict()
        self.procs = []
        self.to_q = None
        self.from_q = None
        self.num_generated = 0

        return step_to_pathperf

    def initialize_fns(self) -> None:
        for nnet_name in self.nnet_par_dict.keys():
            nnet: NNetPar = self.nnet_par_dict[nnet_name]
            nnet_par_info: NNetParInfo = self.nnet_par_info_dict[nnet_name]
            self.nnet_fn_dict[nnet_name] = nnet.get_nnet_par_fn(nnet_par_info, self.targ_update_num)
        self.env.set_nnet_fns(self.nnet_fn_dict)

    def get_up_args_repr(self) -> str:
        return self.up_args.__repr__()

    @abstractmethod
    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        pass

    @abstractmethod
    def get_pathfind(self) -> P:
        pass

    @abstractmethod
    def step(self, pathfind: P, times: Times) -> List[NDArray]:
        pass

    @abstractmethod
    def get_instance_data(self, instances: List[Inst], times: Times) -> List[NDArray]:
        pass

    def set_targ_update_num(self, targ_update_num: Optional[int]) -> None:
        self.targ_update_num = targ_update_num

    def update_runner(self, to_q: Queue, from_q: Queue, step_probs: List[int], to_main_q: Queue, from_main_q: Queue,
                      main_q_idx: int) -> None:
        times: Times = Times()
        # self.set_main_qs(to_main_q, from_main_q, main_q_idx)

        self.initialize_fns()
        step_to_pathperf: Dict[int, PathFindPerf] = dict()
        while True:
            batch_size = to_q.get()
            if batch_size is None:
                break

            pathfind: P = self.get_pathfind()

            insts_rem_all: List[Inst] = []
            insts_rem_last_itr: List[Inst] = []
            for _ in range(self.up_args.up_search_itrs):
                # add instances
                self._add_instances(pathfind, insts_rem_last_itr, batch_size, step_probs, times)
                assert len(pathfind.instances) == batch_size, f"Values were {len(pathfind.instances)} and {batch_size}"

                # step
                data: List[NDArray] = self.step(pathfind, times)
                _put_from_q([data], from_q, times)

                # remove instances
                insts_rem_last_itr = pathfind.remove_finished_instances(self.up_args.up_search_itrs)
                insts_rem_all.extend(insts_rem_last_itr)

            # _put_from_q([self.get_instance_data(insts_rem_all + pathfind.instances, times)], from_q, times)

            # pathfinding performance
            self._update_perf(insts_rem_all, step_to_pathperf)

            times.add_times(pathfind.times, path=["pathfinding"])

        from_q.put((times, step_to_pathperf))
        self.to_main_q = None
        self.from_main_q = None

    def _add_instances(self, pathfind: P, insts_rem: List[Inst], batch_size: int, step_probs: List[int],
                       times: Times) -> None:
        if (len(pathfind.instances) == 0) or (len(insts_rem) > 0):
            # get steps generate
            start_time = time.time()
            steps_gen: List[int]
            if len(insts_rem) > 0:
                steps_gen = [int(inst.inst_info[0]) for inst in insts_rem]
            else:
                steps_gen = np.random.choice(self.up_args.step_max + 1, size=batch_size,
                                             p=np.array(step_probs)).tolist()
            times.record_time("steps_gen", time.time() - start_time)

            # get instance information and kwargs
            start_time = time.time()
            inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
            times.record_time("inst_info", time.time() - start_time)

            instances: List[Inst] = self._get_instances(pathfind, steps_gen, inst_infos, times)

            # add instances
            start_time = time.time()
            pathfind.add_instances(instances)
            times.record_time("inst_add", time.time() - start_time)

    @abstractmethod
    def _get_instances(self, pathfind: P, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[Inst]:
        pass


HNet = TypeVar('HNet', bound=HeurNNet)
H = TypeVar('H', bound=NNetCallable)


class UpdateHasHeur(Update[E, N, Inst, P], Generic[E, N, Inst, P, HNet, H]):
    @staticmethod
    def heur_name() -> str:
        return 'heur'

    def set_heur_nnet(self, heur_nnet: HNet) -> None:
        self.add_nnet_par(self.heur_name(), heur_nnet)

    def set_heur_file(self, heur_file: str) -> None:
        self.set_nnet_file(self.heur_name(), heur_file)

    def get_heur_nnet(self) -> HNet:
        return cast(HNet, self.nnet_par_dict[self.heur_name()])

    def get_heur_fn(self) -> H:
        return cast(H, self.nnet_fn_dict[self.heur_name()])

    @abstractmethod
    def get_pathfind(self) -> P:
        pass


@dataclass
class UpHeurArgs:
    """
    :param up_args: Update args
    :param ub_heur_solns: if True, the target cost-to-go will be min(backup, path_cost_from_state)
    :param backup: 1 is Bellman and -1 is tree backup (i.e. Limited Horizon Bellman-based Learning)
    :param on_heur: if True, number of processes can affect order in which data is seen
    """
    up_args: UpArgs
    ub_heur_solns: bool
    backup: int
    on_heur: bool


class UpdateHeur(UpdateHasHeur[E, N, Inst, P, HNet, H], ABC):
    def __init__(self, env: E, up_heur_args: UpHeurArgs):
        super().__init__(env, up_heur_args.up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args

    def get_up_args_repr(self) -> str:
        return self.up_heur_args.__repr__()

    def get_targ_heur_fn(self) -> H:
        return cast(H, self.nnet_fn_dict[self.heur_name()])


PV = TypeVar('PV', bound=PathFindV)


class UpdateHeurV(UpdateHeur[E, NodeV, Inst, PV, HeurNNetV[State, Goal], HeurFnV[State, Goal]], ABC):
    def __init__(self, env: E, up_heur_args: UpHeurArgs):
        super().__init__(env, up_heur_args)

    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.env.get_start_goal_pairs([0])
        inputs_nnet: List[NDArray[Any]] = self.get_heur_nnet().to_np(states, goals)

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dypes

    def get_heur_fn(self) -> HeurFnV:
        if not self.up_heur_args.on_heur:
            return super().get_heur_fn()
        else:
            def heur_fn(states: List[State], goals: List[Goal]) -> List[float]:
                assert (self.to_main_q is not None) and (self.from_main_q is not None)
                inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals)
                self.to_main_q.put((self.main_q_idx, inputs_np))
                return cast(List[float], self.from_main_q.get())

            return heur_fn

    def step(self, pathfind: PV, times: Times) -> List[NDArray]:
        # take a step
        nodes_popped: List[NodeV] = pathfind.step()
        assert len(nodes_popped) == len(pathfind.instances), (f"Values were {len(nodes_popped)} and "
                                                              f"{len(pathfind.instances)}")

        # backup
        start_time = time.time()
        ctgs_backup: List[float] = []
        for node in nodes_popped:
            node.bellman_backup()
            assert node.backup_val is not None
            ctgs_backup.append(node.backup_val)
        times.record_time("backup", time.time() - start_time)

        # inputs
        start_time = time.time()
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]

    def get_instance_data(self, instances: List[Inst], times: Times) -> List[NDArray]:
        # root heur
        start_time = time.time()
        self.get_pathfind().set_root_heurs(instances)
        times.record_time("root_heur", time.time() - start_time)

        # get popped
        start_time = time.time()
        nodes_popped: List[NodeV] = []
        for instance in instances:
            root_node: NodeV = instance.root_node
            nodes_popped.extend(self.get_pathfind().get_expanded_nodes(root_node))
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        times.record_time("get_popped", time.time() - start_time)

        # get backup
        start_time = time.time()
        ctgs_backup: List[float] = []
        if self.up_heur_args.backup == 1:
            for node in nodes_popped:
                node.bellman_backup()
            if self.up_heur_args.ub_heur_solns:
                for node in nodes_popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                root_node = instance.root_node
                root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")

        for node in nodes_popped:
            assert node.backup_val is not None
            ctgs_backup.append(node.backup_val)
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)
        return inputs_np + [np.array(ctgs_backup)]

    def _get_root_nodes(self, pathfind: PV, steps_gen: List[int], times: Times) -> List[NodeV]:
        # get states/goals
        times_states: Times = Times()
        states_gen, goals_gen = self.env.get_start_goal_pairs(steps_gen, times=times_states)
        times.add_times(times_states, ["get_states"])

        # root nodes
        return pathfind.create_root_nodes(states_gen, goals_gen, compute_init_heur=True)


PQ = TypeVar('PQ', bound=PathFindQ)


class UpdateHeurQ(UpdateHeur[E, NodeQ, Inst, PQ, HeurNNetQ[State, Action, Goal], HeurFnQ[State, Goal, Action]], ABC):
    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.env.get_start_goal_pairs([0])
        actions: List[Action] = self.env.get_state_action_rand(states)
        inputs_nnet: List[NDArray[Any]] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dypes.append((tuple(), np.dtype(np.float64)))

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

    def step(self, pathfind: PQ, times: Times) -> List[NDArray]:
        # take a step
        nodeqacts: List[NodeQAct] = pathfind.step()
        assert len(nodeqacts) == len(pathfind.instances), f"Values were {len(nodeqacts)} and {len(pathfind.instances)}"

    def get_instance_data(self, instances: List[Inst], times: Times) -> List[NDArray]:
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
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)
        return inputs_np + [np.array(ctgs_backup)]

    def update_any_action(self, node_q_l: List[NodeQ]) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        if len(node_q_l) == 0:
            return [], [], [], []
        states: List[State] = []
        goals: List[Goal] = []
        is_solved_l: List[bool] = []
        for node_q in node_q_l:
            assert node_q.is_solved is not None
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

    def _get_root_nodes(self, pathfind: PQ, steps_gen: List[int], times: Times) -> List[NodeQ]:
        # get states/goals
        times_states: Times = Times()
        states_gen, goals_gen = self.env.get_start_goal_pairs(steps_gen, times=times_states)
        times.add_times(times_states, ["get_states"])

        # root nodes
        return pathfind.create_root_nodes(states_gen, goals_gen, compute_init_heur=True)


class UpdateHeurQEnum(UpdateHeurQ[EnvEnumerableActs, Inst, PQ], ABC):
    def get_qvals(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        actions_next: List[List[Action]] = self.env.get_state_actions(states)
        qvals: List[List[float]] = self.get_targ_heur_fn()(states, goals, actions_next)

        return qvals
