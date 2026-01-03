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

from deepxube.nnet.nnet_utils import NNetParInfo, NNetCallable, NNetPar, get_nnet_par_infos, start_nnet_fn_runners, stop_nnet_runners
from deepxube.base.domain import Domain, State, Goal, Action, ActsEnum
from deepxube.base.heuristic import HeurNNetPar, HeurFnV, HeurFnQ, HeurFn, HeurNNetParV, HeurNNetParQ
from deepxube.base.pathfinding import PathFind, PathFindVHeur, PathFindQ, Instance, Node, EdgeQ
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, print_pathfindperf
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd, get_nowait_noerr
from deepxube.utils.misc_utils import split_evenly_w_max
from deepxube.utils.timing_utils import Times

import copy
from torch.multiprocessing import get_context


# TODO par nnets per GPU?
@dataclass
class UpArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data
    balance.

    :param procs: Number of parallel workers used to compute update
    :param up_itrs: How many iterations worth of training instances to obtain for each update
    :param step_max: Maximum number of steps to take when generating problem instances.
    :param search_itrs: Maximum number of pathfinding iterationos to take for each generated problem instances
    States and corresponding goals seen during search will be added to training instances.
    :param up_gen_itrs: How many iterations worth of data to generate per udpate. If None, set to up_itrs
    :param up_batch_size: Maximum number of searches to do at a time. Helps manage memory.
    Decrease if memory is running out during updater. None if as large as possible
    :param nnet_batch_size: Batch size of each nnet used for each process updater. Make smaller if running out
    of memory. None if as large as possible.
    :param v: True if update is verbose.
    :param sync_main: if True, number of processes can affect order in which data is seen
    """
    procs: int
    up_itrs: int
    step_max: int
    search_itrs: int
    up_gen_itrs: Optional[int] = None
    up_batch_size: Optional[int] = None
    nnet_batch_size: Optional[int] = None
    sync_main: bool = False
    v: bool = False

    def get_up_gen_itrs(self) -> int:
        return self.up_itrs if (self.up_gen_itrs is None) else self.up_gen_itrs


@dataclass
class UpHeurArgs:
    """ Arguments when updating heuristic
    :param ub_heur_solns: if True, the target cost-to-go will be min(backup, path_cost_from_state)
    :param backup: 1 is Bellman and -1 is tree backup (i.e. Limited Horizon Bellman-based Learning)
    """
    ub_heur_solns: bool
    backup: int


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


D = TypeVar('D', bound=Domain)
P = TypeVar('P', bound=PathFind)


class Update(Generic[D, P], ABC):
    @staticmethod
    def _update_perf(insts: List[Instance], step_to_pathperf: Dict[int, PathFindPerf]) -> None:
        for inst in insts:
            step_num_inst: int = int(inst.inst_info[0])
            if step_num_inst not in step_to_pathperf.keys():
                step_to_pathperf[step_num_inst] = PathFindPerf()
            step_to_pathperf[step_num_inst].update_perf(inst)

    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs):
        self.domain: D = domain
        self.pathfind_name: str = pathfind_name
        self.pathfind_kwargs: Dict[str, Any] = pathfind_kwargs
        self.up_args: UpArgs = up_args
        self.targ_update_num: Optional[int] = None
        self.nnet_par_dict: Dict[str, NNetPar] = dict()
        self.nnet_file_dict: Dict[str, str] = dict()
        for nnet_name, nnet_file, nnet_par in domain.get_nnet_pars():
            self.add_nnet_par(nnet_name, nnet_par)
            self.set_nnet_file(nnet_name, nnet_file)
        self.nnet_par_info_dict: Dict[str, NNetParInfo] = dict()
        self.nnet_fn_dict: Dict[str, NNetCallable] = dict()

        # update info
        self.nnet_par_info_l_dict: Dict[str, List[NNetParInfo]] = dict()
        self.nnet_runner_proc_l_dict: Dict[str, List[BaseProcess]] = dict()
        self.procs: List[BaseProcess] = []
        self.to_q: Optional[Queue] = None
        self.from_q: Optional[Queue] = None
        self.num_generated: int = 0
        self.to_main_q: Optional[Queue] = None
        self.from_main_q: Optional[Queue] = None
        self.from_main_qs: List[Queue] = []
        self.q_id: int = 0
        self.nnet_par_info_main: Optional[NNetParInfo] = None

    def set_nnet_par_info_l_dict(self) -> None:
        for nnet_name in self.nnet_par_dict.keys():
            self.nnet_par_info_l_dict[nnet_name] = get_nnet_par_infos(self.up_args.procs)

    def start_nnet_runners(self, device: torch.device, on_gpu: bool) -> None:
        for nnet_name, nnet_par in self.nnet_par_dict.items():
            nnet_file: str = self.nnet_file_dict[nnet_name]
            nnet_par_infos: List[NNetParInfo] = self.nnet_par_info_l_dict[nnet_name]
            self.nnet_runner_proc_l_dict[nnet_name] = start_nnet_fn_runners(nnet_par.get_nnet, nnet_par_infos,
                                                                            nnet_file, device, on_gpu,
                                                                            batch_size=self.up_args.nnet_batch_size)

    def set_nnet_par_info(self, nnet_name: str, nnet_par_info: NNetParInfo) -> None:
        assert nnet_name in self.nnet_par_dict.keys(), f"{nnet_name} not in dict"
        assert nnet_name in self.nnet_file_dict.keys(), f"{nnet_name} not in dict"
        assert nnet_name not in self.nnet_par_info_dict.keys(), f"{nnet_name} already in dict"
        self.nnet_par_info_dict[nnet_name] = nnet_par_info

    def clear_nnet_fn_dict(self) -> None:
        self.nnet_fn_dict = dict()

    def add_nnet_par(self, nnet_name: str, nnet_par: NNetPar) -> None:
        assert nnet_name not in self.nnet_par_dict.keys(), f"{nnet_name} already in dict"
        self.nnet_par_dict[nnet_name] = nnet_par

    def set_nnet_file(self, nnet_name: str, nnet_file: str) -> None:
        assert nnet_name in self.nnet_par_dict.keys(), f"{nnet_name} should already be in dict, but it is not"
        self.nnet_file_dict[nnet_name] = nnet_file

    def set_main_qs(self, to_main_q: Queue, from_main_q: Queue, q_id: int) -> None:
        self.to_main_q = to_main_q
        self.from_main_q = from_main_q
        self.q_id = q_id
        self.nnet_par_info_main = NNetParInfo(self.to_main_q, self.from_main_q, self.q_id)

    def start_procs(self) -> Tuple[Queue, List[Queue]]:
        # start updater procs
        # TODO implement safer copy?
        updaters: List[Update] = [copy.deepcopy(self) for _ in range(self.up_args.procs)]

        # parallel heuristic functions
        ctx = get_context("spawn")
        to_main_q: Queue = ctx.Queue()
        self.from_main_qs = []
        self.set_nnet_par_info_l_dict()
        for proc_idx, updater in enumerate(updaters):
            from_main_q: Queue = ctx.Queue(1)
            self.from_main_qs.append(from_main_q)
            updater.set_main_qs(to_main_q, from_main_q, proc_idx)
            for nnet_name in self.nnet_par_info_l_dict.keys():
                updater.set_nnet_par_info(nnet_name, self.nnet_par_info_l_dict[nnet_name][proc_idx])

        self.to_q = ctx.Queue()
        self.from_q = ctx.Queue()
        self.procs = []

        for updater in updaters:
            proc: BaseProcess = ctx.Process(target=updater.update_runner, args=(self.to_q, self.from_q))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

        return to_main_q, self.from_main_qs

    def start_update(self, step_probs: List[int], num_gen: int, targ_update_num: Optional[int], train_batch_size: int,
                     device: torch.device, on_gpu: bool) -> None:
        # start parallel nnet runners
        self.set_targ_update_num(targ_update_num)
        self.start_nnet_runners(device, on_gpu)

        # put update data
        for proc_idx, from_main_q in enumerate(self.from_main_qs):
            from_main_q.put((step_probs, targ_update_num))

        # put work information on to_q
        assert self.to_q is not None

        num_searches: int = num_gen // self.up_args.search_itrs
        if self.up_args.v:
            print(f"Generating {format(num_gen, ',')} training instances with {format(num_searches, ',')} searches")

        assert num_gen % self.up_args.search_itrs == 0, (f"Number of instances to generate per for this updater {num_gen} is not divisible by the max number "
                                                         f"of pathfinding iterations to take during the updater ({self.up_args.search_itrs})")
        up_batch_size: int = train_batch_size if (self.up_args.up_batch_size is None) else self.up_args.up_batch_size
        num_to_send_per: List[int] = split_evenly_w_max(num_searches, self.up_args.procs,
                                                        min(up_batch_size, train_batch_size))
        for num_to_send_per_i in num_to_send_per:
            if num_to_send_per_i > 0:
                self.to_q.put(num_to_send_per_i)

    def get_update_data(self, nowait: bool = False) -> List[List[NDArray]]:
        assert self.from_q is not None
        data_l: List[List[NDArray]] = []
        data_get_l: Optional[List[List[SharedNDArray]]]
        if nowait:
            data_get_l = get_nowait_noerr(self.from_q)
        else:
            data_get_l = self.from_q.get()
        if data_get_l is None:
            return []

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
        if self.up_args.v:
            print(f"Generated {format(self.num_generated, ',')} training instances")
            print(f"Times - {times_up.get_time_str()}")
            print_pathfindperf(step_to_pathperf)

        # clean up clean up everybody do your share
        for nnet_name, nnet_par_infos in self.nnet_par_info_l_dict.items():
            nnet_procs: List[BaseProcess] = self.nnet_runner_proc_l_dict[nnet_name]
            stop_nnet_runners(nnet_procs, nnet_par_infos)

        self.num_generated = 0
        self.nnet_runner_proc_l_dict = dict()

        return step_to_pathperf

    def stop_procs(self) -> None:
        # sending stop signal
        for from_main_q in self.from_main_qs:
            from_main_q.put(None)

        # clean up clean up everybody do your share
        for proc in self.procs:
            proc.join()

        self.procs = []
        self.to_q = None
        self.from_q = None

    def initialize_fns(self, targ_update_num: Optional[int]) -> None:
        for nnet_name in self.nnet_par_dict.keys():
            nnet: NNetPar = self.nnet_par_dict[nnet_name]
            nnet_par_info: NNetParInfo = self.nnet_par_info_dict[nnet_name]
            self.nnet_fn_dict[nnet_name] = nnet.get_nnet_par_fn(nnet_par_info, targ_update_num)
        self.domain.set_nnet_fns(self.nnet_fn_dict)

    def get_up_args_repr(self) -> str:
        return self.up_args.__repr__()

    @abstractmethod
    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        pass

    def get_pathfind(self) -> P:
        pathfind_kwargs: Dict[str, Any] = self.pathfind_kwargs.copy()
        pathfind_kwargs["domain"] = self.domain
        return cast(P, pathfinding_factory.build_class(self.pathfind_name, pathfind_kwargs))

    @abstractmethod
    def _set_pathfind_nnet_fns(self, pathfind: P) -> None:
        pass

    @abstractmethod
    def _step(self, pathfind: P, times: Times) -> List[NDArray]:
        pass

    @abstractmethod
    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        pass

    def set_targ_update_num(self, targ_update_num: Optional[int]) -> None:
        self.targ_update_num = targ_update_num

    def update_runner(self, to_q: Queue, from_q: Queue) -> None:
        while True:
            assert self.from_main_q is not None
            data_q: Optional[Tuple[List[int], Optional[int]]] = self.from_main_q.get()
            if data_q is None:
                break
            times: Times = Times()

            step_probs, targ_update_num = data_q
            self.set_targ_update_num(targ_update_num)
            self.initialize_fns(targ_update_num)

            step_to_pathperf: Dict[int, PathFindPerf] = dict()
            while True:
                batch_size = to_q.get()
                if batch_size is None:
                    break

                pathfind: P = self.get_pathfind()
                self._set_pathfind_nnet_fns(pathfind)

                insts_rem_all: List[Instance] = []
                insts_rem_last_itr: List[Instance] = []
                for _ in range(self.up_args.search_itrs):
                    # add instances
                    self._add_instances(pathfind, insts_rem_last_itr, batch_size, step_probs, times)
                    assert len(pathfind.instances) == batch_size, f"Values were {len(pathfind.instances)} and {batch_size}"

                    # step
                    data: List[NDArray] = self._step(pathfind, times)
                    if self.up_args.sync_main:
                        _put_from_q([data], from_q, times)

                    # remove instances
                    insts_rem_last_itr = pathfind.remove_finished_instances(self.up_args.search_itrs)
                    insts_rem_all.extend(insts_rem_last_itr)

                if not self.up_args.sync_main:
                    _put_from_q([self._get_instance_data(insts_rem_all + pathfind.instances, times)], from_q, times)

                # pathfinding performance
                start_time = time.time()
                self._update_perf(insts_rem_all, step_to_pathperf)
                times.record_time("update_perf", time.time() - start_time)

                times.add_times(pathfind.times, path=["pathfinding"])

            from_q.put((times, step_to_pathperf))
            self.clear_nnet_fn_dict()
        self.to_main_q = None
        self.from_main_q = None
        self.nnet_par_info_main = None

    def _add_instances(self, pathfind: P, insts_rem: List[Instance], batch_size: int, step_probs: List[int],
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

            instances: List[Instance] = self._make_instances(pathfind, steps_gen, inst_infos, times)

            # add instances
            start_time = time.time()
            pathfind.add_instances(instances)
            times.record_time("inst_add", time.time() - start_time)

    @abstractmethod
    def _make_instances(self, pathfind: P, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[Instance]:
        pass


HNet = TypeVar('HNet', bound=HeurNNetPar)
H = TypeVar('H', bound=HeurFn)


class UpdateHasHeur(Update[D, P], Generic[D, P, HNet, H], ABC):
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
        if not self.up_args.sync_main:
            return self._get_heur_fn_from_dict()
        else:
            assert self.nnet_par_info_main is not None
            return cast(H, self.get_heur_nnet().get_nnet_par_fn(self.nnet_par_info_main, None))

    def _get_heur_fn_from_dict(self) -> H:
        return cast(H, self.nnet_fn_dict[self.heur_name()])


class UpdateHeur(UpdateHasHeur[D, P, HNet, H], ABC):
    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs, heur_nnet: HNet):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args
        self.set_heur_nnet(heur_nnet)

    def _get_targ_heur_fn(self) -> H:
        return self._get_heur_fn_from_dict()


class UpdateHeurV(UpdateHeur[Domain, PathFindVHeur, HeurNNetParV, HeurFnV]):
    def __init__(self, domain: Domain, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs,
                 heur_nnet: HeurNNetParV):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args, heur_nnet)
        self.nodes_popped: List[Node] = []

    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.get_start_goal_pairs([0])
        inputs_nnet: List[NDArray[Any]] = self.get_heur_nnet().to_np(states, goals)

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dtypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dtypes

    def _set_pathfind_nnet_fns(self, pathfind: PathFindVHeur) -> None:
        pathfind.set_heur_fn(self.get_heur_fn())

    def _step(self, pathfind: PathFindVHeur, times: Times) -> List[NDArray]:
        # take a step
        nodes_popped: List[Node] = pathfind.step()
        assert len(nodes_popped) == len(pathfind.instances), (f"Values were {len(nodes_popped)} and "
                                                              f"{len(pathfind.instances)}")
        if not self.up_args.sync_main:
            self.nodes_popped.extend(nodes_popped)
            return []
        else:
            # TODO implement for sync_main
            raise NotImplementedError

    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get backup
        start_time = time.time()
        ctgs_backup: List[float] = []
        if self.up_heur_args.backup == 1:
            for node in self.nodes_popped:
                node.bellman_backup()
            if self.up_heur_args.ub_heur_solns:
                for node in self.nodes_popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                root_node: Node = instance.root_node
                root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")

        for node in self.nodes_popped:
            ctgs_backup.append(node.backup_val)
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        states: List[State] = [node.state for node in self.nodes_popped]
        goals: List[Goal] = [node.goal for node in self.nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)

        self.nodes_popped = []
        return inputs_np + [np.array(ctgs_backup)]

    def _make_instances(self, pathfind: P, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[Instance]:
        # get states/goals
        times_states: Times = Times()
        states_gen, goals_gen = self.domain.get_start_goal_pairs(steps_gen, times=times_states)
        times.add_times(times_states, ["get_states"])

        return pathfind.make_instances(states_gen, goals_gen, inst_infos=inst_infos, compute_root_heur=False)


PQ = TypeVar('PQ', bound=PathFindQ)


def _split_init_vs_real_edges(edges: List[EdgeQ]) -> Tuple[List[EdgeQ], List[EdgeQ]]:
    edges_init: List[EdgeQ] = []
    edges_real: List[EdgeQ] = []
    for edge in edges:
        if edge.action is None:
            edges_init.append(edge)
        else:
            edges_real.append(edge)
    return edges_init, edges_real


class UpdateHeurQ(UpdateHeur[D, PQ, HeurNNetParQ, HeurFnQ], ABC):
    def __init__(self, domain: D, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs,
                 heur_nnet: HeurNNetParQ):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args, heur_nnet)
        self.edges_popped: List[EdgeQ] = []

    def get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.get_start_goal_pairs([0])
        actions: List[Action] = self.domain.get_state_action_rand(states)
        inputs_nnet: List[NDArray[Any]] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dtypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dtypes

    def _step(self, pathfind: PQ, times: Times) -> List[NDArray]:
        # take a step
        edges_popped: List[EdgeQ] = pathfind.step()
        assert len(edges_popped) == len(pathfind.instances), f"Values were {len(edges_popped)} and {len(pathfind.instances)}"

        if not self.up_args.sync_main:
            self.edges_popped.extend(edges_popped)
            return []
        else:
            start_time = time.time()
            states, goals, is_solved_l, actions, tcs, states_next = self._get_edge_data(edges_popped)
            ctgs_backup: List[float] = self._q_learning_backup_targ(goals, is_solved_l, tcs, states_next)
            times.record_time("backup_sync", time.time() - start_time)

            return self._inputs_ctgs_np(states, goals, actions, ctgs_backup, times)

    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        states, goals, actions, ctgs_backup = self._backup_edges(self.edges_popped, times)

        # to_np
        inputs_ctgs_np: List[NDArray] = self._inputs_ctgs_np(states, goals, actions, ctgs_backup, times)

        self.edges_popped = []
        return inputs_ctgs_np

    def _backup_edges(self, edges: List[EdgeQ], times: Times) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        start_time = time.time()
        edges_init, edges_real = _split_init_vs_real_edges(edges)
        times.record_time("split_edges", time.time() - start_time)

        # get backup of initial edge with random action
        # TODO this could be taking up a lot of GPU since includes more instances in parallel (i.e. both removed and current)
        start_time = time.time()
        states, goals, actions, ctgs_backup = self._backup_any_next_edge(edges_init)
        assert len(states) == len(goals) == len(actions) == len(ctgs_backup), \
            f"Values were {len(states)}, {len(goals)}, {len(actions)}, {len(ctgs_backup)}, "
        times.record_time("backup_init", time.time() - start_time)

        # get backup for real edges
        start_time = time.time()
        for edge_real in edges_real:
            node: Node = edge_real.node
            states.append(node.state)
            goals.append(node.goal)
            action: Optional[Action] = edge_real.action
            assert action is not None

            actions.append(action)
            ctgs_backup.append(node.backup_act(action))
        times.record_time("backup_real", time.time() - start_time)

        return states, goals, actions, ctgs_backup

    def _backup_any_next_edge(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        if len(edges) == 0:
            return [], [], [], []
        states, goals, is_solved_l, actions, tcs, states_next = self._edge_init_next_random(edges)
        ctgs_backup: List[float] = self._q_learning_backup_targ(goals, is_solved_l, tcs, states_next)

        return states, goals, actions, ctgs_backup

    def _get_edge_data(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:
        edges_init, edges_real = _split_init_vs_real_edges(edges)
        states, goals, is_solved_l, actions, tcs, states_next = self._edge_init_next_random(edges_init)
        for edge_real in edges_real:
            node: Node = edge_real.node
            states.append(node.state)
            goals.append(node.goal)
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)
            assert edge_real.action is not None
            actions.append(edge_real.action)
            tc, node_next = node.edge_dict[edge_real.action]
            tcs.append(tc)
            states_next.append(node_next.state)

        return states, goals, is_solved_l, actions, tcs, states_next

    def _edge_init_next_random(self, edges: List[EdgeQ]) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:
        if len(edges) == 0:
            return [], [], [], [], [], []

        node_l: List[Node] = [edge.node for edge in edges]
        states: List[State] = [node.state for node in node_l]
        goals: List[Goal] = [node.goal for node in node_l]
        is_solved_l: List[bool] = []
        for node in node_l:
            assert node.is_solved is not None
            is_solved_l.append(node.is_solved)

        actions: List[Action] = self.domain.get_state_action_rand(states)

        states_next, tcs = self.domain.next_state(states, actions)
        assert len(states) == len(goals) == len(is_solved_l) == len(actions) == len(tcs) == len(states_next), \
            f"Values were {len(states)}, {len(goals)}, {len(is_solved_l)}, {len(actions)}, {len(tcs)}, {len(states_next)}"

        return states, goals, is_solved_l, actions, tcs, states_next

    @abstractmethod
    def _get_qvals_targ(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        pass

    def _q_learning_backup_targ(self, goals: List[Goal], is_solved_l: List[bool], tcs: List[float], states_next: List[State]) -> List[float]:
        # min cost-to-go for next state
        qvals_next_l: List[List[float]] = self._get_qvals_targ(states_next, goals)
        qvals_next_min: List[float] = [min(qvals_next) for qvals_next in qvals_next_l]

        # backup cost-to-go
        ctg_backups: NDArray = np.array(tcs) + np.array(qvals_next_min)
        ctg_backups = ctg_backups * np.logical_not(np.array(is_solved_l))

        return cast(List[float], ctg_backups.tolist())

    def _inputs_ctgs_np(self, states: List[State], goals: List[Goal], actions: List[Action], ctgs_backup: List[float], times: Times) -> List[NDArray]:
        start_time = time.time()
        inputs_np: List[NDArray] = self.get_heur_nnet().to_np(states, goals, [[action] for action in actions])
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]

    def _get_root_nodes(self, pathfind: PQ, steps_gen: List[int], times: Times) -> List[Node]:
        # get states/goals
        times_states: Times = Times()
        states_gen, goals_gen = self.domain.get_start_goal_pairs(steps_gen, times=times_states)
        times.add_times(times_states, ["get_states"])

        # root nodes
        return pathfind.create_root_nodes(states_gen, goals_gen, compute_init_heur=False)


class UpdateHeurQEnum(UpdateHeurQ[ActsEnum, PQ], ABC):
    def _get_qvals_targ(self, states: List[State], goals: List[Goal]) -> List[List[float]]:
        actions_next: List[List[Action]] = self.domain.get_state_actions(states)
        qvals: List[List[float]] = self._get_targ_heur_fn()(states, goals, actions_next)

        return qvals
