from typing import List, Dict, Tuple, Any, Generic, TypeVar, Optional, cast, Type
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.process import BaseProcess
from multiprocessing.context import SpawnContext  # noqa

import numpy as np
import torch
from numpy.typing import NDArray

from deepxube.pytorch.nnet_utils import NNetParInfo
from deepxube.base.factory import DelimParser
from deepxube.base.domain import Domain, State, Action, Goal, GoalSampleableFromState
from deepxube.base.pathfind_fns import (PFNs, DeepXubeNNetPar, HeurVFn, HeurQFn, PolicyFn, HeurVNNetPar, HeurQNNetPar, PolicyNNetPar, UFNs,
                                        UFNsHeurV, UFNsHeurQ, UFNsPolicy)
from deepxube.base.pathfinding import (PFNsT, PFNsHV_T, PFNsHQ_T, PFNsP_T, PathFind, PathFindSup, Instance, InstanceNodeStatic, InstanceEdgeStatic, get_path,
                                       Node)
from deepxube.factories.pathfinding_factory import pathfinding_factory, get_pathfind_name_kwargs, get_pathfind_from_arg
from deepxube.pathfinding.utils.performance import PathFindPerf, print_pathfindperf
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd, get_nowait_noerr
from deepxube.utils.misc_utils import split_evenly, split_evenly_w_max
from deepxube.utils.timing_utils import Times

import gc
import copy
from torch.multiprocessing import get_context


# TODO par nnets per GPU?
@dataclass
class UpArgs:
    """ Each time an instance is solved, a new one is created with the same number of steps to maintain training data balance.

    :param procs: Number of parallel workers used to compute update
    :param step_max: Maximum number of steps to take when generating problem instances.
    :param search_itrs: Maximum number of pathfinding iterationos to take for each generated problem instances
    States and corresponding goals seen during search will be added to training instances.
    :param up_itrs: How many iterations worth of training instances to obtain for each update
    :param up_gen_itrs: How many iterations worth of data to generate per udpate. If None, set to up_itrs
    :param rb: amount of data generated from previous updates to keep in replay buffer. Total replay buffer size will
    then be batch_size * up_gen_itrs * rb.
    :param up_batch_size: Maximum number of searches to do at a time. Helps manage memory.
    Decrease if memory is running out during updater. None if as large as possible
    :param nnet_batch_size: Batch size of each nnet used for each process updater. Make smaller if running out
    of memory. None if as large as possible.
    :param sync_main: if True, number of processes can affect order in which data is seen
    :param v: True if update is verbose.
    """
    procs: int
    step_max: int
    search_itrs: int
    up_itrs: int
    up_gen_itrs: Optional[int]
    rb: int
    up_batch_size: Optional[int]
    nnet_batch_size: Optional[int]
    sync_main: bool
    v: bool

    def get_up_gen_itrs(self) -> int:
        return self.up_itrs if (self.up_gen_itrs is None) else self.up_gen_itrs


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


InstT = TypeVar('InstT', bound=Instance)
D = TypeVar('D', bound=Domain)
P = TypeVar('P', bound=PathFind)
UFNsT = TypeVar("UFNsT", bound=UFNs)
UFNsHV_T = TypeVar("UFNsHV_T", bound=UFNsHeurV)
UFNsHQ_T = TypeVar("UFNsHQ_T", bound=UFNsHeurQ)
UFNsP_T = TypeVar("UFNsP_T", bound=UFNsPolicy)


class Update(Generic[D, PFNsT, P, InstT, UFNsT], ABC):
    @staticmethod
    @abstractmethod
    def domain_type() -> Type[D]:
        pass

    @staticmethod
    @abstractmethod
    def pathfind_functions_type() -> Type[PFNsT]:
        pass

    @staticmethod
    @abstractmethod
    def pathfind_type() -> Type[P]:
        pass

    @staticmethod
    @abstractmethod
    def updater_functions_type() -> Type[UFNsT]:
        pass

    @classmethod
    def get_incompat_reason(cls, domain: Domain, pathfind_fns_t: Type[PFNs], pathfind_t: Type[PathFind], updater_fns_t: Type[UFNs]) -> Optional[str]:
        if not isinstance(domain, cls.domain_type()):
            return f"Domain {domain} is not an instance of {cls.domain_type()}"
        elif not issubclass(pathfind_fns_t, cls.pathfind_functions_type()):
            return f"PathFind functions type {pathfind_fns_t} is not a subclass of {cls.pathfind_functions_type()}"
        elif not issubclass(pathfind_t, cls.pathfind_type()):
            return f"PathFind type {pathfind_t} is not a subclass of {cls.pathfind_type()}"
        elif not issubclass(updater_fns_t, cls.updater_functions_type()):
            return f"Updater functions type {updater_fns_t} is not a subclass of {cls.updater_functions_type()}"

        return None

    @staticmethod
    def _update_perf(insts: List[InstT], step_to_pathperf: Dict[int, PathFindPerf]) -> None:
        for inst in insts:
            step_num_inst: int = int(inst.inst_info[0])
            if step_num_inst not in step_to_pathperf.keys():
                step_to_pathperf[step_num_inst] = PathFindPerf()
            step_to_pathperf[step_num_inst].update_perf(inst)

    def __init__(self, domain: D, pathfind_name_args: str, up_fns: UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1,
                 up_itrs: int = 100, up_gen_itrs: Optional[int] = None, rb: int = 0, up_batch_size: Optional[int] = None,
                 nnet_batch_size: Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: Any):
        self.domain: D = domain

        self.pathfind_name_args: str = pathfind_name_args
        pathfind_t: Type[PathFind] = pathfinding_factory.get_type(get_pathfind_name_kwargs(pathfind_name_args)[0])

        incompat_reason: Optional[str] = self.get_incompat_reason(domain, pathfind_t.pathfind_functions_type(), pathfind_t, type(up_fns))
        if incompat_reason is not None:
            raise TypeError(incompat_reason)

        self.up_fns: UFNsT = up_fns

        # kwargs
        self.up_args: UpArgs = UpArgs(procs=procs, step_max=step_max, search_itrs=search_itrs, up_itrs=up_itrs, up_gen_itrs=up_gen_itrs, rb=rb,
                                      up_batch_size=up_batch_size, nnet_batch_size=nnet_batch_size, sync_main=sync_main, v=v)

        # nnet objects
        # TODO redo domain nnet_pars
        """
        for nnet_name, (nnet_file, nnet_par) in domain.get_nnet_par_dict().items():
            self.add_nnet_par(nnet_name, nnet_par)
            self.set_nnet_file(nnet_name, nnet_file)
        self.nnet_fn_dict: Dict[str, NNetCallable] = dict()
        """

        # update objects
        self.targ_update_nums: Dict[str, int] = dict()
        self.procs: List[BaseProcess] = []
        self.to_q: Optional[Queue] = None
        self.from_q: Optional[Queue] = None
        self.num_generated: int = 0
        self.to_main_q: Optional[Queue] = None
        self.from_main_q: Optional[Queue] = None
        self.from_main_qs: List[Queue] = []
        self.q_id: int = 0
        self.nnet_par_info_main: Optional[NNetParInfo] = None

    @abstractmethod
    def get_train_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        pass

    @abstractmethod
    def get_train_nnet_par(self) -> DeepXubeNNetPar:
        pass

    def set_nnet_par_info_l(self) -> None:
        for up_fn in self.up_fns.get_up_fns():
            up_fn.set_nnet_par_info_l(self.up_args.procs)

    def set_nnet_par_info(self, nnet_name: str, nnet_par_info: NNetParInfo) -> None:
        self.up_fns.get_up_fn(nnet_name).set_nnet_par_info(nnet_par_info)

    def start_nnet_runners(self, device: torch.device, on_gpu: bool) -> None:
        for up_fn in self.up_fns.get_up_fns():
            up_fn.start_nnet_runners(device, on_gpu, self.up_args.nnet_batch_size)

    def init_nnet_fns(self) -> None:
        for up_fn in self.up_fns.get_up_fns():
            up_fn.init_nnet_par_fn()

        # self.domain.set_nnet_fns(self.nnet_fn_dict)  # TODO set domain fns

    def clear_nnet_fns(self) -> None:
        for up_fn in self.up_fns.get_up_fns():
            up_fn.clear_nnet_fn()

    def set_main_qs(self, to_main_q: Queue, from_main_q: Queue, q_id: int) -> None:
        self.to_main_q = to_main_q
        self.from_main_q = from_main_q
        self.q_id = q_id
        assert (self.to_main_q is not None) and (self.from_main_q is not None)
        self.nnet_par_info_main = NNetParInfo(self.to_main_q, self.from_main_q, self.q_id)

    def start_procs(self, rb_size: int) -> Tuple[Queue, List[Queue]]:
        # start updater procs
        # TODO implement safer copy?
        updaters: List[Update] = [copy.deepcopy(self) for _ in range(self.up_args.procs)]

        # parallel heuristic functions
        ctx = get_context("spawn")
        to_main_q: Queue = ctx.Queue()
        self.from_main_qs = []
        self.set_nnet_par_info_l()
        for proc_idx, updater in enumerate(updaters):
            from_main_q: Queue = ctx.Queue(1)
            self.from_main_qs.append(from_main_q)
            updater.set_main_qs(to_main_q, from_main_q, proc_idx)
            for field_name in self.up_fns.get_field_names():
                updater.set_nnet_par_info(field_name, self.up_fns.get_up_fn(field_name).get_nnet_par_infos(proc_idx))

        # get rb sizes
        rb_sizes_q: List[int] = [0] * len(updaters)
        if rb_size > 0:
            rb_sizes_q = split_evenly(rb_size, len(updaters))
            assert min(rb_sizes_q) > 0, "Number of processes must not exceed that of the size of the replay buffer"

        # start procs
        self.to_q = ctx.Queue()
        self.from_q = ctx.Queue()
        self.procs = []
        for updater, rb_size in zip(updaters, rb_sizes_q):
            proc: BaseProcess = ctx.Process(target=updater.update_runner, args=(self.to_q, self.from_q, rb_size))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

        return to_main_q, self.from_main_qs

    def start_update(self, step_probs: List[int], num_gen: int, train_batch_size: int,
                     device: torch.device, on_gpu: bool) -> None:
        # start parallel nnet runners
        self.start_nnet_runners(device, on_gpu)

        # put update data
        for proc_idx, from_main_q in enumerate(self.from_main_qs):
            from_main_q.put((step_probs, self.targ_update_nums.copy()))

        # put work information on to_q
        assert self.to_q is not None

        num_searches: int = num_gen // self.up_args.search_itrs
        if self.up_args.v:
            print(f"Generating {format(num_gen, ',')} training instances with at least {format(num_searches, ',')} searches")

        assert num_gen % self.up_args.search_itrs == 0, (f"Number of instances to generate per for this updater {num_gen} is not divisible by the max number "
                                                         f"of pathfinding iterations to take during the updater ({self.up_args.search_itrs})")

        up_batch_size: int = train_batch_size if (self.up_args.up_batch_size is None) else self.up_args.up_batch_size
        num_to_send_per: List[int] = split_evenly_w_max(num_searches, self.up_args.procs, min(up_batch_size, train_batch_size))
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
        print(f"Times - {times_up.get_time_str()}")
        if self.up_args.v:
            print(f"Generated {format(self.num_generated, ',')} training instances")
            print_pathfindperf(step_to_pathperf)

        # clean up clean up everybody do your share
        for up_fn in self.up_fns.get_up_fns():
            up_fn.stop_nnet_runners()

        self.num_generated = 0

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

    def get_pathfind(self) -> P:
        return cast(P, get_pathfind_from_arg(self.domain, self._get_pathfind_functions(), self.pathfind_name_args)[0])

    def set_targ_update_num(self, nnet_name: str, targ_update_num: int) -> None:
        self.targ_update_nums[nnet_name] = targ_update_num

    def update_runner(self, to_q: Queue, from_q: Queue, rb_size: int) -> None:
        if self.up_args.sync_main:
            assert rb_size > 0, "must use a replay buffer if doing sync_main"
        self._init_replay_buffer(rb_size)

        while True:
            assert self.from_main_q is not None
            data_q: Optional[Tuple[List[int], Dict[str, int]]] = self.from_main_q.get()
            if data_q is None:
                break
            times: Times = Times()

            step_probs, targ_update_nums = data_q
            self.targ_update_nums = targ_update_nums.copy()
            self.init_nnet_fns()

            step_to_pathperf: Dict[int, PathFindPerf] = dict()
            while True:
                batch_size = to_q.get()
                if batch_size is None:
                    break

                pathfind: P = self.get_pathfind()
                # self._set_pathfind_nnet_fns(pathfind)

                insts_rem_last_itr: List[InstT] = []
                put_from_q: List[List[NDArray]] = []
                for _ in range(self.up_args.search_itrs):
                    # add instances
                    self._add_instances(pathfind, insts_rem_last_itr, batch_size, step_probs, times)
                    assert len(pathfind.instances) == batch_size, f"Values were {len(pathfind.instances)} and {batch_size}"

                    # step
                    if self.up_args.sync_main:
                        data: List[NDArray] = self._step_sync_main(pathfind, times)
                        _put_from_q([data], from_q, times)
                    else:
                        self._step(pathfind, times)

                    # remove instances
                    insts_rem_last_itr = pathfind.remove_finished_instances(self.up_args.search_itrs)
                    if len(insts_rem_last_itr) > 0:
                        put_from_q.append(self._get_instance_data(insts_rem_last_itr, rb_size, times))

                    # performance
                    start_time = time.time()
                    self._update_perf(insts_rem_last_itr, step_to_pathperf)
                    times.record_time("update_perf", time.time() - start_time)

                if not self.up_args.sync_main:
                    if len(pathfind.instances) > 0:
                        put_from_q.append(self._get_instance_data(pathfind.instances, rb_size, times))
                    _put_from_q(put_from_q, from_q, times)

                times.add_times(pathfind.times, path=["pathfinding"])

                # garbage collection
                start_time = time.time()
                del insts_rem_last_itr
                del put_from_q
                del pathfind
                gc.collect()
                times.record_time("gc", time.time() - start_time)

            from_q.put((times, step_to_pathperf))
            self.clear_nnet_fns()
        self.to_main_q = None
        self.from_main_q = None
        self.nnet_par_info_main = None

    def _add_instances(self, pathfind: P, insts_rem: List[InstT], batch_size: int, step_probs: List[int],
                       times: Times) -> None:
        if (len(pathfind.instances) == 0) or (len(insts_rem) > 0):
            # get steps generate
            start_time = time.time()
            steps_gen: List[int]
            if len(insts_rem) > 0:
                steps_gen = [int(inst.inst_info[0]) for inst in insts_rem]
            else:
                steps_gen = np.random.choice(self.up_args.step_max + 1, size=batch_size, p=np.array(step_probs)).tolist()
            times.record_time("steps_gen", time.time() - start_time)

            # get instance information and kwargs
            start_time = time.time()
            inst_infos: List[Tuple[int]] = [(step_gen,) for step_gen in steps_gen]
            times.record_time("inst_info", time.time() - start_time)

            instances: List[InstT] = self._make_instances(pathfind, steps_gen, inst_infos, times)

            # add instances
            start_time = time.time()
            pathfind.add_instances(instances)
            times.record_time("inst_add", time.time() - start_time)

    @abstractmethod
    def _step(self, pathfind: P, times: Times) -> None:
        pass

    @abstractmethod
    def _step_sync_main(self, pathfind: P, times: Times) -> List[NDArray]:
        pass

    @abstractmethod
    def _get_pathfind_functions(self) -> PFNsT:
        pass

    def _get_instance_data(self, instances: List[InstT], rb_size: int, times: Times) -> List[NDArray]:
        if rb_size == 0:
            return self._get_instance_data_norb(instances, times)
        else:
            return self._get_instance_data_rb(instances, times)

    @abstractmethod
    def _get_instance_data_norb(self, instances: List[InstT], times: Times) -> List[NDArray]:
        pass

    @abstractmethod
    def _get_instance_data_rb(self, instances: List[InstT], times: Times) -> List[NDArray]:
        pass

    @abstractmethod
    def _make_instances(self, pathfind: P, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[InstT]:
        pass

    @abstractmethod
    def _init_replay_buffer(self, max_size: int) -> None:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}, {self.up_args.__repr__()}"


class UpdateHER(Update[GoalSampleableFromState, PFNsT, P, InstT, UFNsT], ABC):
    def _step_sync_main(self, pathfind: P, times: Times) -> List[NDArray]:
        raise NotImplementedError("Cannot train with sync_main if also doing hindsight experience replay (HER) since goal relabeling is done after search is "
                                  "complete.")

    def _get_instance_data_norb(self, instances: List[InstT], times: Times) -> List[NDArray]:
        raise NotImplementedError("Must use replay buffer if doing HER.")

    def _get_her_goals(self, instances: List[InstT], times: Times) -> Tuple[List[InstT], List[Goal]]:
        """ If instance is not finisheed and solved, get deepest states out all nodes that have children + root node for relabeled goal.
            :return: Instances and their corresponding goals (order of instances changes)
        """
        # get states/goals or mark for goal relabeling
        instances_goalkeep: List[InstT] = []
        instances_relabel: List[InstT] = []

        rand_keeps: List[float] = cast(List[float], np.random.uniform(0, 1, size=len(instances)).tolist())
        for instance, rand_keep in zip(instances, rand_keeps):
            if instance.finished() and instance.has_soln():
                instances_goalkeep.append(instance)
            else:
                instances_relabel.append(instance)

        # get goals goalkeep
        goals_goalkeep: List[Goal] = [instance.root_node.goal for instance in instances_goalkeep]

        # get relabeled goals
        goals_relabel: List[Goal] = []
        if len(instances_relabel) > 0:
            # get start states and deepest states
            start_time = time.time()
            states_start: List[State] = []
            states_deepest: List[State] = []
            for instance in instances_relabel:
                states_start.append(instance.root_node.state)

                # get all descendants that have children
                nodes_desc: List[Node] = instance.root_node.get_all_descendants()
                node_desc_w_children: List[Node] = [node_desc for node_desc in nodes_desc if len(node_desc.edge_dict) > 0]

                # get state of deepest node
                state_deepest: State = instance.root_node.state
                deepest_depth: int = 0
                for node in node_desc_w_children:
                    depth: int = len(get_path(node)[0])
                    if depth > deepest_depth:
                        deepest_depth = depth
                        state_deepest = node.state
                states_deepest.append(state_deepest)

            times.record_time("node_deepest", time.time() - start_time, path=["HER"])

            # relabel
            start_time = time.time()
            goals_relabel = self.domain.sample_goal_from_state(states_start, states_deepest)
            times.record_time("relabel", time.time() - start_time, path=["HER"])

        return instances_goalkeep + instances_relabel, goals_goalkeep + goals_relabel


class UpdateHasHeurV(Update[D, PFNsT, P, InstT, UFNsHV_T], ABC):
    def get_heurv_nnet_par(self) -> HeurVNNetPar:
        return self.up_fns.heurv

    def get_heurv_fn(self) -> HeurVFn:
        return self._get_targ_heurv_fn()

    def _get_targ_heurv_fn(self) -> HeurVFn:
        heurv_nnet_par: HeurVNNetPar = self.get_heurv_nnet_par()
        update_num: int = self.targ_update_nums[heurv_nnet_par.get_field_name()]
        if update_num == 0:
            return heurv_nnet_par.get_default_fn()
        else:
            return heurv_nnet_par.get_nnet_par_fn()


class UpdateHasHeurQ(Update[D, PFNsT, P, InstT, UFNsHQ_T], ABC):
    def get_heurq_nnet_par(self) -> HeurQNNetPar:
        return self.up_fns.heurq

    def get_heurq_fn(self) -> HeurQFn:
        return self._get_targ_heurq_fn()

    def _get_targ_heurq_fn(self) -> HeurQFn:
        heurq_nnet_par: HeurQNNetPar = self.get_heurq_nnet_par()
        update_num: int = self.targ_update_nums[heurq_nnet_par.get_field_name()]
        if update_num == 0:
            return heurq_nnet_par.get_default_fn()
        else:
            return heurq_nnet_par.get_nnet_par_fn()


class UpdateHasPolicy(Update[D, PFNsP_T, P, InstT, UFNsP_T], ABC):
    def get_policy_nnet_par(self) -> PolicyNNetPar:
        return self.up_fns.policy

    def get_policy_fn(self) -> PolicyFn:
        return self._get_targ_policy_fn()

    def _get_targ_policy_fn(self) -> PolicyFn:
        policy_nnet_par: PolicyNNetPar = self.get_policy_nnet_par()
        update_num: int = self.targ_update_nums[policy_nnet_par.get_field_name()]

        if update_num == 0:
            return policy_nnet_par.get_default_fn()
        else:
            return policy_nnet_par.get_nnet_par_fn()


PS = TypeVar('PS', bound=PathFindSup)


class UpdateSup(Update[D, PFNs, PS, InstT, UFNsT], ABC):
    @staticmethod
    def pathfind_functions_type() -> Type[PFNs]:
        return PFNs

    def _step(self, pathfind: PS, times: Times) -> None:
        pathfind.step()

    def _get_pathfind_functions(self) -> PFNs:
        return PFNs()

    def _make_instances(self, pathfind: PS, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[InstT]:
        return pathfind.make_instances_sup(steps_gen, inst_infos)

    def _step_sync_main(self, pathfind: PS, times: Times) -> List[NDArray]:
        raise NotImplementedError("No sync_main option for supervised update")

    def _get_instance_data_rb(self, instances: List[InstT], times: Times) -> List[NDArray]:
        raise NotImplementedError("No replay buffer used with supervised update")

    def _init_replay_buffer(self, max_size: int) -> None:
        pass


@dataclass
class UpRLArgs:
    ub_heur_solns: bool = False
    lhbl: bool = False


class UpdateRL(Update[D, PFNsT, P, InstT, UFNsT], ABC):
    def __init__(self, *args: Any, ub_heur_solns: bool = False, lhbl: bool = False, **kwargs: Any):
        self.up_rl_args: UpRLArgs = UpRLArgs(ub_heur_solns=ub_heur_solns, lhbl=lhbl)
        super().__init__(*args, **kwargs)

    def _make_instances(self, pathfind: P, steps_gen: List[int], inst_infos: List[Any], times: Times) -> List[InstT]:
        # get states/goals
        times_states: Times = Times()
        states_gen, goals_gen = self.domain.sample_problem_instances(steps_gen, times=times_states)
        times.add_times(times_states, ["get_states"])

        return pathfind.make_instances(states_gen, goals_gen, inst_infos=inst_infos, compute_root_vals=False)

    def __repr__(self) -> str:
        return f"{super().__repr__()}, {self.up_rl_args.__repr__()}"


class UpdateHeur(Update[D, PFNsT, P, InstT, UFNsT], ABC):
    pass


class UpdateHeurV(UpdateHeur[D, PFNsHV_T, P, InstanceNodeStatic, UFNsHV_T], UpdateHasHeurV[D, PFNsHV_T, P, InstanceNodeStatic, UFNsHV_T], ABC):
    def get_train_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.sample_problem_instances([0])
        inputs_nnet: List[NDArray[Any]] = self.get_heurv_nnet_par().process_inputs(states, goals).inputs_nnet

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dtypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dtypes

    def get_train_nnet_par(self) -> DeepXubeNNetPar:
        return self.get_heurv_nnet_par()

    def get_heurv_fn(self) -> HeurVFn:
        if not self.up_args.sync_main:
            return super().get_heurv_fn()
        else:
            assert self.nnet_par_info_main is not None
            return self.get_heurv_nnet_par().get_nnet_par_fn_w_info(self.nnet_par_info_main)


class UpdateHeurQ(UpdateHeur[D, PFNsHQ_T, P, InstanceEdgeStatic, UFNsHQ_T], UpdateHasHeurQ[D, PFNsHQ_T, P, InstanceEdgeStatic, UFNsHQ_T], ABC):
    def get_train_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.sample_problem_instances([0])
        actions: List[Action] = self.domain.sample_state_action(states)
        inputs_nnet: List[NDArray[Any]] = self.get_heurq_nnet_par().process_inputs(states, goals, [[action] for action in actions]).inputs_nnet

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))
        shapes_dtypes.append((tuple(), np.dtype(np.float64)))

        return shapes_dtypes

    def get_train_nnet_par(self) -> DeepXubeNNetPar:
        return self.get_heurq_nnet_par()

    def get_heurq_fn(self) -> HeurQFn:
        if not self.up_args.sync_main:
            return super().get_heurq_fn()
        else:
            assert self.nnet_par_info_main is not None
            return self.get_heurq_nnet_par().get_nnet_par_fn_w_info(self.nnet_par_info_main)


class UpdatePolicy(UpdateHasPolicy[D, PFNsP_T, P, InstT, UFNsP_T], ABC):
    def get_train_nnet_par(self) -> DeepXubeNNetPar:
        return self.get_policy_nnet_par()

    def get_train_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.domain.sample_problem_instances([0])
        actions: List[Action] = self.domain.sample_state_action(states)
        inputs_nnet: List[NDArray[Any]] = self.get_policy_nnet_par().to_np_train(states, goals, actions)

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dtypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dtypes

    def get_policy_fn(self) -> PolicyFn:
        if not self.up_args.sync_main:
            return super().get_policy_fn()
        else:
            raise NotImplementedError("sync_main not yet implemented for policy_fn")


class UpdateParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("p", "procs", int, "Number of parallel workers used to compute update", default=1)
        self.add_argument("sm", "step_max", int, "Maximum num of steps to generate problem instances", default=100)
        self.add_argument("sitrs", "search_itrs", int, "Number of search iterations", default=1)

        self.add_argument("up", "up_itrs", int, "How many iterations worth of training instances to obtain for each update.", default=100)
        self.add_argument("upg", "up_gen_itrs", int, "How many iterations worth of data to generate per udpate. If not given then it is set to "
                                                     "up_itrs.")
        self.add_argument("rb", "rb", int, "Amount of data generated from previous updates to keep in replay buffer. If 0, then no replay buffer is used. "
                                           "Total replay buffer size will then be batch_size * up_gen_itrs * rb.", default=0)

        self.add_argument("upbs", "up_batch_size", int, "Maximum number of searches to do at a time. Helps manage memory. Decrease if memory is "
                                                        "running out during updater. If not set, then it is as large as possible", default=100)
        self.add_argument("nnbs", "nnet_batch_size", int, "Batch size of each nnet used for each process updater. Make smaller if running out "
                                                          "of memory. If not set, then it is as large as possible.", default=20000)
        self.add_argument("sync", "sync_main", None, "Synchronize functions used to search with training process. "
                                                     "If True, number of processes can affect order in which data is seen")
        self.add_argument("v", "v", None, "Verbose")

    @property
    def delim(self) -> str:
        return "_"


class UpdateRLParser(UpdateParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("ubsoln", "ub_heur_solns", None, "If True, the target cost-to-go will be min(backup, path_cost_from_state)")
        self.add_argument("lhbl", "lhbl", None, "If True, compute targets by backing up search tree with Limited Horizon Bellman-based Learning")
