from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, List, Dict, Tuple, cast, Optional, Union, Type

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import Optimizer

from deepxube.base.factory import DelimParser
from deepxube.base.heuristic import DeepXubeNNet
from deepxube.base.pathfind_fns import DeepXubeNNetPar
from deepxube.base.updater import Update
from deepxube.pathfinding.utils.performance import PathFindPerf, get_eq_weighted_perf
from deepxube.utils.data_utils import sel_l, SharedNDArray
from deepxube.nnet.nnet_utils import nnet_in_out_shared_q
from deepxube.utils.data_utils import get_nowait_noerr
from deepxube.nnet import nnet_utils
from deepxube.utils.timing_utils import Times

from multiprocessing import Queue
import numpy as np
from numpy.typing import NDArray

import pickle
import os
import shutil
import time
import re


@dataclass
class TrainArgs:
    """
    :param batch_size: Batch size
    :param max_itrs: Maximum number of iterations
    :param balance_steps: If true, steps are balanced based on solve percentage
    :param loss_thresh: Loss threshold for updating.
    :param checkpoint: Save checkpoint file of network being trained at initialization and at every given number of update checks.
    Checkpoint number given is training iteration, not update number. If 0 then checkpointing is not done.
    :param grad_accum: Number of times to split batch into sub-batches for gradient accumulation
    :param display: Number of iterations to display progress when training nnet. No display if 0.
    """
    batch_size: int
    max_itrs: int
    balance_steps: bool
    loss_thresh: float
    checkpoint: int
    grad_accum: int
    display: int


class DataBuffer:
    def __init__(self, max_size: int, shapes: List[Tuple[int, ...]], dtypes: List[np.dtype]):
        self.arrays: List[NDArray] = []
        self.max_size: int = max_size
        self.curr_size: int = 0
        self.add_idx: int = 0

        # first add
        start_time = time.time()
        print(f"Initializing data buffer with max size {format(self.max_size, ',')}")
        print("Input array sizes:")
        for array_idx, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            print(f"index: {array_idx}, dtype: {dtype}, shape:", shape)
            array: NDArray = np.empty((self.max_size,) + shape, dtype=dtype)
            self.arrays.append(array)

        print(f"Data buffer initialized. Time: {time.time() - start_time}")

    def add(self, arrays_add: List[NDArray]) -> None:
        self.curr_size = min(self.curr_size + arrays_add[0].shape[0], self.max_size)
        assert len(self.arrays) > 0, "Data buffer should have at least one array."
        self._add_circular(arrays_add)

    def sample(self, sel_idxs: NDArray) -> List[NDArray]:
        arrays_samp: List[NDArray] = sel_l(self.arrays, sel_idxs)

        return arrays_samp

    def size(self) -> int:
        return self.curr_size

    def clear(self) -> None:
        self.curr_size = 0
        self.add_idx = 0

    def _add_circular(self, arrays_add: List[NDArray]) -> None:
        start_idx: int = 0
        num_add: int = arrays_add[0].shape[0]
        assert len(self.arrays) == len(arrays_add), "should have same number of arrays"
        while start_idx < num_add:
            num_add_i: int = min(num_add - start_idx, self.max_size - self.add_idx)
            end_idx: int = start_idx + num_add_i
            add_idx_end: int = self.add_idx + num_add_i

            for input_idx in range(len(self.arrays)):
                self.arrays[input_idx][self.add_idx:add_idx_end] = arrays_add[input_idx][start_idx:end_idx]

            start_idx = end_idx
            self.add_idx = add_idx_end
            if self.add_idx == self.max_size:
                self.add_idx = 0


class Status:
    def __init__(self, step_max: int, balance_steps: bool):
        self.itr: int = 0
        self.update_num: int = 0
        self.targ_update_num: int = 0
        self.step_max: int = step_max
        self.step_probs: NDArray
        self.step_max_curr: int = 1
        if balance_steps:
            self.step_probs = np.zeros(self.step_max + 1)
            self.step_probs[0:2] = 0.5
            self.step_probs[2:] = 0
        else:
            self.step_probs = np.ones(self.step_max + 1)/(self.step_max + 1)
        self.per_solved_best: float = 0.0

    def update_step_probs(self, step_to_search_perf: Dict[int, PathFindPerf]) -> None:
        ave_solve: float = float(np.mean([step_to_search_perf[step].per_solved()
                                          for step in step_to_search_perf.keys()]))
        if ave_solve >= 50.0:
            self.step_max_curr = min(self.step_max_curr * 2, self.step_max)

        self.step_probs = np.zeros(self.step_max + 1)
        self.step_probs[np.arange(0, self.step_max_curr + 1)] = 1 / (self.step_max_curr + 1)


class TrainSummary:
    def __init__(self) -> None:
        self.itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]] = dict()
        self.itr_to_steps_to_pathfindstats: Dict[int, Dict[int, Dict]] = dict()

    def update_pathfindstats(self, step_to_pathfindperf: Dict[int, PathFindPerf], itr: int) -> None:
        self.itr_to_steps_to_pathfindstats[itr] = dict()
        for step, pathfindperf in step_to_pathfindperf.items():
            self.itr_to_steps_to_pathfindstats[itr][step] = dict()
            self.itr_to_steps_to_pathfindstats[itr][step]["per_solved"] = pathfindperf.per_solved()
            self.itr_to_steps_to_pathfindstats[itr][step]["path_costs"] = pathfindperf.stats()[1]
            self.itr_to_steps_to_pathfindstats[itr][step]["search_itrs"] = pathfindperf.stats()[2]
            self.itr_to_steps_to_pathfindstats[itr][step]["ctgs_backup"] = float(np.mean(pathfindperf.ctgs_bkup))
            self.itr_to_steps_to_pathfindstats[itr][step]["num_instances"] = len(pathfindperf.ctgs_bkup)


NNet = TypeVar('NNet', bound=DeepXubeNNet)
Up = TypeVar('Up', bound=Update)


def update_optimizer(optimizer: Optimizer, nnet: Union[DataParallel, DeepXubeNNet], train_itr: int) -> None:
    if isinstance(nnet, DataParallel):
        nnet = nnet.module
    assert isinstance(nnet, DeepXubeNNet)
    nnet.update_optimizer(optimizer, train_itr)


class Train(Generic[NNet, Up], ABC):
    @staticmethod
    @abstractmethod
    def data_parallel() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def nnet_type() -> Type[NNet]:
        pass

    @staticmethod
    @abstractmethod
    def updater_type() -> Type[Up]:
        pass

    @classmethod
    def is_compat(cls, nnet: DeepXubeNNet, updater: Update) -> bool:
        if not isinstance(nnet, cls.nnet_type()):
            return False
        if not isinstance(updater, cls.updater_type()):
            return False

        return True

    @staticmethod
    @abstractmethod
    def get_nnet_name() -> str:
        pass

    def __init__(self, nnet_dir: str, updater: Up, device: torch.device, on_gpu: bool, writer: SummaryWriter, batch_size: int = 100, max_itrs: int = 100000,
                 balance_steps: bool = False, loss_thresh: float = np.inf, checkpoint: int = 0, grad_accum: int = 1, display: int = 100) -> None:
        self.nnet_dir: str = nnet_dir
        if not os.path.exists(self.nnet_dir):
            os.makedirs(self.nnet_dir)

        self.nnet_name: str = self.get_nnet_name()
        self.nnet_file: str = f"{self.nnet_dir}/{self.nnet_name}.pt"
        self.nnet_targ_file: str = f"{self.nnet_dir}/{self.nnet_name}_targ.pt"

        self.updater: Up = updater
        nnet_par: DeepXubeNNetPar = self.updater.get_train_nnet_par()
        self.nnet_field_name: str = nnet_par.get_field_name()
        nnet_par.set_nnet_file(self.nnet_targ_file)
        self.nnet: NNet = nnet_par.get_nnet()
        assert self.is_compat(self.nnet, self.updater)

        self.writer: SummaryWriter = writer

        # kwargs
        self.train_args: TrainArgs = TrainArgs(batch_size=batch_size, max_itrs=max_itrs, balance_steps=balance_steps, loss_thresh=loss_thresh,
                                               checkpoint=checkpoint, grad_accum=grad_accum, display=display)
        self.device: torch.device = device
        self.on_gpu: bool = on_gpu

        # load status
        self.status_file: str = f"{self.nnet_dir}/{self.nnet_name}_status.pkl"
        self.status: Status
        if os.path.isfile(self.status_file):
            self.status = pickle.load(open(self.status_file, "rb"))
            print(f"Loaded with itr: {self.status.itr}, update_num: {self.status.update_num}, targ_update_num: {self.status.targ_update_num}")
        else:
            self.status = Status(self.updater.up_args.step_max, self.train_args.balance_steps)
            # noinspection PyTypeChecker
            pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)

        self.train_summary_file: str = f"{self.nnet_dir}/{self.nnet_name}_train_summary.pkl"
        self.train_summary: TrainSummary
        if os.path.isfile(self.train_summary_file):
            self.train_summary = pickle.load(open(self.train_summary_file, "rb"))
        else:
            self.train_summary = TrainSummary()
            # noinspection PyTypeChecker
            pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)

        # load nnet
        if os.path.isfile(self.nnet_file):
            self.nnet = cast(NNet, nnet_utils.load_nnet(self.nnet_file, self.nnet))
        else:
            torch.save(self.nnet.state_dict(), self.nnet_file)
            if (self.train_args.checkpoint > 0) and (self.status.update_num == 0):
                self._save_checkpoint()

        if not os.path.isfile(self.nnet_targ_file):
            torch.save(self.nnet.state_dict(), self.nnet_targ_file)
        self.optimizer: Optimizer = self.nnet.get_optimizer()

        self.nnet.to(self.device)
        if self.data_parallel():
            self.nnet = cast(NNet, nn.DataParallel(self.nnet))

        # init data buffer
        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = self.updater.get_train_shapes_dtypes()
        db_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes]
        db_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes]
        self.db: DataBuffer = DataBuffer(self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs(), db_shapes, db_dtypes)

        # optimizer and criterion
        self.train_start_time = time.time()

    def train_loop(self) -> None:
        # start procs
        to_main_q, from_main_qs = self.updater.start_procs(self.updater.up_args.rb * self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs())

        # train loop
        while self.status.itr < self.train_args.max_itrs:
            # update
            self.updater.set_targ_update_num(self.nnet_field_name, self.status.targ_update_num)

            self._update_step(to_main_q, from_main_qs)
            torch.cuda.empty_cache()

        # stop procs
        self.updater.stop_procs()
        print("Done")

    def _update_step(self, to_main_q: Queue, from_main_qs: List[Queue]) -> None:
        self.db.clear()
        itr_init: int = self.status.itr

        # print info
        start_info_l: List[str] = [f"itr: {self.status.itr}", f"update_num: {self.status.update_num}", f"targ_update: {self.status.targ_update_num}"]

        num_gen: int = self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs()
        start_info_l.append(f"num_gen: {format(num_gen, ',')}")
        if self.train_args.balance_steps:
            start_info_l.append(f"step max (curr): {self.status.step_max_curr}")
        print(f"\nGetting Data - {', '.join(start_info_l)}")
        times: Times = Times()

        # start updater
        start_time = time.time()
        self.updater.start_update(self.status.step_probs.tolist(), num_gen, self.train_args.batch_size, self.device, self.on_gpu)
        times.record_time("up_start", time.time() - start_time)

        # do training
        self.train_start_time = time.time()
        loss: float
        if not self.updater.up_args.sync_main:
            self._get_update_data(num_gen, times)
            self._end_update(itr_init, times)
            loss = self._train(times)
        else:
            loss = self._train_sync_main(num_gen, times, to_main_q, from_main_qs)
            self._end_update(itr_init, times)

        # save nnet
        start_time = time.time()
        torch.save(self.nnet.state_dict(), self.nnet_file)
        if (self.train_args.checkpoint > 0) and (self.status.update_num % self.train_args.checkpoint == 0):
            self._save_checkpoint()
        times.record_time("save_net", time.time() - start_time)

        # update nnet
        update_targ: bool = False
        if loss < self.train_args.loss_thresh:
            update_targ = True

        if update_targ:
            shutil.copy(self.nnet_file, self.nnet_targ_file)
            self.status.targ_update_num = self.status.targ_update_num + 1
        self.status.update_num += 1

        start_time = time.time()
        # noinspection PyTypeChecker
        pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)
        # noinspection PyTypeChecker
        pickle.dump(self.train_summary, open(self.train_summary_file, "wb"), protocol=-1)
        times.record_time("save_status", time.time() - start_time)
        print(f"Train - itrs: {self.updater.up_args.up_itrs}, loss: {loss:.2E}, targ_updated: {update_targ}")
        print(f"Times - {times.get_time_str()}")

    def _get_update_data(self, num_gen: int, times: Times) -> None:
        start_time = time.time()
        while self.db.size() < num_gen:
            data_l: List[List[NDArray]] = self.updater.get_update_data()
            for data in data_l:
                self.db.add(data)
        times.record_time("up_data", time.time() - start_time)

    def _train(self, times: Times) -> float:
        loss: float = np.inf
        first_itr_in_update: bool = True
        sel_idx_start: int = 0
        sel_idxs_rand_order: NDArray = np.random.choice(self.db.size(), size=self.db.size(), replace=False)
        for _ in range(self.updater.up_args.up_itrs):
            # sample data
            start_time = time.time()
            sel_idxs: NDArray = np.arange(sel_idx_start, sel_idx_start + self.train_args.batch_size) % self.db.size()

            batch: List[NDArray] = self.db.sample(sel_idxs_rand_order[sel_idxs])
            times.record_time("data_samp", time.time() - start_time)

            # train
            loss = self._train_itr(batch, first_itr_in_update, times)
            first_itr_in_update = False
            self.status.itr += 1

            # update sel_idx
            if sel_idxs.max() == (self.db.size() - 1):
                sel_idx_start = 0
                sel_idxs_rand_order = np.random.choice(self.db.size(), size=self.db.size(), replace=False)
            else:
                sel_idx_start = int(sel_idxs[-1]) + 1

        return loss

    def _train_sync_main(self, num_gen: int, times: Times, to_main_q: Queue, from_main_qs: List[Queue]) -> float:
        loss: float = np.inf
        update_train_itr: int = 0
        first_itr_in_update: bool = True
        while update_train_itr < self.updater.up_args.up_itrs:
            # data from updater should not be more that train_args.batch_size
            start_time = time.time()
            sel_idxs: NDArray
            if self.db.size() == num_gen:
                sel_idxs = np.random.randint(self.db.size(), size=self.train_args.batch_size)
            else:
                # compute heuristic values for ongoing search and get data
                self.nnet.eval()
                while self.db.size() < ((update_train_itr + 1) * self.train_args.batch_size):
                    # compute heuristic values
                    q_res: Optional[Tuple[int, List[SharedNDArray]]] = get_nowait_noerr(to_main_q)
                    if q_res is not None:
                        proc_id, inputs_np_shm = q_res
                        nnet_in_out_shared_q(self.nnet, inputs_np_shm, self.updater.up_args.nnet_batch_size,
                                             self.device, from_main_qs[proc_id])

                    # get update data
                    data_l_i: List[List[NDArray]] = self.updater.get_update_data(nowait=True)
                    for data in data_l_i:
                        self.db.add(data)

                # select incides
                sel_idxs = np.arange(update_train_itr * self.train_args.batch_size, (update_train_itr + 1) * self.train_args.batch_size)

            batch: List[NDArray] = self.db.sample(sel_idxs)

            times.record_time("up_data", time.time() - start_time)

            # train
            loss = self._train_itr(batch, first_itr_in_update, times)
            update_train_itr += 1
            first_itr_in_update = False
            self.status.itr += 1

        return loss

    @abstractmethod
    def _train_itr(self, batch: List[NDArray], first_itr_in_update: bool, times: Times) -> float:
        pass

    def _end_update(self, itr_init: int, times: Times) -> None:
        start_time = time.time()
        step_to_search_perf: Dict[int, PathFindPerf] = self.updater.end_update()
        self.train_summary.update_pathfindstats(step_to_search_perf, itr_init)
        if self.train_args.balance_steps:
            self.status.update_step_probs(step_to_search_perf)

        per_solved_ave, path_costs_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)

        self.writer.add_scalar("train/pathfind/solved", per_solved_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/path_cost", path_costs_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/search_itrs", search_itrs_ave, self.status.itr)

        post_up_info_l: List[str] = [f"%solved: {per_solved_ave:.2f}", f"path_costs: {path_costs_ave:.3f}",
                                     f"search_itrs: {search_itrs_ave:.3f}"] + self._add_post_up_info()

        print(f"Data - {', '.join(post_up_info_l)}")

        times.record_time("up_end", time.time() - start_time)

    def _save_checkpoint(self) -> None:
        assert re.compile(".*.pt$").match(self.nnet_file) is not None, "nnet file should end in '.pt'"
        nnet_file_chkpt: str = re.sub(".pt$", f"_chkpt_{self.status.itr}.pt", self.nnet_file)
        torch.save(self.nnet.state_dict(), nnet_file_chkpt)

    @abstractmethod
    def _add_post_up_info(self) -> List[str]:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}, {self.train_args.__repr__()}"


class TrainParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("bs", "batch_size", int, "Batch size.", default=100)
        self.add_argument("maxit", "max_itrs", int, "Maximum number of iterations.", default=100000)
        self.add_argument("bal", "balance_steps", None, "If true, steps are balanced based on solve percentage.")
        self.add_argument("lt", "loss_thresh", float, "Loss threshold for updating.", default=np.inf)
        self.add_argument("chkpt", "checkpoint", int, "Save checkpoint file of network being trained at initialization and at every given number of update "
                                                      "checks. Checkpoint number given is training iteration, not update number. If 0 then checkpointing is "
                                                      "not done.", default=0)
        self.add_argument("accum", "grad_accum", int, "Number of times to split batch into sub-batches for gradient accumulation.", default=1)
        self.add_argument("disp", "display", int, "Number of iterations to display progress when training nnet. No display if 0.", default=0)

    @property
    def delim(self) -> str:
        return "_"
