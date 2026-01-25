from typing import List, Tuple, Dict, Optional

from deepxube.base.heuristic import HeurNNet
from deepxube.base.updater import UpdateHeur
from deepxube.pathfinding.utils.performance import PathFindPerf, get_eq_weighted_perf
from deepxube.training.train_utils import DataBuffer, train_heur_nnet_step, TrainArgs, ctgs_summary
from deepxube.nnet.nnet_utils import nnet_in_out_shared_q
from deepxube.utils.data_utils import get_nowait_noerr
from deepxube.utils.timing_utils import Times
from deepxube.utils.data_utils import sel_l, SharedNDArray
from deepxube.nnet import nnet_utils

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle

from multiprocessing import Queue
import os
import shutil

import numpy as np
from numpy.typing import NDArray
import time


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
        self.itr_to_in_out: Dict[int, Tuple[NDArray, NDArray]] = dict()
        self.itr_to_steps_to_pathfindperf: Dict[int, Dict[int, PathFindPerf]] = dict()

    def update_step_probs(self, step_to_search_perf: Dict[int, PathFindPerf]) -> None:
        ave_solve: float = float(np.mean([step_to_search_perf[step].per_solved()
                                          for step in step_to_search_perf.keys()]))
        if ave_solve >= 50.0:
            self.step_max_curr = min(self.step_max_curr * 2, self.step_max)

        self.step_probs = np.zeros(self.step_max + 1)
        self.step_probs[np.arange(0, self.step_max_curr + 1)] = 1 / (self.step_max_curr + 1)


class TrainHeur:
    def __init__(self, nnet: HeurNNet, updater: UpdateHeur, to_main_q: Queue, from_main_qs: List[Queue], nnet_file: str,
                 nnet_targ_file: str, status_file: str, device: torch.device, on_gpu: bool, writer: SummaryWriter,
                 train_args: TrainArgs) -> None:
        self.updater: UpdateHeur = updater
        self.to_main_q: Queue = to_main_q
        self.from_main_qs: List[Queue] = from_main_qs
        self.nnet: nn.Module = nnet
        self.nnet_file = nnet_file
        self.nnet_targ_file: str = nnet_targ_file
        self.writer: SummaryWriter = writer
        self.train_args: TrainArgs = train_args
        self.device: torch.device = device
        self.on_gpu: bool = on_gpu

        # load status
        self.status_file: str = status_file
        self.status: Status
        if os.path.isfile(self.status_file):
            self.status = pickle.load(open(self.status_file, "rb"))
            print(f"Loaded with itr: {self.status.itr}, update_num: {self.status.update_num}, "
                  f"targ_update_num: {self.status.targ_update_num}")
        else:
            self.status = Status(self.updater.up_args.step_max, train_args.balance_steps)
            # noinspection PyTypeChecker
            pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)

        # load nnet
        if os.path.isfile(self.nnet_file):
            self.nnet = nnet_utils.load_nnet(self.nnet_file, self.nnet)
        else:
            torch.save(self.nnet.state_dict(), self.nnet_file)
        if not os.path.isfile(self.nnet_targ_file):
            torch.save(self.nnet.state_dict(), self.nnet_targ_file)

        self.nnet.to(self.device)
        self.nnet = nn.DataParallel(self.nnet)

        """
        # init greedy perf for update
        if self.train_args.targ_up_searches > 0:
            print("Getting init greedy performance")
            per_solved: float = self._update_greedy_perf(0)
            self.status.per_solved_best = per_solved
        """

        # init data buffer
        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = updater.get_heur_train_shapes_dtypes()
        db_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes]
        db_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes]
        self.db: DataBuffer = DataBuffer(self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs(), db_shapes,
                                         db_dtypes)

        # optimizer and criterion
        self.optimizer: Optimizer = optim.Adam(self.nnet.parameters(), lr=self.train_args.lr)
        self.criterion = nn.MSELoss()
        self.train_start_time = time.time()

    def update_step(self) -> None:
        self.db.clear()
        itr_init: int = self.status.itr

        # print info
        start_info_l: List[str] = [f"itr: {self.status.itr}", f"update_num: {self.status.update_num}",
                                   f"targ_update: {self.status.targ_update_num}"]

        num_gen: int = self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs()
        start_info_l.append(f"num_gen: {format(num_gen, ',')}")
        if self.train_args.balance_steps:
            start_info_l.append(f"step max (curr): {self.status.step_max_curr}")
        print(f"\nGetting Data - {', '.join(start_info_l)}")
        times: Times = Times()

        # start updater
        start_time = time.time()
        self.updater.start_update(self.status.step_probs.tolist(), num_gen, self.status.targ_update_num,
                                  self.train_args.batch_size, self.device, self.on_gpu)
        times.record_time("up_start", time.time() - start_time)

        # do training
        self.train_start_time = time.time()
        loss: float
        ctgs_l: List[NDArray]
        if not self.updater.up_args.sync_main:
            ctgs_l = self._get_update_data(num_gen, times)
            self._end_update(itr_init, ctgs_l, times)
            loss = self._train_no_sync_main(times)
        else:
            loss, ctgs_l = self._train_sync_main(num_gen, times)
            self._end_update(itr_init, ctgs_l, times)

        # save nnet
        start_time = time.time()
        torch.save(self.nnet.state_dict(), self.nnet_file)
        times.record_time("save", time.time() - start_time)

        # update nnet
        update_targ: bool = False
        if loss < self.train_args.loss_thresh:
            if self.train_args.targ_up_searches <= 0:
                update_targ = True
            else:
                """
                start_time = time.time()
                per_solved: float = self._update_greedy_perf(self.status.targ_update_num + 1)
                print(f"Greedy policy solved (best): {per_solved:.2f}% ({self.status.per_solved_best:.2f}%)")
                if per_solved > self.status.per_solved_best:
                    update_targ = True
                    self.status.per_solved_best = per_solved
                times.record_time("greedy_policy_test", time.time() - start_time)
                """
                raise NotImplementedError

        if update_targ:
            shutil.copy(self.nnet_file, self.nnet_targ_file)
            self.status.targ_update_num = self.status.targ_update_num + 1
        self.status.update_num += 1

        # noinspection PyTypeChecker
        pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)
        print(f"Train - itrs: {self.updater.up_args.up_itrs}, loss: {loss:.2E}, targ_updated: {update_targ}")
        print(f"Times - {times.get_time_str()}")

    def _get_update_data(self, num_gen: int, times: Times) -> List[NDArray]:
        start_time = time.time()
        ctgs_l: List[NDArray] = []
        while self.db.size() < num_gen:
            data_l: List[List[NDArray]] = self.updater.get_update_data()
            for data in data_l:
                ctgs_l.append(data[-1])
                self.db.add(data)
        times.record_time("up_data", time.time() - start_time)

        return ctgs_l

    def _train_no_sync_main(self, times: Times) -> float:
        # train
        loss: float = np.inf
        first_itr_in_update: bool = True
        for _ in range(self.updater.up_args.up_itrs):
            # sample data
            start_time = time.time()
            batch: List[NDArray] = self.db.sample(self.train_args.batch_size)
            times.record_time("data_samp", time.time() - start_time)

            # train
            loss = self._train_itr(batch[:-1], batch[-1], first_itr_in_update, times)
            first_itr_in_update = False

        return loss

    def _train_sync_main(self, num_gen: int, times: Times) -> Tuple[float, List[NDArray]]:
        loss: float = np.inf
        ctgs_l: List[NDArray] = []
        update_train_itr: int = 0
        first_itr_in_update: bool = True
        while update_train_itr < self.updater.up_args.up_itrs:
            batch: List[NDArray]
            # data from updater should not be more that train_args.batch_size
            start_time = time.time()
            if self.db.size() == num_gen:
                batch = self.db.sample(self.train_args.batch_size)
            else:
                self.nnet.eval()
                while self.db.size() < ((update_train_itr + 1) * self.train_args.batch_size):
                    # get heuristic values for ongoing search
                    q_res: Optional[Tuple[int, List[SharedNDArray]]] = get_nowait_noerr(self.to_main_q)
                    if q_res is not None:
                        proc_id, inputs_np_shm = q_res
                        nnet_in_out_shared_q(self.nnet, inputs_np_shm, self.updater.up_args.nnet_batch_size,
                                             self.device, self.from_main_qs[proc_id])

                    # get update data
                    data_l_i: List[List[NDArray]] = self.updater.get_update_data(nowait=True)
                    for data in data_l_i:
                        ctgs_l.append(data[-1])
                        self.db.add(data)
                sel_idxs: NDArray = np.arange(update_train_itr * self.train_args.batch_size,
                                              (update_train_itr + 1) * self.train_args.batch_size)
                batch = sel_l(self.db.arrays, sel_idxs)

            times.record_time("up_data", time.time() - start_time)

            # train
            start_time = time.time()
            loss = self._train_itr(batch[:-1], batch[-1], first_itr_in_update, times)
            update_train_itr += 1
            first_itr_in_update = False
            times.record_time("train", time.time() - start_time)

        return loss, ctgs_l

    def _train_itr(self, inputs_batch_np: List[NDArray], ctgs_batch_np: NDArray, log_in_out: bool, times: Times) -> float:
        start_time = time.time()
        ctgs_batch_np = np.expand_dims(ctgs_batch_np.astype(np.float32), 1)
        self.nnet.train()
        ctgs_batch_nnet, loss = train_heur_nnet_step(self.nnet, inputs_batch_np, ctgs_batch_np, self.optimizer, self.criterion, self.device, self.status.itr,
                                                     self.train_args, self.train_start_time)
        self.writer.add_scalar("train/loss", loss, self.status.itr)

        if log_in_out:
            self.status.itr_to_in_out[self.status.itr] = (ctgs_batch_np, ctgs_batch_nnet)
        self.status.itr += 1
        times.record_time("train", time.time() - start_time)
        return loss

    def _end_update(self, itr_init: int, ctgs_l: List[NDArray], times: Times) -> None:
        start_time = time.time()
        step_to_search_perf: Dict[int, PathFindPerf] = self.updater.end_update()
        self.status.itr_to_steps_to_pathfindperf[itr_init] = step_to_search_perf
        if self.train_args.balance_steps:
            self.status.update_step_probs(step_to_search_perf)

        per_solved_ave, path_costs_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)
        ctgs_mean, ctgs_min, ctgs_max = ctgs_summary(ctgs_l)

        self.writer.add_scalar("train/pathfind/solved", per_solved_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/path_cost", path_costs_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/search_itrs", search_itrs_ave, self.status.itr)

        self.writer.add_scalar("train/ctgs/mean", ctgs_mean, self.status.itr)
        self.writer.add_scalar("train/ctgs/min", ctgs_min, self.status.itr)
        self.writer.add_scalar("train/ctgs/max", ctgs_max, self.status.itr)

        post_up_info_l: List[str] = [f"%solved: {per_solved_ave:.2f}", f"path_costs: {path_costs_ave:.3f}",
                                     f"search_itrs: {search_itrs_ave:.3f}",
                                     f"cost-to-go (mean/min/max): {ctgs_mean:.2f}/{ctgs_min:.2f}/{ctgs_max:.2f}"]

        print(f"Data - {', '.join(post_up_info_l)}")
        times.record_time("up_end", time.time() - start_time)

    """
    def _update_greedy_perf(self, update_num: int) -> float:
        # get updater
        updater_greedy: UpdateHeurRL
        heur_nnet: HeurNNetPar = self.updater.get_heur_nnet()
        up_greedy_args: UpGreedyPolicyArgs = UpGreedyPolicyArgs(0.0, 0.0)
        up_heur_args: UpHeurArgs = UpHeurArgs(False, 1)
        up_args = dataclasses.replace(self.updater.up_args)
        up_args.sync_main = False
        if isinstance(heur_nnet, HeurNNetParV):
            updater_greedy = UpdateHeurGrPolVEnum(self.updater.domain, up_args, up_heur_args, up_greedy_args, heur_nnet)
        elif isinstance(heur_nnet, HeurNNetParQ):
            updater_greedy = UpdateHeurGrPolQEnum(self.updater.domain, up_args, up_heur_args, up_greedy_args, heur_nnet)
        else:
            raise ValueError(f"Unknown heuristic function type {heur_nnet}")

        # do greedy update
        updater_greedy.set_heur_file(self.nnet_file)
        step_probs = np.ones(self.updater.up_args.step_max + 1) / (self.updater.up_args.step_max + 1)
        num_gen: int = self.updater.up_args.search_itrs * self.train_args.targ_up_searches
        updater_greedy.start_procs()
        updater_greedy.start_update(step_probs.tolist(), num_gen, update_num, self.train_args.batch_size,
                                    self.device, self.on_gpu)
        while updater_greedy.num_generated < num_gen:
            self.updater.get_update_data()

        step_to_search_perf: Dict[int, PathFindPerf] = updater_greedy.end_update()
        updater_greedy.stop_procs()
        per_solved_ave, path_cost_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)
        # print(f"%solved: {per_solved_ave:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f} "
        #      f"(greedy perf)")

        return per_solved_ave
    """
