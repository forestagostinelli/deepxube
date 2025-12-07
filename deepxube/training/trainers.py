from typing import List, Tuple, Dict, Optional

from deepxube.base.heuristic import HeurNNet, HeurNNetV, HeurNNetQ
from deepxube.base.updater import UpdateHeur, UpHeurArgs
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, get_eq_weighted_perf
from deepxube.updater.updaters import UpdateHeurGrPolVEnum, UpdateHeurGrPolQEnum
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

            # wo_soln_steps: NDArray = np.arange(1, len(self.step_probs))
            # wo_soln_weights: NDArray = (1.0 / wo_soln_steps)/(1.0 / wo_soln_steps).sum()
            # self.step_probs[2:] = wo_soln_weights / 2.0
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


class TrainHeur:
    def __init__(self, updater: UpdateHeur, nnet_file: str, nnet_targ_file: str, status_file: str,
                 device: torch.device, on_gpu: bool, writer: SummaryWriter, train_args: TrainArgs) -> None:
        self.updater: UpdateHeur = updater
        self.nnet: nn.Module = updater.get_heur_nnet().get_nnet()
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
        self.updater.set_heur_file(self.nnet_targ_file)

        self.nnet.to(self.device)
        self.nnet = nn.DataParallel(self.nnet)

        # init greedy perf for update
        if self.train_args.targ_up_searches > 0:
            print("Getting init greedy performance")
            per_solved: float = self.update_greedy_perf(0)
            self.status.per_solved_best = per_solved

        # init replay buffer
        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = updater.get_shapes_dtypes()
        db_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes]
        db_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes]
        self.db: DataBuffer = DataBuffer(self.train_args.batch_size * self.updater.up_args.up_gen_itrs, db_shapes,
                                         db_dtypes)

        # optimizer and criterion
        self.optimizer: Optimizer = optim.Adam(self.nnet.parameters(), lr=self.train_args.lr)
        self.criterion = nn.MSELoss()

    def update_step(self) -> None:
        # print info
        start_info_l: List[str] = [f"itr: {self.status.itr}", f"update_num: {self.status.update_num}",
                                   f"targ_update: {self.status.targ_update_num}"]

        num_gen: int = self.train_args.batch_size * self.updater.up_args.up_gen_itrs
        start_info_l.append(f"num_gen: {format(num_gen, ',')}")
        if self.train_args.balance_steps:
            start_info_l.append(f"step max (curr): {self.status.step_max_curr}")
        print(f"\nGetting Data - {', '.join(start_info_l)}")
        times: Times = Times()

        # start updater
        start_time = time.time()
        to_main_q, from_main_qs = self.updater.start_update(self.status.step_probs.tolist(), num_gen, self.device,
                                                            self.on_gpu, self.status.targ_update_num,
                                                            self.train_args.batch_size)
        self.db.clear()
        times.record_time("up_start", time.time() - start_time)

        ctgs_l: List[NDArray] = []
        if not self.updater.up_args.sync_main:
            # get update data
            start_time = time.time()
            while self.db.size() < num_gen:
                data_l: List[List[NDArray]] = self.updater.get_update_data()
                for data in data_l:
                    ctgs_l.append(data[-1])
                    self.db.add(data)
            times.record_time("up_data", time.time() - start_time)

        # train nnet
        update_train_itr: int = 0
        loss: float = np.inf
        while update_train_itr < self.updater.up_args.up_itrs:
            batch: List[NDArray]
            if not self.updater.up_args.sync_main:
                batch = self.db.sample(self.train_args.batch_size)
            else:
                # data from updater should not be more that train_args.batch_size
                start_time = time.time()
                if self.db.size() == num_gen:
                    batch = self.db.sample(self.train_args.batch_size)
                else:
                    self.nnet.eval()
                    while self.db.size() < ((update_train_itr + 1) * self.train_args.batch_size):
                        # get heuristic values for ongoing search
                        q_res: Optional[Tuple[int, List[SharedNDArray]]] = get_nowait_noerr(to_main_q)
                        if q_res is not None:
                            proc_id, inputs_np_shm = q_res
                            nnet_in_out_shared_q(self.nnet, inputs_np_shm, self.updater.up_args.up_nnet_batch_size,
                                                 self.device, from_main_qs[proc_id])

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
            inputs_batch_np: List[NDArray] = batch[:-1]
            ctgs_batch_np: NDArray = np.expand_dims(batch[-1].astype(np.float32), 1)
            self.nnet.train()
            loss = train_heur_nnet_step(self.nnet, inputs_batch_np, ctgs_batch_np, self.optimizer, self.criterion,
                                        self.device, self.status.itr, self.train_args)

            update_train_itr += 1
            self.status.itr += 1
            times.record_time("train", time.time() - start_time)

        # end update
        start_time = time.time()
        step_to_search_perf: Dict[int, PathFindPerf] = self.updater.end_update()

        per_solved_ave, path_costs_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)
        self.writer.add_scalar("train/pathfind/solved", per_solved_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/path_cost", path_costs_ave, self.status.itr)
        self.writer.add_scalar("train/pathfind/search_itrs", search_itrs_ave, self.status.itr)

        if self.train_args.balance_steps:
            self.status.update_step_probs(step_to_search_perf)

        ctgs_mean, ctgs_min, ctgs_max = ctgs_summary(ctgs_l)
        self.writer.add_scalar("train/ctgs/mean", ctgs_mean, self.status.itr)
        self.writer.add_scalar("train/ctgs/min", ctgs_min, self.status.itr)
        self.writer.add_scalar("train/ctgs/max", ctgs_max, self.status.itr)

        post_up_info_l: List[str] = [f"%solved: {per_solved_ave:.2f}", f"path_costs: {path_costs_ave:.3f}",
                                     f"search_itrs: {search_itrs_ave:.3f}",
                                     f"cost-to-go (mean/min/max): {ctgs_mean:.2f}/{ctgs_min:.2f}/{ctgs_max:.2f}"]
        print(f"Data - {', '.join(post_up_info_l)}")

        times.record_time("up_end", time.time() - start_time)

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
                start_time = time.time()
                per_solved: float = self.update_greedy_perf(self.status.targ_update_num + 1)
                print(f"Greedy policy solved (best): {per_solved:.2f}% ({self.status.per_solved_best:.2f}%)")
                if per_solved > self.status.per_solved_best:
                    update_targ = True
                    self.status.per_solved_best = per_solved
                times.record_time("greedy_policy_test", time.time() - start_time)

        if update_targ:
            shutil.copy(self.nnet_file, self.nnet_targ_file)
            self.status.targ_update_num = self.status.targ_update_num + 1
        self.status.update_num += 1

        # noinspection PyTypeChecker
        pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)
        print(f"Train - itrs: {update_train_itr}, loss: {loss:.2E}, targ_updated: {update_targ}")
        print(f"Times - {times.get_time_str()}")

    def update_greedy_perf(self, update_num: int) -> float:
        # get updater
        updater_greedy: UpdateHeur
        heur_nnet: HeurNNet = self.updater.get_heur_nnet()
        up_heur_args: UpHeurArgs = UpHeurArgs(self.updater.up_args, False, 1)
        up_heur_args.up_args.sync_main = False
        if isinstance(heur_nnet, HeurNNetV):
            updater_greedy = UpdateHeurGrPolVEnum(self.updater.env, up_heur_args, heur_nnet, 0.0)
        elif isinstance(heur_nnet, HeurNNetQ):
            updater_greedy = UpdateHeurGrPolQEnum(self.updater.env, up_heur_args, heur_nnet, 0.0, 0.0)
        else:
            raise ValueError(f"Unknown heuristic function type {heur_nnet}")

        # do greedy update
        updater_greedy.set_heur_file(self.nnet_file)
        step_probs = np.ones(self.updater.up_args.step_max + 1) / (self.updater.up_args.step_max + 1)
        num_gen: int = self.updater.up_args.up_search_itrs * self.train_args.targ_up_searches
        updater_greedy.start_update(step_probs.tolist(), num_gen, self.device, self.on_gpu, update_num,
                                    self.train_args.batch_size)
        while updater_greedy.num_generated < num_gen:
            self.updater.get_update_data()

        step_to_search_perf: Dict[int, PathFindPerf] = updater_greedy.end_update()
        per_solved_ave, path_cost_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)
        # print(f"%solved: {per_solved_ave:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f} "
        #      f"(greedy perf)")

        return per_solved_ave
