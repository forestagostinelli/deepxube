from typing import List, Tuple, Dict

from deepxube.base.heuristic import HeurNNet, HeurNNetV, HeurNNetQ
from deepxube.base.updater import UpdateHeur
from deepxube.pathfinding.pathfinding_utils import PathFindPerf, get_eq_weighted_perf
from deepxube.updater.updaters import UpdateHeurGrPolVEnum, UpdateHeurGrPolQEnum
from deepxube.training.train_utils import ReplayBuffer, train_heur_nnet, TrainArgs, ctgs_summary
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
        self.step_max: int = step_max
        self.step_probs: NDArray
        self.split_idx: int = 1
        if balance_steps:
            self.step_probs = np.zeros(self.step_max + 1)
            self.step_probs[0] = 0.5

            wo_soln_steps: NDArray = np.arange(1, len(self.step_probs))
            wo_soln_weights: NDArray = (1.0 / wo_soln_steps)/(1.0 / wo_soln_steps).sum()
            self.step_probs[1:] = wo_soln_weights / 2.0
        else:
            self.step_probs = np.ones(self.step_max + 1)/(self.step_max + 1)
        self.per_solved_best: float = 0.0

    def update_step_probs(self, step_to_search_perf: Dict[int, PathFindPerf]) -> None:
        self.split_idx = self._get_split_idx(step_to_search_perf)
        self.step_probs[np.arange(0, self.split_idx + 1)] = 1 / (self.split_idx + 1)
        if self.split_idx < self.step_max:
            self.step_probs[np.arange(0, self.split_idx + 1)] = self.step_probs[np.arange(0, self.split_idx + 1)] / 2.0
            wo_soln_steps: NDArray = np.arange(self.split_idx + 1, self.step_max + 1)
            wo_soln_weights: NDArray = (1.0 / wo_soln_steps)/(1.0 / wo_soln_steps).sum()
            self.step_probs[wo_soln_steps] = wo_soln_weights / 2.0

            # num_steps_left: int = self.step_max - self.split_idx
            # self.step_probs[np.arange(self.split_idx + 1, self.step_max + 1)] = 1 / num_steps_left / 2.0
        """
        for step in range(self.step_max + 1):
            if step not in step_to_search_perf.keys():
                per_solved_per_step_l.append(0.0)
            else:
                search_perf: PathFindPerf = step_to_search_perf[step]
                per_solved_per_step_l.append(search_perf.per_solved())
        per_solved_per_step: NDArray = np.array(per_solved_per_step_l)

        wo_soln_mask: NDArray = np.array(per_solved_per_step == 0)
        num_wo_soln: int = int(np.sum(wo_soln_mask))
        if num_wo_soln == 0:
            self.step_probs = per_solved_per_step / per_solved_per_step.sum()
        else:
            w_soln_mask: NDArray = per_solved_per_step > 0
            w_soln_weights: NDArray = per_solved_per_step[w_soln_mask]/per_solved_per_step[w_soln_mask].sum()
            self.step_probs[w_soln_mask] = w_soln_weights / 2.0

            wo_soln_steps: NDArray = np.arange(len(self.step_probs))[wo_soln_mask]
            wo_soln_weights: NDArray = (1.0 / wo_soln_steps)/(1.0 / wo_soln_steps).sum()
            self.step_probs[wo_soln_mask] = wo_soln_weights / 2.0
            # num_w_soln_eff: float = per_solved_per_step.sum() / 100.0
            # num_tot_eff: float = num_w_soln_eff + 1
            # self.step_probs = num_w_soln_eff * per_solved_per_step / per_solved_per_step.sum() / num_tot_eff
            # self.step_probs[per_solved_per_step == 0] = 1 / num_tot_eff / num_wo_soln
        """

    def _get_split_idx(self, step_to_search_perf: Dict[int, PathFindPerf]) -> int:
        per_solved_l: List[float] = []
        steps_sort: List[int] = list(step_to_search_perf.keys())
        steps_sort.sort()
        for step_idx, step in enumerate(steps_sort):
            per_solved_l.append(step_to_search_perf[step].per_solved())
            if float(np.mean(per_solved_l)) < 50.0:
                return steps_sort[max(step_idx - 1, 0)]
        return self.step_max


class TrainHeur:
    def __init__(self, updater: UpdateHeur, step_max: int, nnet_file: str, nnet_targ_file: str, status_file: str,
                 device: torch.device, on_gpu: bool, writer: SummaryWriter, train_args: TrainArgs,
                 rb_past_up: int) -> None:
        self.updater: UpdateHeur = updater
        self.step_max: int = step_max
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
            print(f"Loaded with itr: {self.status.itr}, update_num: {self.status.update_num}")
        else:
            self.status = Status(self.step_max, train_args.balance_steps)
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
        rb_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes]
        rb_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes]
        self.rb: ReplayBuffer = ReplayBuffer(self.train_args.batch_size * updater.up_args.up_gen_itrs * rb_past_up,
                                             rb_shapes, rb_dtypes)

        # optimizer and criterion
        self.optimizer: Optimizer = optim.Adam(self.nnet.parameters(), lr=self.train_args.lr)
        self.criterion = nn.MSELoss()

    def update_step(self) -> None:
        # step probs
        if self.train_args.balance_steps:
            steps_show: List[int] = list(np.unique(np.linspace(0, self.status.step_max, 30, dtype=int)))
            step_prob_str: str = ', '.join([f'{step}:{self.status.step_probs[step]:.2E}' for step in steps_show])
            print(f"Split idx: {self.status.split_idx}")
            print(f"Step probs: {step_prob_str}")

        # get update data
        num_gen: int = self.train_args.batch_size * self.updater.up_args.up_gen_itrs
        data_l, step_to_search_perf = self.updater.get_update_data(self.step_max, self.status.step_probs.tolist(),
                                                                   num_gen, self.device, self.on_gpu,
                                                                   self.status.update_num)
        ctgs_l: List[NDArray] = [data[-1] for data in data_l]
        ctgs_summary(ctgs_l, self.writer, self.status.itr)
        self.updater.print_update_summary(step_to_search_perf, self.writer, self.status.itr)
        if self.train_args.balance_steps:
            self.status.update_step_probs(step_to_search_perf)

        # get batches
        print(f"Getting training batches, Replay buffer size: {format(self.rb.size(), ',')}")
        start_time = time.time()
        for data in data_l:
            self.rb.add(data)
        batches: List[Tuple[List[NDArray], NDArray]] = []
        for _ in range(self.updater.up_args.up_itrs):
            arrays_samp: List[NDArray] = self.rb.sample(self.train_args.batch_size)
            inputs_batch_np: List[NDArray] = arrays_samp[:-1]
            ctgs_batch_np: NDArray = np.expand_dims(arrays_samp[-1].astype(np.float32), 1)
            batches.append((inputs_batch_np, ctgs_batch_np))
        print(f"Time: {time.time() - start_time}")

        # train nnet
        print("Training model for %i iterations" % len(batches))
        last_loss = train_heur_nnet(self.nnet, batches, self.optimizer, self.criterion, self.device, self.status.itr,
                                    self.train_args)
        self.status.itr += self.updater.up_args.up_itrs
        print("Last loss was %f" % last_loss)

        # save nnet
        torch.save(self.nnet.state_dict(), self.nnet_file)

        # update nnet
        update: bool = False
        if self.train_args.targ_up_searches <= 0:
            update = True
        else:
            print("Getting greedy performance")
            per_solved: float = self.update_greedy_perf(self.status.update_num + 1)
            print(f"Greedy policy solved (best): {per_solved:.2f}% ({self.status.per_solved_best:.2f}%)")
            if per_solved > self.status.per_solved_best:
                update = True
                self.status.per_solved_best = per_solved

        if update:
            print("Updating target network")
            shutil.copy(self.nnet_file, self.nnet_targ_file)
            self.status.update_num = self.status.update_num + 1

        # noinspection PyTypeChecker
        pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)

    def update_greedy_perf(self, update_num: int) -> float:
        # get updater
        updater_greedy: UpdateHeur
        heur_nnet: HeurNNet = self.updater.get_heur_nnet()
        if isinstance(heur_nnet, HeurNNetV):
            updater_greedy = UpdateHeurGrPolVEnum(self.updater.env, self.updater.up_args, False, 1, heur_nnet, 0.0)
        elif isinstance(heur_nnet, HeurNNetQ):
            updater_greedy = UpdateHeurGrPolQEnum(self.updater.env, self.updater.up_args, False, heur_nnet, 0.0, 0.0)
        else:
            raise ValueError(f"Unknown heuristic function type {heur_nnet}")

        # do greedy update
        updater_greedy.set_heur_file(self.nnet_file)
        step_probs = np.ones(self.step_max + 1) / (self.step_max + 1)
        num_gen: int = self.updater.up_args.up_search_itrs * self.train_args.targ_up_searches
        _, step_to_search_perf = updater_greedy.get_update_data(self.step_max, step_probs.tolist(), num_gen,
                                                                self.device, self.on_gpu, update_num)
        per_solved_ave, path_cost_ave, search_itrs_ave = get_eq_weighted_perf(step_to_search_perf)
        print(f"%solved: {per_solved_ave:.2f}, path_costs: {path_cost_ave:.3f}, search_itrs: {search_itrs_ave:.3f} "
              f"(greedy perf)")

        return per_solved_ave
