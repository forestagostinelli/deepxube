from typing import List, Tuple, Dict

from deepxube.base.updater import UpdateHeur
from deepxube.pathfinding.pathfinding_utils import PathFindPerf
from deepxube.training.train_utils import ReplayBuffer, train_heur_nnet, TrainArgs, ctgs_summary
from deepxube.nnet import nnet_utils

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os

import numpy as np
from numpy.typing import NDArray
import time


class TrainHeur:
    def __init__(self, updater: UpdateHeur, nnet_file: str, device: torch.device, on_gpu: bool, writer: SummaryWriter,
                 train_args: TrainArgs, rb_past_up: int) -> None:
        self.updater: UpdateHeur = updater
        self.nnet: nn.Module = updater.heur_nnet.get_nnet()
        self.nnet_file = nnet_file
        self.writer: SummaryWriter = writer
        self.train_args: TrainArgs = train_args
        self.device: torch.device = device
        self.on_gpu: bool = on_gpu

        if os.path.isfile(self.nnet_file):
            self.nnet = nnet_utils.load_nnet(self.nnet_file, self.nnet)
        else:
            torch.save(self.nnet.state_dict(), self.nnet_file)

        self.nnet.to(self.device)
        self.nnet = nn.DataParallel(self.nnet)

        shapes_dtypes: List[Tuple[Tuple[int, ...], np.dtype]] = updater.get_shapes_dtypes()
        rb_shapes: List[Tuple[int, ...]] = [x[0] for x in shapes_dtypes]
        rb_dtypes: List[np.dtype] = [x[1] for x in shapes_dtypes]
        self.rb: ReplayBuffer = ReplayBuffer(self.train_args.batch_size * updater.up_args.up_gen_itrs * rb_past_up,
                                             rb_shapes, rb_dtypes)

        self.optimizer: Optimizer = optim.Adam(self.nnet.parameters(), lr=self.train_args.lr)
        self.criterion = nn.MSELoss()

    def update_step(self, step_max: int, step_probs: List[int], itr: int) -> Dict[int, PathFindPerf]:
        num_gen: int = self.train_args.batch_size * self.updater.up_args.up_gen_itrs
        self.updater.set_heur_file(self.nnet_file)
        data_l, step_to_search_perf = self.updater.get_update_data(step_max, step_probs, num_gen, self.device,
                                                                   self.on_gpu, itr)
        ctgs_l: List[NDArray] = [data[-1] for data in data_l]
        ctgs_summary(ctgs_l, self.writer, itr)
        self.updater.print_update_summary(step_to_search_perf, self.writer, itr)

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
        last_loss = train_heur_nnet(self.nnet, batches, self.optimizer, self.criterion, self.device, itr,
                                    self.train_args)
        print("Last loss was %f" % last_loss)

        return step_to_search_perf

    def save_nnet(self) -> None:
        torch.save(self.nnet.state_dict(), self.nnet_file)
