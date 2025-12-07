import time
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from deepxube.utils.data_utils import sel_l
from deepxube.nnet import nnet_utils

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn


@dataclass
class TrainArgs:
    """
    :param batch_size: Batch size
    :param lr: Initial learning rate
    :param lr_d: Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)
    :param max_itrs: Maximum number of iterations
    :param balance_steps: If true, steps are balanced based on solve percentage
    :param rb: amount of data generated from previous updates to keep in replay buffer. Total replay buffer size will
    then be train_args.batch_size * up_args.up_gen_itrs * rb_past_up. Cost-to-go target is not recomputed.
    :param targ_up_searches: If > 0, do a greedy search with updater for minimum given number of searches to test
    if target network should be updated. Otherwise, it will be updated automatically.
    :param loss_thresh: Loss threshold
    :param display: Number of iterations to display progress. No display if 0.
    """
    batch_size: int
    lr: float
    lr_d: float
    max_itrs: int
    balance_steps: bool
    rb: int
    targ_up_searches: int
    loss_thresh: float
    display: bool


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

    def sample(self, num: int) -> List[NDArray]:
        sel_idxs: NDArray = np.random.randint(self.size(), size=num)

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


def ctgs_summary(ctgs_l: List[NDArray]) -> Tuple[float, float, float]:
    ctgs_min: float = np.inf
    ctgs_max: float = -np.inf
    ctgs_mean: float = 0

    num_tot: int = 0
    for ctgs in ctgs_l:
        ctgs_min = min(ctgs.min(), ctgs_min)
        ctgs_max = max(ctgs.max(), ctgs_max)
        ctgs_mean += ctgs.sum()
        num_tot += ctgs.shape[0]
    ctgs_mean = ctgs_mean/float(num_tot)

    return ctgs_mean, ctgs_min, ctgs_max


def train_heur_nnet_step(nnet: nn.Module, inputs_np: List[NDArray], ctgs_np: NDArray, optimizer: Optimizer,
                         criterion: nn.Module, device: torch.device, train_itr: int, train_args: TrainArgs) -> float:
    # train network
    nnet.train()

    # zero the parameter gradients
    optimizer.zero_grad()
    lr_itr: float = train_args.lr * (train_args.lr_d ** train_itr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_itr

    # send data to device
    inputs_batch: List[Tensor] = nnet_utils.to_pytorch_input(inputs_np, device)
    ctgs_batch: Tensor = torch.tensor(ctgs_np, device=device)

    # forward
    ctgs_nnet: Tensor = nnet(inputs_batch)

    # loss
    assert ctgs_nnet.size() == ctgs_batch.size()
    loss = criterion(ctgs_nnet, ctgs_batch)

    # backwards
    loss.backward()

    # step
    optimizer.step()

    # display progress
    if (train_args.display > 0) and (train_itr % train_args.display == 0):
        print("Itr: %i, lr: %.2E, loss: %.2E, targ_ctg: %.2f, "
              "nnet_ctg: %.2f, " % (train_itr, lr_itr, loss.item(), ctgs_batch.mean().item(), ctgs_nnet.mean().item()))

    return float(loss.item())
