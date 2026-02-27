import time
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from deepxube.base.trainer import TrainArgs
from deepxube.nnet import nnet_utils

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn


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
                         criterion: nn.Module, device: torch.device, train_itr: int, train_args: TrainArgs, start_time: float) -> Tuple[NDArray, float]:
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
    ctgs_nnet: Tensor = nnet(inputs_batch)[0]

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
              f"nnet_ctg: %.2f, time: {time.time() - start_time:.2f}" % (train_itr, lr_itr, loss.item(), ctgs_batch.mean().item(), ctgs_nnet.mean().item()))

    return ctgs_nnet.cpu().data.numpy(), float(loss.item())
