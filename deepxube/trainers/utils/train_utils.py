import time
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from deepxube.base.heuristic import DeepXubeNNet
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


def train_nnet_step(nnet: Union[DeepXubeNNet, nn.DataParallel], data_np: List[NDArray], optimizer: Optimizer, device: torch.device, train_itr: int,
                    train_args: TrainArgs, start_time: float) -> Tuple[List[Tensor], float]:
    # train network
    nnet.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # send data to device
    data: List[Tensor] = nnet_utils.to_pytorch_input(data_np, device)

    # forward
    fwd_tr_tensors: List[Tensor] = nnet(data)
    print_info: bool = (train_args.display > 0) and (train_itr % train_args.display == 0)
    if isinstance(nnet, nn.DataParallel):
        loss, loss_str = nnet.module.get_loss_and_info(fwd_tr_tensors, print_info)
    else:
        loss, loss_str = nnet.get_loss_and_info(fwd_tr_tensors, print_info)

    # backwards
    loss.backward()

    # step
    optimizer.step()

    # display progress
    if print_info:
        print_str: str = f"Itr: {train_itr}, loss: {loss.item():.2E}"
        if loss_str is not None:
            print_str = f"{print_str}, {loss_str}"
        print_str = f"{print_str}, Time: {time.time() - start_time:.2f}"
        print(print_str)

    # return
    return fwd_tr_tensors, float(loss.item())
