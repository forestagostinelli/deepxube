import time
from typing import List, Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray

from deepxube.base.heuristic import DeepXubeNNet
from deepxube.base.trainer import TrainArgs
from deepxube.nnet import nnet_utils
from deepxube.utils.data_utils import combine_l_l
from deepxube.utils.misc_utils import split_evenly

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


def get_deepxube_nnet(nnet: Union[DeepXubeNNet, nn.DataParallel]) -> DeepXubeNNet:
    nnet_deepxube: DeepXubeNNet
    if isinstance(nnet, nn.DataParallel):
        assert isinstance(nnet.module, DeepXubeNNet)
        nnet_deepxube = nnet.module
    else:
        nnet_deepxube = nnet

    return nnet_deepxube


def train_nnet_step(nnet: Union[DeepXubeNNet, nn.DataParallel], data_np: List[NDArray], optimizer: Optimizer, device: torch.device, train_itr: int,
                    train_args: TrainArgs, start_time: float) -> Tuple[List[NDArray], float]:
    # train network
    nnet.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # get accum info
    batch_size: int = data_np[0].shape[0]
    assert batch_size >= train_args.grad_accum
    batch_size_accums: List[int] = split_evenly(batch_size, train_args.grad_accum)
    start_idx: int = 0

    print_info: bool = (train_args.display > 0) and (train_itr % train_args.display == 0)
    loss_str: Optional[str] = None
    loss_tot: float = 0.0
    fwd_tr_tensors_tot_np_l: List[List[NDArray]] = []
    for batch_size_accum in batch_size_accums:
        # send data to device
        end_idx: int = start_idx + batch_size_accum
        data_np_accum: List[NDArray] = [data_np_i[start_idx:end_idx] for data_np_i in data_np]
        batch_size_i: int = data_np_accum[0].shape[0]
        data: List[Tensor] = nnet_utils.to_pytorch_input(data_np_accum, device)

        # forward
        fwd_tr_tensors: List[Tensor] = nnet(data)
        loss, loss_str = get_deepxube_nnet(nnet).get_loss_and_info(fwd_tr_tensors, print_info)
        fwd_tr_tensors_tot_np_l.append([tens.cpu().data.numpy() for tens in fwd_tr_tensors])

        # backwards
        loss = (batch_size_i * loss) / batch_size
        loss.backward()
        loss_tot += loss.item()

        start_idx = end_idx

    # step
    optimizer.step()

    # display progress
    if print_info:
        print_str_l: List[str] = [f"Itr: {train_itr}", f"loss: {loss_tot:.2E}"]
        if loss_str is not None:
            print_str_l.append(loss_str)
        print_str_l.append(f"Time: {time.time() - start_time:.2f}")
        print(', '.join(print_str_l))

    # return
    fwd_tr_tensors_np: List[NDArray] = combine_l_l(fwd_tr_tensors_tot_np_l, "concat")
    for fwd_tr_tensor_np in fwd_tr_tensors_np:
        assert fwd_tr_tensor_np.shape[0] == batch_size

    return fwd_tr_tensors_np, loss_tot
