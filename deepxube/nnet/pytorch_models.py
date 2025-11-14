from typing import Any, List, Optional, Union, Callable
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn.utils import parametrizations
import numpy as np


class OneHot(nn.Module):
    def __init__(self, data_dim: int, one_hot_depth: int) -> None:
        super().__init__()
        self.data_dim: int = data_dim
        self.one_hot_depth: int = one_hot_depth

    def forward(self, x: Tensor) -> Tensor:
        # preprocess input
        if self.one_hot_depth > 0:
            x = nn.functional.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.data_dim * self.one_hot_depth)
        else:
            x = x.float()

        return x


class SPLASH(nn.Module):
    def __init__(self, num_hinges: int = 5, init: str = "RELU"):
        super().__init__()
        assert num_hinges > 0, "Number of hinges should be greater than zero, but is %s" % num_hinges
        assert ((num_hinges + 1) % 2) == 0, "Number of hinges should be odd, but is %s" % num_hinges
        init = init.upper()

        self.num_hinges: int = num_hinges
        self.num_each_side: int = int((self.num_hinges + 1) / 2)

        self.hinges: List[float] = list(np.linspace(0, 2.5, self.num_each_side))

        self.output_bias: Parameter = Parameter(torch.zeros(1), requires_grad=True)

        self.coeffs_right: Parameter
        self.coeffs_left: Parameter
        if init == "RELU":
            self.coeffs_right = Parameter(torch.cat((torch.ones(1), torch.zeros(self.num_each_side - 1))),
                                          requires_grad=True)
            self.coeffs_left = Parameter(torch.zeros(self.num_each_side), requires_grad=True)
        elif init == "LINEAR":
            self.coeffs_right = Parameter(torch.cat((torch.ones(1), torch.zeros(self.num_each_side - 1))),
                                          requires_grad=True)
            self.coeffs_left = Parameter(torch.cat((-torch.ones(1), torch.zeros(self.num_each_side - 1))),
                                         requires_grad=True)
        else:
            raise ValueError("Unknown init %s" % init)

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = torch.zeros_like(x)

        # output for x > 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_right[idx] * torch.clamp(x - self.hinges[idx], min=0)

        # output for x < 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_left[idx] * torch.clamp(-x - self.hinges[idx], min=0)

        output = output + self.output_bias

        return output


class SPLASH1(nn.Module):
    def __init__(self, init: str = "RELU"):
        super().__init__()
        init = init.upper()

        self.output_bias: Parameter = Parameter(torch.zeros(1), requires_grad=True)

        self.coeff_right: Parameter = Parameter(torch.ones(1), requires_grad=True)
        self.coeff_left: Parameter
        if init == "RELU":
            self.coeff_left = Parameter(torch.zeros(1), requires_grad=True)
        elif init == "LINEAR":
            self.coeff_left = Parameter(-torch.ones(1), requires_grad=True)
        else:
            raise ValueError("Unknown init %s" % init)

    def forward(self, x: Tensor) -> Tensor:
        x = (self.coeff_right * nn.functional.relu(x)) - (self.coeff_left * nn.functional.relu(-x)) + self.output_bias

        return x


class LinearAct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = 1

    def forward(self, x: Tensor) -> Tensor:
        self.dummy = 1  # so PyCharm does not complain
        return x


def get_act_fn(act: str) -> nn.Module:
    act = act.upper()
    if act == "RELU":
        return nn.ReLU()
    elif act == "ELU":
        return nn.ELU()
    elif act == "SIGMOID":
        return nn.Sigmoid()
    elif act == "TANH":
        return nn.Tanh()
    elif act == "SPLASH":
        return SPLASH()
    elif act == "SPLASH1":
        return SPLASH1()
    elif act == "LINEAR":
        return LinearAct()
    else:
        raise ValueError("Un-defined activation type %s" % act)


class ResnetModel(nn.Module):
    def __init__(self, block_init: Callable[[], nn.Module], num_resnet_blocks: int, act_fn: str):
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.act_fns: nn.ModuleList = nn.ModuleList()

        # resnet blocks
        for block_num in range(num_resnet_blocks):
            block_net: nn.Module = block_init()
            module_list: nn.ModuleList = nn.ModuleList([block_net])

            self.blocks.append(module_list)
            self.act_fns.append(get_act_fn(act_fn))

    def forward(self, x: Tensor) -> Tensor:
        # resnet blocks
        module_list: nn.Module
        for module_list, act_fn in zip(self.blocks, self.act_fns):
            assert isinstance(module_list, nn.ModuleList)
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        return x


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], acts: List[str], batch_norms: Optional[List[bool]] = None,
                 weight_norms: Optional[List[bool]] = None, group_norms: Optional[List[int]] = None):
        super().__init__()
        if batch_norms is None:
            batch_norms = [False] * len(dims)
        if weight_norms is None:
            weight_norms = [False] * len(dims)
        if group_norms is None:
            group_norms = [-1] * len(dims)
        self.layers: nn.ModuleList = nn.ModuleList()

        # layers
        for dim, act, batch_norm, weight_norm, group_norm in zip(dims, acts, batch_norms, weight_norms, group_norms,
                                                                 strict=True):
            module_list = nn.ModuleList()

            # linear
            if weight_norm:
                module_list.append(nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, dim)))
            else:
                module_list.append(nn.Linear(input_dim, dim))

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm1d(dim))

            # group norm
            if group_norm > 0:
                module_list.append(nn.GroupNorm(group_norm, dim))

            # activation
            module_list.append(get_act_fn(act))
            self.layers.append(module_list)

            input_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        module_list: nn.Module
        for module_list in self.layers:
            assert isinstance(module_list, nn.ModuleList)
            for module in module_list:
                x = module(x)

        return x


class Conv2dModel(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, channel_sizes: List[int], kernel_sizes: List[int], paddings: List[int],
                 layer_acts: List[str], batch_norms: Optional[List[bool]] = None, strides: Optional[List[int]] = None,
                 transpose: bool = False, weight_norms: Optional[List[bool]] = None,
                 dropouts: Optional[List[float]] = None):
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        if strides is None:
            strides = [1] * len(channel_sizes)

        if batch_norms is None:
            batch_norms = [False] * len(channel_sizes)

        if weight_norms is None:
            weight_norms = [False] * len(channel_sizes)

        if dropouts is None:
            dropouts = [0.0] * len(channel_sizes)

        # layers
        for chan_out, kernel_size, padding, batch_norm, act, stride, weight_norm, dropout in \
                zip(channel_sizes, kernel_sizes, paddings, batch_norms, layer_acts, strides, weight_norms,
                    dropouts):

            module_list = nn.ModuleList()

            # linear
            conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d]
            if transpose:
                conv_layer = nn.ConvTranspose2d(chan_in, chan_out, kernel_size, padding=padding, stride=stride)
            else:
                conv_layer = nn.Conv2d(chan_in, chan_out, kernel_size, padding=padding, stride=stride)

            if weight_norm:
                conv_layer = parametrizations.weight_norm(conv_layer)

            module_list.append(conv_layer)

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm2d(chan_out))

            # activation
            module_list.append(get_act_fn(act))

            # dropout
            if dropout > 0.0:
                module_list.append(nn.Dropout(dropout))

            self.layers.append(module_list)

            chan_in = chan_out

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        module_list: nn.Module
        for module_list in self.layers:
            assert isinstance(module_list, nn.ModuleList)
            for module in module_list:
                x = module(x)

        return x
