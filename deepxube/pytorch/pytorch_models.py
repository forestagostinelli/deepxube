from typing import List, Optional, Union, Callable, Tuple
import torch
from torch.nn.parameter import Parameter
from torch import Tensor, nn
from torch.nn.utils import parametrizations
import numpy as np


class OneHot(nn.Module):
    def __init__(self, one_hot_depth: int, flatten_oh: bool) -> None:
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.flatten_oh: bool = flatten_oh

    def forward(self, x: Tensor) -> Tensor:
        if self.one_hot_depth > 1:
            x = nn.functional.one_hot(x.long(), self.one_hot_depth).float()
            if not self.flatten_oh:
                return x
            else:
                *leading, d, c = x.shape
                return x.reshape(*leading, d * c)
        else:
            return x.float()


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


class ReLU2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = 1

    def forward(self, x: Tensor) -> Tensor:
        self.dummy = 1  # so PyCharm does not complain
        return nn.functional.relu(x).square()


def get_act_fn(act: str) -> nn.Module:
    act = act.upper()
    if act == "RELU":
        return nn.ReLU()
    if act == "RELU2":
        return ReLU2()
    elif act == "SILU":
        return nn.modules.activation.SiLU()
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


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "nearest"):
        super().__init__()
        self.mode: str = mode
        self.conv: nn.Module = Conv2dModel(in_channels, [out_channels], [3], [1], ["LINEAR"])

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)
        return x


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
        for module_list, act_fn in zip(self.blocks, self.act_fns, strict=True):
            assert isinstance(module_list, nn.ModuleList)
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        return x


class DeepSetEnc(nn.Module):
    def __init__(self, elem_enc_net: nn.Module, elem_enc_dim: int, set_enc_dim: int):
        super().__init__()

        self.elem_enc_net: nn.Module = elem_enc_net

        self.set_enc_dim: int = set_enc_dim
        self.smax_net = nn.Linear(elem_enc_dim, self.set_enc_dim)
        self.enc_val_net = nn.Linear(elem_enc_dim, self.set_enc_dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """
        :param x: Tensor (b, n, *elem_shape)
        :param valid: Tensor (b, n), True/1 = valid element, False/0 = padding
        """
        b, n = x.shape[:2]
        elem_shape = x.shape[2:]
        assert valid.shape == (b, n), f"valid has shape {tuple(valid.shape)}, expected {(b, n)}"

        valid = valid.bool()

        # encode only the valid elements, so padding never reaches the network and doesn't contaminate batch norm statistics
        valid_flat = valid.reshape(-1)  # (b * n)
        x_flat = x.reshape(b * n, *elem_shape)  # (b * n, *elem_shape)
        enc_base = self.elem_enc_net(x_flat[valid_flat])  # (num_valid, *elem_enc_shape)

        # get softmax and corresponding values
        enc_base = enc_base.flatten(1)  # (num_valid, elem_enc_dim), works when num_valid == 0
        smax_valid = self.smax_net(enc_base)   # (num_valid, set_enc_dim)
        vals_valid = self.enc_val_net(enc_base)  # (num_valid, set_enc_dim)

        # scatter back into full (b * n, set_enc_dim) tensors
        neg_fill = torch.finfo(smax_valid.dtype).min
        smax_vals = smax_valid.new_full((b * n, self.set_enc_dim), neg_fill)  # (b * n, set_enc_dim)
        smax_vals[valid_flat] = smax_valid
        enc_vals = vals_valid.new_zeros((b * n, self.set_enc_dim))  # (b * n, set_enc_dim)
        enc_vals[valid_flat] = vals_valid

        smax_vals = smax_vals.view(b, n, self.set_enc_dim)  # (b, n, set_enc_dim)
        enc_vals = enc_vals.view(b, n, self.set_enc_dim)  # (b, n, set_enc_dim)

        # temperature uses the number of real elements in each set
        n_valid = valid.sum(dim=1).clamp(min=1).to(smax_vals.dtype).view(b, 1, 1)
        smax = torch.softmax(smax_vals / n_valid.sqrt(), dim=1)

        return (smax * enc_vals).sum(dim=1)


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], acts: List[str], batch_norms: Optional[List[bool]] = None,
                 weight_norms: Optional[List[bool]] = None, group_norms: Optional[List[int]] = None, zero_inits: Optional[List[bool]] = None,
                 pre: bool = False):
        # not pre: weight -> norm -> act; pre: norm -> act -> weight
        super().__init__()
        if batch_norms is None:
            batch_norms = [False] * len(dims)
        if weight_norms is None:
            weight_norms = [False] * len(dims)
        if group_norms is None:
            group_norms = [-1] * len(dims)
        if zero_inits is None:
            zero_inits = [False] * len(dims)

        self.layers: nn.ModuleList = nn.ModuleList()

        # layers
        for dim, act, batch_norm, weight_norm, group_norm, zero_init in zip(dims, acts, batch_norms, weight_norms, group_norms, zero_inits, strict=True):
            module_list = nn.ModuleList()

            # norms
            norm_dim: int = input_dim if pre else dim

            if batch_norm:
                module_list.append(nn.BatchNorm1d(norm_dim))

            # group norm
            if group_norm > 0:
                module_list.append(nn.GroupNorm(group_norm, norm_dim))

            # activation
            module_list.append(get_act_fn(act))

            # linear
            module_linear: nn.Linear = nn.Linear(input_dim, dim)
            if weight_norm:
                module_linear = nn.utils.parametrizations.weight_norm(module_linear)

            # zero init
            assert not (zero_init and weight_norm), "Zero init and weight norm should not both be True"

            if zero_init:
                nn.init.zeros_(module_linear.weight)
                if module_linear.bias is not None:
                    nn.init.zeros_(module_linear.bias)

            # add linear
            if pre:
                module_list.append(module_linear)
            else:
                module_list.insert(0, module_linear)

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
    def __init__(self, chan_in: int, channel_sizes: List[int], kernel_sizes: List[int], paddings: List[int],
                 layer_acts: List[str], batch_norms: Optional[List[bool]] = None, strides: Optional[List[int]] = None,
                 transpose: bool = False, weight_norms: Optional[List[bool]] = None, group_norms: Optional[List[int]] = None,
                 dropouts: Optional[List[float]] = None, zero_inits: Optional[List[bool]] = None, pre: bool = False):
        # not pre: weight -> norm -> act; pre: norm -> act -> weight
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        if strides is None:
            strides = [1] * len(channel_sizes)

        if batch_norms is None:
            batch_norms = [False] * len(channel_sizes)
        if weight_norms is None:
            weight_norms = [False] * len(channel_sizes)
        if group_norms is None:
            group_norms = [-1] * len(channel_sizes)

        if dropouts is None:
            dropouts = [0.0] * len(channel_sizes)

        if zero_inits is None:
            zero_inits = [False] * len(channel_sizes)

        # layers
        for chan_out, kernel_size, padding, batch_norm, act, stride, weight_norm, group_norm, dropout, zero_init in \
                zip(channel_sizes, kernel_sizes, paddings, batch_norms, layer_acts, strides, weight_norms, group_norms,
                    dropouts, zero_inits, strict=True):

            module_list = nn.ModuleList()

            # norms
            chan_norm: int = chan_in if pre else chan_out
            if batch_norm:
                module_list.append(nn.BatchNorm2d(chan_norm))

            if group_norm > 0:
                module_list.append(nn.GroupNorm(group_norm, chan_norm))

            # activation
            module_list.append(get_act_fn(act))

            # dropout
            if dropout > 0.0:
                module_list.append(nn.Dropout(dropout))

            # linear
            conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d]
            if transpose:
                conv_layer = nn.ConvTranspose2d(chan_in, chan_out, kernel_size, padding=padding, stride=stride)
            else:
                conv_layer = nn.Conv2d(chan_in, chan_out, kernel_size, padding=padding, stride=stride)

            if weight_norm:
                conv_layer = parametrizations.weight_norm(conv_layer)

            # zero init
            assert not (zero_init and weight_norm), "Zero init and weight norm should not both be True"

            if zero_init:
                nn.init.zeros_(conv_layer.weight)
                if conv_layer.bias is not None:
                    nn.init.zeros_(conv_layer.bias)

            # add linear layer
            if pre:
                module_list.append(conv_layer)
            else:
                module_list.insert(0, conv_layer)

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


def make_onehots(input_dims: List[int], one_hot_depths: List[int]) -> Tuple[nn.ModuleList, int]:
    one_hots: nn.ModuleList = nn.ModuleList()
    input_dim_tot: int = 0
    for input_dim, one_hot_depth in zip(input_dims, one_hot_depths, strict=True):
        assert one_hot_depth >= 1
        one_hots.append(OneHot(one_hot_depth, True))
        input_dim_tot += input_dim * one_hot_depth

    return one_hots, input_dim_tot
