from typing import List, Dict, Any, Type
import torch
from torch import nn, Tensor

from deepxube.base.factory import DelimParser
from deepxube.base.nnet_input import TwoDIn
from deepxube.base.nnet import HeurNNet
from deepxube.pytorch.pytorch_models import Conv2dModel, ResnetModel, OneHot

from deepxube.factories.nnet_factory import deepxube_nnet_factory


@deepxube_nnet_factory.register_class("resnet_2d")
class Resnet2D(HeurNNet[TwoDIn]):
    @staticmethod
    def nnet_input_type() -> Type[TwoDIn]:
        return TwoDIn

    def __init__(self, nnet_input: TwoDIn, out_dim: int, q_fix: bool, num_chan: int = 64, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, group_norm: int = -1, act_fn: str = "RELU"):
        super().__init__(nnet_input, out_dim, q_fix)

        chan_dims, (height, width), one_hot_depths, q_fix_1x1 = self.nnet_input.get_input_info()

        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        chan_in_tot: int = 0
        for chan_dim, one_hot_depth in zip(chan_dims, one_hot_depths, strict=True):
            assert one_hot_depth >= 1
            self.one_hots.append(OneHot(one_hot_depth, False))
            chan_in_tot += chan_dim * one_hot_depth

        # res net
        def res_block_init() -> nn.Module:
            return Conv2dModel(num_chan, [num_chan] * 2, [3] * 2, [1] * 2, [act_fn, "LINEAR"],
                               batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2, group_norms=[group_norm] * 2)

        self.heur = nn.Sequential(
            Conv2dModel(chan_in_tot, [num_chan], [1], [0], ["LINEAR"]),
            ResnetModel(res_block_init, num_blocks, act_fn),
        )

        if self.q_fix and (q_fix_1x1 is not None):
            assert (height * width * q_fix_1x1) == out_dim
            self.out = nn.Sequential(
                Conv2dModel(num_chan, [q_fix_1x1], [1], [0], ["LINEAR"]),
                nn.Flatten(),
            )
        else:
            self.out = nn.Sequential(
                Conv2dModel(num_chan, [1], [1], [0], ["LINEAR"]),
                nn.Flatten(),
                nn.Linear(height * width, out_dim)
            )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        inputs_oh: List[Tensor] = []
        for input_i, one_hot in zip(inputs, self.one_hots):
            input_i_oh: Tensor = one_hot(input_i)
            if len(input_i_oh.shape) == 5:
                input_i_oh = input_i_oh.permute((0, 1, 4, 2, 3)).flatten(1, 2)
            inputs_oh.append(input_i_oh)

        # inputs_oh: List[Tensor] = [one_hot(input_i).permute((0, 1, 4, 2, 3)).flatten(1, 2) for input_i, one_hot in zip(inputs, self.one_hots)]
        x: Tensor = self.heur(torch.cat(inputs_oh, dim=1))
        x = self.out(x)
        return x


@deepxube_nnet_factory.register_parser("resnet_2d")
class Resnet2DParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("C", "num_chan", int, "Number of convolutional channels")
        self.add_argument("B", "num_blocks", int, "number of residual blocks")
        self.add_argument("bn", "batch_norm", None, "Batch normalization")
        self.add_argument("wn", "weight_norm", None, "Weight normalization")
        self.add_argument("gn", "group_norm", int, "Number of groups")

    @property
    def delim(self) -> str:
        return "_"
