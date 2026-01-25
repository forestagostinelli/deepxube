from typing import List, Dict, Any, Type
import torch
from torch import nn, Tensor
import re

from deepxube.base.factory import Parser
from deepxube.base.nnet_input import TwoDIn
from deepxube.base.heuristic import HeurNNet
from deepxube.nnet.pytorch_models import Conv2dModel, ResnetModel, OneHot

from deepxube.factories.heuristic_factory import heuristic_factory


@heuristic_factory.register_class("resnet_2d")
class Resnet2D(HeurNNet[TwoDIn]):
    @staticmethod
    def nnet_input_type() -> Type[TwoDIn]:
        return TwoDIn

    def __init__(self, nnet_input: TwoDIn, out_dim: int, q_fix: bool, num_chan: int = 64, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, act_fn: str = "RELU"):
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
                               batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2)

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
        inputs_oh: List[Tensor] = [one_hot(input_i).permute((0, 1, 4, 2, 3)).flatten(1, 2) for input_i, one_hot in zip(inputs, self.one_hots)]
        x: Tensor = self.heur(torch.cat(inputs_oh, dim=1))
        x = self.out(x)
        return x


@heuristic_factory.register_parser("resnet_2d")
class ResnetFCParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            chan_re = re.search(r"^(\S+)C$", args_str_i)
            blocks_re = re.search(r"^(\S+)B$", args_str_i)
            bn_re = re.search(r"^bn$", args_str_i)
            wn_re = re.search(r"^wn$", args_str_i)
            if chan_re is not None:
                kwargs["num_chan"] = int(chan_re.group(1))
            elif blocks_re is not None:
                kwargs["num_blocks"] = int(blocks_re.group(1))
            elif bn_re is not None:
                kwargs["batch_norm"] = True
            elif wn_re is not None:
                kwargs["weight_norm"] = True
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("Arguments are delimited by '_' and can be in any order.\n<num>C (number of channels), "
                "<num>B (number of blocks), bn (batch_norm), wn (weight_norm).\n"
                "E.g. resnet_2d.64C_4B_bn")
