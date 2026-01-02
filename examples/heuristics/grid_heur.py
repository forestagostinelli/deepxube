from typing import List, Type, Dict, Any
from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.heuristic import HeurNNet
from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel
from deepxube.factories.heuristic_factory import heuristic_factory

from domains.grid import GridNNetInput
import re


@heuristic_factory.register_class("gridnet")
class GridNet(HeurNNet[GridNNetInput]):
    @staticmethod
    def nnet_input_type() -> Type[GridNNetInput]:
        return GridNNetInput

    def __init__(self, nnet_input: GridNNetInput, out_dim: int, q_fix: bool, chan_size: int = 8, fc_size: int = 100):
        super().__init__(nnet_input, out_dim, q_fix)
        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        grid_dim: int = self.nnet_input.get_input_info()

        self.heur: nn.Module = nn.Sequential(
            Conv2dModel(2, [chan_size, chan_size], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(grid_dim * grid_dim * chan_size, [fc_size], ["RELU"], batch_norms=[True]),
            nn.Linear(fc_size, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        return self.heur(inputs[0])


@heuristic_factory.register_parser("gridnet")
class GridNetParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            channel_re = re.search(r"^(\S+)CH$", args_str_i)
            fc_re = re.search(r"^(\S+)FC$", args_str_i)
            if channel_re is not None:
                kwargs["chan_size"] = int(channel_re.group(1))
            elif fc_re is not None:
                kwargs["fc_size"] = int(fc_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("Arguments are delimited by '_' and can be in any order.\n<num>C (number of channels), "
                "<num>FC (width of fully-connected layer), bn (batch_norm), wn (weight_norm).\n"
                "E.g. gridnet.10CH_200FC")
