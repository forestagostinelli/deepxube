from typing import List, Type
import torch
from torch import nn, Tensor

from domains.grid import GridNNetInput

from deepxube.base.heuristic import HeurNNet
from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel, ResnetModel
from deepxube.factories.heuristic_factory import register_heur_nnet


@register_heur_nnet("gridnet")
class GridNet(HeurNNet[GridNNetInput]):
    @staticmethod
    def nnet_input_type() -> Type[GridNNetInput]:
        return GridNNetInput

    def __init__(self, nnet_input: GridNNetInput, out_dim: int, q_fix: bool):
        super().__init__(nnet_input, out_dim, q_fix)
        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        grid_dim: int = self.nnet_input.get_input_info()

        self.heur = nn.Sequential(
            Conv2dModel(2, [4, 4], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(grid_dim * grid_dim * 2, [100], ["RELU"], batch_norms=[True]),
            nn.Linear(100, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        breakpoint()
        inputs_oh: List[Tensor] = [one_hot(input_i) for input_i, one_hot in zip(inputs, self.one_hots)]
        x: Tensor = self.heur(torch.cat(inputs_oh, dim=1))
        return x
