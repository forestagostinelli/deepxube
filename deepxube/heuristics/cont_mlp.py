from typing import List, Dict, Any, Type, cast

import torch
from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.nnet_input import FlatIn
from deepxube.base.heuristic import HeurNNet
from deepxube.factories.heuristic_factory import heuristic_factory


@heuristic_factory.register_class("cont_mlp")
class ContMLP(HeurNNet[FlatIn]):
    """Lightweight MLP for continuous state/goal/action vectors (flattened numeric inputs)."""

    @staticmethod
    def nnet_input_type() -> Type[FlatIn]:
        return FlatIn

    def __init__(self, nnet_input: FlatIn, out_dim: int, q_fix: bool, hidden: int = 64):
        super().__init__(nnet_input, out_dim, q_fix)
        input_dims, _ = self.nnet_input.get_input_info()
        feat_dim: int = sum(input_dims)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        x = torch.cat(inputs, dim=1)
        return cast(Tensor, self.net(x))


@heuristic_factory.register_parser("cont_mlp")
class ContMLPParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"hidden": int(args_str)} if len(args_str) > 0 else {}

    def help(self) -> str:
        return "Optional hidden layer width (int). Example: 'cont_mlp.128'"
