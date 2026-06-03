from typing import List, Type, Tuple
import torch
from torch import nn, Tensor

from deepxube.base.factory import DelimParser
from deepxube.base.nnet_input import FlatIn, FlatInPolicy
from deepxube.base.heuristic import HeurNNet, PolicyVAE
from deepxube.nnet.pytorch_models import FullyConnectedModel, ResnetModel, OneHot, make_onehots

from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.heuristic_factory import policy_factory


@heuristic_factory.register_class("resnet_fc")
class ResnetFCHeur(HeurNNet[FlatIn]):
    @staticmethod
    def nnet_input_type() -> Type[FlatIn]:
        return FlatIn

    def __init__(self, nnet_input: FlatIn, out_dim: int, q_fix: bool, res_dim: int = 1000, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = "RELU"):
        super().__init__(nnet_input, out_dim, q_fix)
        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        input_dim_tot: int = 0
        input_dims, one_hot_depths = self.nnet_input.get_input_info()
        for input_dim, one_hot_depth in zip(input_dims, one_hot_depths, strict=True):
            assert one_hot_depth >= 1
            self.one_hots.append(OneHot(one_hot_depth, True))
            input_dim_tot += input_dim * one_hot_depth

        # res net
        self.res_dim: int = res_dim

        group_norm: int = -1
        if layer_norm:
            group_norm = 1

        def res_block_init() -> nn.Module:
            return FullyConnectedModel(res_dim, [res_dim] * 2, [act_fn, "LINEAR"],
                                       batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2,
                                       group_norms=[group_norm] * 2)

        self.heur = nn.Sequential(
            nn.Linear(input_dim_tot, res_dim),
            ResnetModel(res_block_init, num_blocks, act_fn),
            nn.Linear(res_dim, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        inputs_oh: List[Tensor] = [one_hot(input_i) for input_i, one_hot in zip(inputs, self.one_hots)]
        x: Tensor = self.heur(torch.cat(inputs_oh, dim=1))
        return x


@policy_factory.register_class("resnet_fc")
class ResnetFCPolicy(PolicyVAE[FlatInPolicy]):
    @staticmethod
    def nnet_input_type() -> Type[FlatInPolicy]:
        return FlatInPolicy

    def __init__(self, nnet_input: FlatInPolicy, num_samp: int, kl_weight: float, enc_dim: int = 10, res_dim: int = 1000, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = "RELU"):
        super().__init__(nnet_input, num_samp, kl_weight)
        # one hots
        input_dims, one_hot_depths = self.nnet_input.get_input_info()
        input_dims_sg: List[int] = input_dims[:self.nnet_input.states_goals_actions_split_idx()]
        one_hot_depths_sg: List[int] = one_hot_depths[:self.nnet_input.states_goals_actions_split_idx()]
        input_dims_acts: List[int] = input_dims[self.nnet_input.states_goals_actions_split_idx():]
        one_hot_depths_acts: List[int] = one_hot_depths[self.nnet_input.states_goals_actions_split_idx():]

        self.one_hots_sg, input_dim_sg = make_onehots(input_dims_sg, one_hot_depths_sg)
        self.one_hots_acts, input_dim_acts = make_onehots(input_dims_acts, one_hot_depths_acts)
        input_dim_tot: int = input_dim_sg + input_dim_acts

        self.enc_dim: int = enc_dim

        # res net
        self.res_dim: int = res_dim

        group_norm: int = -1
        if layer_norm:
            group_norm = 1

        def res_block_init() -> nn.Module:
            return FullyConnectedModel(res_dim, [res_dim] * 2, [act_fn, "LINEAR"],
                                       batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2,
                                       group_norms=[group_norm] * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim_tot, res_dim),
            ResnetModel(res_block_init, num_blocks, act_fn),
            nn.Linear(res_dim, self.enc_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim_sg + self.enc_dim, res_dim),
            ResnetModel(res_block_init, num_blocks, act_fn),
            nn.Linear(res_dim, input_dim_acts)
        )

    def latent_shape(self) -> Tuple[int, ...]:
        return (self.enc_dim,)

    def encode(self, states_goals: List[Tensor], actions: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        states_goals_oh: List[Tensor] = [one_hot(input_i) for input_i, one_hot in zip(states_goals, self.one_hots_sg)]
        actions_oh: List[Tensor] = [one_hot(input_i) for input_i, one_hot in zip(actions, self.one_hots_acts)]
        mu_logvar: Tensor = self.encoder(torch.cat(states_goals_oh + actions_oh, dim=1))

        return actions_oh, mu_logvar[:, :self.enc_dim], mu_logvar[:, self.enc_dim:]

    def decode(self, states_goals: List[Tensor], z: Tensor) -> List[Tensor]:
        states_goals_oh: List[Tensor] = [one_hot(input_i) for input_i, one_hot in zip(states_goals, self.one_hots_sg)]
        x: Tensor = self.decoder(torch.cat(states_goals_oh + [z], dim=1))
        return [x]


class ResnetFCParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("H", "res_dim", int, "dimensionality of hidden layers in residual blocks")
        self.add_argument("B", "num_blocks", int, "number of residual blocks")
        self.add_argument("bn", "batch_norm", None, "Batch normalization")
        self.add_argument("wn", "weight_norm", None, "Weight normalization")
        self.add_argument("ln", "layer_norm", None, "Layer normalization")

    @property
    def delim(self) -> str:
        return "_"


@heuristic_factory.register_parser("resnet_fc")
class ResnetFCParserHeur(ResnetFCParser):
    pass


@policy_factory.register_parser("resnet_fc")
class ResnetFCParserPolicy(ResnetFCParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("ln", "layer_norm", None, "Layer normalization")
        self.add_argument("E", "enc_dim", int, "Dimensionality of encoding layer")
        self.add_argument("KL", "kl_weight", int, "KL divergence penalty", default=1.0)
