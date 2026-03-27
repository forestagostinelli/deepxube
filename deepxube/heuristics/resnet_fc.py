from typing import List, Dict, Any, Type, Tuple
import torch
from torch import nn, Tensor
import re

from deepxube.base.factory import Parser
from deepxube.base.nnet_input import FlatIn, FlatInPolicy
from deepxube.base.heuristic import HeurNNet, PolicyVAE
from deepxube.nnet.pytorch_models import FullyConnectedModel, ResnetModel, OneHot

from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.heuristic_factory import policy_factory


@heuristic_factory.register_class("resnet_fc")
class ResnetFCHeur(HeurNNet[FlatIn]):
    @staticmethod
    def nnet_input_type() -> Type[FlatIn]:
        return FlatIn

    def __init__(self, nnet_input: FlatIn, out_dim: int, q_fix: bool, res_dim: int = 1000, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, group_norm: int = -1, act_fn: str = "RELU"):
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


def _make_onehots(input_dims: List[int], one_hot_depths: List[int]) -> Tuple[nn.ModuleList, int]:
    one_hots: nn.ModuleList = nn.ModuleList()
    input_dim_tot: int = 0
    for input_dim, one_hot_depth in zip(input_dims, one_hot_depths, strict=True):
        assert one_hot_depth >= 1
        one_hots.append(OneHot(one_hot_depth, True))
        input_dim_tot += input_dim * one_hot_depth

    return one_hots, input_dim_tot


@policy_factory.register_class("resnet_fc")
class ResnetFCPolicy(PolicyVAE[FlatInPolicy]):
    @staticmethod
    def nnet_input_type() -> Type[FlatInPolicy]:
        return FlatInPolicy

    def __init__(self, nnet_input: FlatInPolicy, kl_weight: float, enc_dim: int = 10, res_dim: int = 1000, num_blocks: int = 4, batch_norm: bool = False,
                 weight_norm: bool = False, group_norm: int = -1, act_fn: str = "RELU"):
        super().__init__(nnet_input, kl_weight)
        # one hots
        input_dims, one_hot_depths = self.nnet_input.get_input_info()
        input_dims_sg: List[int] = input_dims[:self.nnet_input.states_goals_actions_split_idx()]
        one_hot_depths_sg: List[int] = one_hot_depths[:self.nnet_input.states_goals_actions_split_idx()]
        input_dims_acts: List[int] = input_dims[self.nnet_input.states_goals_actions_split_idx():]
        one_hot_depths_acts: List[int] = one_hot_depths[self.nnet_input.states_goals_actions_split_idx():]

        self.one_hots_sg, input_dim_sg = _make_onehots(input_dims_sg, one_hot_depths_sg)
        self.one_hots_acts, input_dim_acts = _make_onehots(input_dims_acts, one_hot_depths_acts)
        input_dim_tot: int = input_dim_sg + input_dim_acts

        self.enc_dim: int = enc_dim

        # res net
        self.res_dim: int = res_dim

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


@heuristic_factory.register_parser("resnet_fc")
class ResnetFCParserHeur(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            hidden_re = re.search(r"^(\S+)H$", args_str_i)
            blocks_re = re.search(r"^(\S+)B$", args_str_i)
            bn_re = re.search(r"^bn$", args_str_i)
            wn_re = re.search(r"^wn$", args_str_i)
            if hidden_re is not None:
                kwargs["res_dim"] = int(hidden_re.group(1))
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
        return ("Arguments are delimited by '_' and can be in any order.\n<num>H (number of hidden units), "
                "<num>B (number of blocks), bn (batch_norm), wn (weight_norm).\n"
                "E.g. resnet_fc.1000H_4B_bn")


@policy_factory.register_parser("resnet_fc")
class ResnetFCParserPolicy(ResnetFCParserHeur):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        kwargs["kl_weight"] = 1.0
        for args_str_i in args_str_l:
            hidden_re = re.search(r"^(\S+)H$", args_str_i)
            blocks_re = re.search(r"^(\S+)B$", args_str_i)
            enc_dim_re = re.search(r"^(\S+)E$", args_str_i)
            kl_re = re.search(r"^(\S+)KL$", args_str_i)
            bn_re = re.search(r"^bn$", args_str_i)
            wn_re = re.search(r"^wn$", args_str_i)
            if hidden_re is not None:
                kwargs["res_dim"] = int(hidden_re.group(1))
            elif blocks_re is not None:
                kwargs["num_blocks"] = int(blocks_re.group(1))
            elif bn_re is not None:
                kwargs["batch_norm"] = True
            elif wn_re is not None:
                kwargs["weight_norm"] = True
            elif enc_dim_re is not None:
                kwargs["enc_dim"] = int(enc_dim_re.group(1))
            elif kl_re is not None:
                kwargs["kl_weight"] = float(kl_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("Arguments are delimited by '_' and can be in any order.\n<num>H (number of hidden units), "
                "<num>B (number of blocks), <enc_dim>E (encoding dimensionality), bn (batch_norm), wn (weight_norm).\n"
                "E.g. resnet_fc.1000H_4B_10E_bn")
