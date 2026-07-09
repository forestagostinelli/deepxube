""" Definition of heuristic and policy neural networks and parallel functions """
from abc import abstractmethod, ABC
from typing import List, Any, TypeVar, Tuple, Optional, Generic, Type

from deepxube.base.nnet_input import NNetInput, PolicyNNetIn

import torch
from torch import nn, Tensor
from torch import optim
from torch.optim import Optimizer


In = TypeVar('In', bound=NNetInput)


class DeepXubeNNet(nn.Module, Generic[In], ABC):
    """ The PyTorch module from which all modules used inherit """
    @staticmethod
    @abstractmethod
    def nnet_input_type() -> Type[In]:
        """

        :return: The type of NNetInput expected
        """
        pass

    def __init__(self, nnet_input: In):
        super().__init__()
        assert isinstance(nnet_input, self.nnet_input_type()), f"NNetInput {nnet_input} must be an instance of {self.nnet_input_type()}."
        self.nnet_input: In = nnet_input
        self.lr: float = 0.001
        self.lr_d: float = 0.9999993

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """

        :param inputs: List of tensors where the first dimension of each tensor is equal to the number of inputs
        :return: List of tensors where the first dimension of each tensor is equal to the number of outputs
        """
        if self.training:
            return self._forward_train(inputs)
        else:
            return self._forward_eval(inputs)

    def get_optimizer(self) -> Optimizer:
        """

        :return: The optimizer used to train the neural network
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def update_optimizer(self, optimizer: Optimizer, train_itr: int) -> None:
        """ Update the optimizer based on the current training iteration

        :param optimizer: Current optimizer
        :param train_itr: Training iteration
        :return: None
        """
        lr_itr: float = self.lr * (self.lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

    @abstractmethod
    def get_loss_and_info(self, fwd_tr_tensors: List[Tensor], get_info: bool) -> Tuple[Tensor, Optional[str]]:
        """ Reduce tensors to compute loss and get information about loss

        :param fwd_tr_tensors: List of tensors obtained from _forward_train
        :param get_info: if true get string info
        :return: scalar Tensor representing loss and optional string with information about loss
        """
        pass

    @abstractmethod
    def _forward_eval(self, inputs: List[Tensor]) -> List[Tensor]:
        """ Called during eval
        """
        pass

    @abstractmethod
    def _forward_train(self, inputs: List[Tensor]) -> List[Tensor]:
        """ Called during training

        :param inputs:
        :return: List of tensors for computing loss. IMPORTANT: do not perform reduction over the batch. get_loss_and_info method will do this so that
        DataParallel can be used (i.e. the first dimension of each tensor should equal the batch size)
        """
        pass


# neural networks

class HeurNNet(DeepXubeNNet[In]):
    def __init__(self, nnet_input: In, out_dim: int, q_fix: bool, **kwargs: Any):
        """

        :param nnet_input: Neural network input
        :param out_dim: Output dimensionality. If q_fix is true, this is the dimensionality of the number of actions, must be 1 otherwise
        :param q_fix: If true, the last element in the list of numpy arrays of the input corresponds to the index of the action
        :param kwargs: kwargs
        """
        super().__init__(nnet_input)
        self.out_dim: int = out_dim
        self.q_fix: bool = q_fix

    def get_loss_and_info(self, fwd_tr_tensors: List[Tensor], get_info: bool) -> Tuple[Tensor, Optional[str]]:
        assert len(fwd_tr_tensors) == 2
        ctgs_nnet: Tensor = fwd_tr_tensors[0]
        ctgs_targ: Tensor = fwd_tr_tensors[1]

        assert ctgs_nnet.size() == ctgs_targ.size()

        loss_mse: Tensor = torch.mean((ctgs_targ - ctgs_nnet) ** 2)

        info: Optional[str] = None
        if get_info:
            info = f"targ_ctg: {ctgs_targ.mean().item():.2f}, nnet_ctg: {ctgs_nnet.mean().item():.2f}"

        return loss_mse, info

    def _forward_train(self, inputs: List[Tensor]) -> List[Tensor]:
        ctgs_targ: Tensor = inputs.pop(-1)
        ctgs_nnet: Tensor = self._forward_heur(inputs)

        return [ctgs_nnet, ctgs_targ]

    def _forward_eval(self, inputs: List[Tensor]) -> List[Tensor]:
        return [self._forward_heur(inputs)]

    def _forward_heur(self, inputs: List[Tensor]) -> Tensor:
        if self.q_fix:
            action_idxs: Tensor = inputs[-1].long()
            x: Tensor = self._forward(inputs[:-1])
            return torch.gather(x, 1, action_idxs)
        else:
            return self._forward(inputs)

    @abstractmethod
    def _forward(self, inputs: List[Tensor]) -> Tensor:
        pass


PNNetIn = TypeVar('PNNetIn', bound=PolicyNNetIn)


class PolicyNNet(DeepXubeNNet[PNNetIn], ABC):
    """
    _forward_train: get states, goals, and actions
    _forward_eval: Condition on states and goals to sample self.num_samp actions
    """
    def __init__(self, nnet_input: PNNetIn, num_samp: int, **kwargs: Any):
        self.num_samp: int = num_samp
        super().__init__(nnet_input)


def _flatten_list(data_l: List[Tensor]) -> Tensor:
    return torch.cat([torch.flatten(data_i) for data_i in data_l])


class PolicyVAE(PolicyNNet[PNNetIn]):
    @staticmethod
    def _compute_recon_loss(action_proc: List[Tensor], actions_recon: List[Tensor]) -> Tensor:
        loss_recons: List[Tensor] = []
        for actions_proc_i, actions_recon_i in zip(action_proc, actions_recon):
            mean_dims: Tuple[int, ...] = tuple(range(1, len(actions_proc_i.shape)))
            loss_recon_i: Tensor = torch.mean((actions_recon_i - actions_proc_i) ** 2, dim=mean_dims)
            loss_recons.append(loss_recon_i)

        return torch.stack(loss_recons, dim=0).mean(dim=0)

    def __init__(self, nnet_input: PNNetIn, num_samp: int, kl_weight: float, **kwargs: Any):
        super().__init__(nnet_input, num_samp)
        self.norm_dist = torch.distributions.Normal(0, 1)
        self.kl_weight: float = kl_weight

    def get_loss_and_info(self, fwd_tr_tensors: List[Tensor], get_info: bool) -> Tuple[Tensor, Optional[str]]:
        loss_recon_mean: Tensor = torch.mean(fwd_tr_tensors[0], dim=0)
        loss_kl_mean: Tensor = torch.mean(fwd_tr_tensors[1], dim=0)

        loss: Tensor = loss_recon_mean + (self.kl_weight * loss_kl_mean)

        loss_str: Optional[str] = None
        if get_info:
            loss_str = f"loss_recon: {loss_recon_mean.item():.2E}, loss_kl: {loss_kl_mean.item():.2E}"

        return loss, loss_str

    def _forward_eval(self, states_goals: List[Tensor]) -> List[Tensor]:
        recons_l: List[List[Tensor]] = []
        z_l: List[Tensor] = []
        for _ in range(self.num_samp):
            z: Tensor = self.norm_dist.sample((states_goals[0].shape[0],) + self.latent_shape()).to(states_goals[0].device)
            recons: List[Tensor] = self.decode(states_goals, z)
            recons_l.append(recons)
            z_l.append(z)

        recons_all: List[Tensor] = []
        for recon_idx in range(len(recons_l[0])):
            recons_i: Tensor = torch.stack([recons_i[recon_idx] for recons_i in recons_l], dim=1)
            recons_all.append(recons_i)

        z_all: Tensor = torch.stack(z_l, dim=1)
        return recons_all + [self.norm_dist.log_prob(z_all).sum(dim=2)]

    def _forward_train(self, states_goals_actions: List[Tensor]) -> List[Tensor]:
        split_idx: int = self.nnet_input.states_goals_actions_split_idx()
        states_goals: List[Tensor] = states_goals_actions[:split_idx]
        actions: List[Tensor] = states_goals_actions[split_idx:]

        actions_proc, mu, logvar = self.encode(states_goals, actions)
        sum_dims: Tuple[int, ...] = tuple(range(1, len(mu.shape)))
        loss_kl: Tensor = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=sum_dims)

        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.norm_dist.sample(mu.shape).to(mu.device)
        actions_recon: List[Tensor] = self.decode(states_goals, z)

        loss_recon: Tensor = self._compute_recon_loss(actions_proc, actions_recon)

        return [loss_recon, loss_kl]

    @abstractmethod
    def latent_shape(self) -> Tuple[int, ...]:
        """

        :return: Dimensions of latent
        """
        pass

    @abstractmethod
    def encode(self, states_goals: List[Tensor], actions: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        """ Conditon on states and goals and map actions to mu and logvar

        :param states_goals:
        :param actions:
        :return: processed input actions, mu, and logvar
        """

    @abstractmethod
    def decode(self, states_goals: List[Tensor], z: Tensor) -> List[Tensor]:
        """ Conditon on states and goals and map sampled latent to reconstructed actions

        :param states_goals:
        :param z: Latent state
        :return:
        """

    def __repr__(self) -> str:
        return f"{super().__repr__()}\nKL Weight: {self.kl_weight}"
