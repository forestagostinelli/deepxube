from abc import abstractmethod, ABC
from typing import List, Any, TypeVar, Generic, cast, Tuple, Optional, Type, Protocol, runtime_checkable, Union

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Goal, Action
from deepxube.base.nnet_input import NNetInput, PolicyNNetIn
from deepxube.nnet.nnet_utils import NNetParInfo, nnet_batched, NNetPar, get_nnet_par_out
from deepxube.utils import misc_utils

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.optim.optimizer import Optimizer


In = TypeVar('In', bound=NNetInput)


class DeepXubeNNet(nn.Module, Generic[In], ABC):
    @staticmethod
    @abstractmethod
    def nnet_input_type() -> Type[In]:
        pass

    def __init__(self, nnet_input: In):
        super().__init__()
        assert isinstance(nnet_input, self.nnet_input_type()), f"NNetInput {nnet_input} must be an instance of {self.nnet_input_type()}."
        self.nnet_input: In = nnet_input
        self.lr: float = 0.001
        self.lr_d: float = 0.9999993

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        if self.training:
            return self._forward_train(inputs)
        else:
            return self._forward_eval(inputs)

    def get_optimizer(self) -> Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr)

    def update_optimizer(self, optimizer: Optimizer, train_itr: int) -> None:
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
        super().__init__(nnet_input)
        self.num_samp: int = num_samp


def _flatten_list(data_l: List[Tensor]) -> Tensor:
    return torch.cat([torch.flatten(data_i) for data_i in data_l])


class PolicyVAE(PolicyNNet[PNNetIn]):
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

    def _compute_recon_loss(self, action_proc: List[Tensor], actions_recon: List[Tensor]) -> Tensor:
        loss_recons: List[Tensor] = []
        for actions_proc_i, actions_recon_i in zip(action_proc, actions_recon):
            mean_dims: Tuple[int, ...] = tuple(range(1, len(actions_proc_i.shape)))
            loss_recon_i: Tensor = torch.mean((actions_recon_i - actions_proc_i) ** 2, dim=mean_dims)
            loss_recons.append(loss_recon_i)

        return torch.stack(loss_recons, dim=0).mean(dim=0)

    def __repr__(self) -> str:
        return f"{super().__repr__()}\nKL Weight: {self.kl_weight}"


# functions

@runtime_checkable
class HeurFnV(Protocol):
    def __call__(self, states: List[State], goals: List[Goal]) -> List[float]:
        ...


@runtime_checkable
class HeurFnQ(Protocol):
    def __call__(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        ...


HeurFn = Union[HeurFnV, HeurFnQ]


@runtime_checkable
class PolicyFn(Protocol):
    def __call__(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
        """ Map states and goals to sampled actions along with their probability (or log probability) densities

        """
        ...


# parallelizable functions

H = TypeVar('H', bound=HeurFn)


class HeurNNetPar(NNetPar[H]):
    @abstractmethod
    def get_nnet(self) -> HeurNNet:
        pass

    @abstractmethod
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device, update_num: Optional[int]) -> H:
        pass

    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> H:
        pass


class HeurNNetParV(HeurNNetPar[HeurFnV]):
    @staticmethod
    def _get_output(heurs: NDArray[np.float64], update_num: Optional[int]) -> List[float]:
        heurs = np.maximum(heurs[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            heurs = heurs * 0
        return cast(List[float], heurs.astype(np.float64).tolist())

    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnV:
        nnet.eval()

        def heuristic_fn(states: List[State], goals: List[Goal]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_np(states, goals)
            heurs: NDArray[np.float64] = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]

            return self._get_output(heurs, update_num)
        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnV:
        def heuristic_fn(states: List[State], goals: List[Goal]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_np(states, goals)
            heurs: NDArray[np.float64] = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]

            return self._get_output(heurs, update_num)

        return heuristic_fn

    @abstractmethod
    def to_np(self, states: List[State], goals: List[Goal]) -> List[NDArray[Any]]:
        pass


class HeurNNetParQ(HeurNNetPar[HeurFnQ]):
    @abstractmethod
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device, update_num: Optional[int]) -> HeurFnQ:
        pass

    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        pass

    @abstractmethod
    def to_np(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray[Any]]:
        pass


class HeurNNetParQFixOut(HeurNNetParQ, ABC):
    """ DQN with a fixed output shape

    """
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device, update_num: Optional[int]) -> HeurFnQ:
        nnet.eval()

        def heuristic_fn(states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
            inputs_nnet: List[NDArray] = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray[np.float64] = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]
            return self._get_output(states, q_vals_np, update_num)

        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        def heuristic_fn(states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
            inputs_nnet: List[NDArray] = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray[np.float64] = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]
            return self._get_output(states, q_vals_np, update_num)

        return heuristic_fn

    def to_np(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray[Any]]:
        self._check_same_num_acts(actions_l)
        return self._to_np_fixed_acts(states, goals, actions_l)

    @abstractmethod
    def _to_np_fixed_acts(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray[Any]]:
        pass

    def _get_input(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray]:
        inputs_nnet: List[NDArray] = self.to_np(states, goals, actions_l)
        return inputs_nnet

    @staticmethod
    def _get_output(states: List[State], q_vals_np: NDArray[np.float64], update_num: Optional[int]) -> List[List[float]]:
        assert q_vals_np.shape[0] == len(states)
        q_vals_np = np.maximum(q_vals_np, 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0
        q_vals_l: List[List[float]] = [q_vals_np[state_idx].astype(np.float64).tolist() for state_idx in
                                       range(len(states))]
        return q_vals_l

    @staticmethod
    def _check_same_num_acts(actions_l: List[List[Action]]) -> None:
        assert len(set(len(actions) for actions in actions_l)) == 1, "num actions should be the same for all instances"


class HeurNNetParQIn(HeurNNetParQ, ABC):
    """ DQN that takes a single action as input

    """
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnQ:
        nnet.eval()

        def heuristic_fn(states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
            inputs_nnet, states_rep, split_idxs = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]
            return self._get_output(states_rep, q_vals_np, split_idxs, update_num)

        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        def heuristic_fn(states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
            inputs_nnet, states_rep, split_idxs = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]
            return self._get_output(states_rep, q_vals_np, split_idxs, update_num)

        return heuristic_fn

    def to_np(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray[Any]]:
        assert all((len(actions) == 1) for actions in actions_l), "there should only be one action per state/goal pair"
        actions_one: List[Action] = [actions[0] for actions in actions_l]
        return self._to_np_one_act(states, goals, actions_one)

    @abstractmethod
    def _to_np_one_act(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray[Any]]:
        pass

    def _get_input(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> Tuple[List[NDArray], List[State], List[int]]:
        actions_flat, split_idxs = misc_utils.flatten(actions_l)
        states_rep: List[State] = []
        goals_rep: List[Goal] = []
        for state, goal, actions in zip(states, goals, actions_l, strict=True):
            states_rep.extend([state] * len(actions))
            goals_rep.extend([goal] * len(actions))
        inputs_nnet: List[NDArray] = self._to_np_one_act(states_rep, goals_rep, actions_flat)

        return inputs_nnet, states_rep, split_idxs

    @staticmethod
    def _get_output(states_rep: List[State], q_vals_np: NDArray[np.float64], split_idxs: List[int], update_num: Optional[int]) -> List[List[float]]:
        assert q_vals_np.shape[0] == len(states_rep)
        q_vals_np = np.maximum(q_vals_np[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0

        q_vals_flat: List[float] = q_vals_np.astype(np.float64).tolist()
        q_vals_l: List[List[float]] = misc_utils.unflatten(q_vals_flat, split_idxs)
        return q_vals_l


class PolicyNNetPar(NNetPar[PolicyFn]):
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device, update_num: Optional[int]) -> PolicyFn:
        nnet.eval()

        def policy_fn(states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
            inputs_nnet: List[NDArray] = self.to_np_fn(states, goals)
            nnet_out_np: List[NDArray[np.float64]] = nnet_batched(nnet, inputs_nnet, batch_size, device)

            return self._np_to_acts_and_pdfs(nnet_out_np[0:-1], nnet_out_np[-1], len(states))

        return policy_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> PolicyFn:
        def policy_fn(states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
            inputs_nnet: List[NDArray] = self.to_np_fn(states, goals)
            nnet_out_np: List[NDArray[np.float64]] = get_nnet_par_out(inputs_nnet, nnet_par_info)

            return self._np_to_acts_and_pdfs(nnet_out_np[0:-1], nnet_out_np[-1], len(states))

        return policy_fn

    @abstractmethod
    def get_nnet(self) -> PolicyNNet:
        pass

    @abstractmethod
    def to_np_fn(self, states: List[State], goals: List[Goal]) -> List[NDArray[Any]]:
        pass

    @abstractmethod
    def to_np_train(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray[Any]]:
        pass

    @abstractmethod
    def _nnet_out_to_actions(self, nnet_out: List[NDArray[np.float64]]) -> List[Action]:
        pass

    def _np_to_acts_and_pdfs(self, actions_np: List[NDArray[np.float64]], pdfs_np: NDArray[np.float64],
                             num_states: int) -> Tuple[List[List[Action]], List[List[float]]]:
        # assert dimensions match
        assert len(pdfs_np.shape) == 2
        assert pdfs_np.shape[0] == num_states
        for actions_np_i in actions_np:
            assert actions_np_i.shape[0] == num_states
            assert actions_np_i.shape[0] == pdfs_np.shape[0]
            assert actions_np_i.shape[1] == pdfs_np.shape[1]

        # convert to action object rep
        actions_l: List[List[Action]] = []
        pdfs_l: List[List[float]] = []
        for state_idx in range(num_states):
            actions_np_state: List[NDArray[np.float64]] = [actions_np_i[state_idx] for actions_np_i in actions_np]
            pdfs_state: List[float] = pdfs_np[state_idx, :].tolist()

            actions_l.append(self._nnet_out_to_actions(actions_np_state))
            pdfs_l.append(pdfs_state)

        return actions_l, pdfs_l

    def __repr__(self) -> str:
        nnet: PolicyNNet = self.get_nnet()
        return f"{nnet}\n#Samp: {nnet.num_samp}"
