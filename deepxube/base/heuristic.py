from abc import abstractmethod, ABC
from typing import List, Any, TypeVar, Generic, cast, Tuple, Optional, Type, Protocol, runtime_checkable, Union

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal, Action
from deepxube.base.nnet_input import NNetInput
from deepxube.nnet.nnet_utils import NNetParInfo, nnet_batched, NNetPar, get_nnet_par_out
from deepxube.utils import misc_utils
import torch
from torch import nn, Tensor
import random


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

    @abstractmethod
    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        pass


# neural networks

class HeurNNet(DeepXubeNNet[In]):
    def __init__(self, nnet_input: In, out_dim: int, q_fix: bool, **kwargs: Any):
        super().__init__(nnet_input)
        self.out_dim: int = out_dim
        self.q_fix: bool = q_fix

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        if self.q_fix:
            action_idxs: Tensor = inputs[-1].long()
            x: Tensor = self._forward(inputs[:-1])
            return [torch.gather(x, 1, action_idxs)]
        else:
            return [self._forward(inputs)]

    @abstractmethod
    def _forward(self, inputs: List[Tensor]) -> Tensor:
        pass


class PolicyNNet(DeepXubeNNet[In], ABC):
    def __init__(self, nnet_input: In, **kwargs: Any):
        super().__init__(nnet_input)
        self.norm_dist = torch.distributions.Normal(0, 1)
        self.criterion_recon = nn.MSELoss()

    def forward(self, states_goals: List[Tensor]) -> List[Tensor]:
        """ Condition on states and goals to sample actions

        :param states_goals:
        :return:
        """
        z: Tensor = self.norm_dist.sample((states_goals[0].shape[0],) + self.latent_shape()).to(states_goals[0].device)
        recons: Tensor = self.decode(states_goals, z)
        return [recons, self.norm_dist.log_prob(z)]

    def autoencode(self, states_goals: List[Tensor], actions: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param states_goals:
        :param actions:
        :return: reconstruction loss, kl loss
        """
        mu, logvar = self.encode(states_goals, actions)
        sum_dims: Tuple[int, ...] = tuple(range(1, len(mu.shape)))
        loss_kl: Tensor = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=sum_dims), dim=0)

        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.norm_dist.sample(mu.shape).to(mu.device)
        recons: Tensor = self.decode(states_goals, z)

        loss_recon: Tensor = self.criterion_recon(recons, actions)

        return loss_recon, loss_kl

    @abstractmethod
    def latent_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def encode(self, states_goals: List[Tensor], actions: Tensor) -> Tuple[Tensor, Tensor]:
        """ Conditon on states and goals and map actions to mu and logvar

        :param states_goals:
        :param actions:
        :return: mu and logvar
        """

    @abstractmethod
    def decode(self, states_goals: List[Tensor], z: Tensor) -> Tensor:
        """ Conditon on states and goals and map sampled latent to reconstructed actions

        :param states_goals:
        :param z:
        :return:
        """


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
    def __call__(self, domain: Domain, states: List[State], goals: List[Goal], num_samp: int, num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
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


def policy_fn_rand(domain: Domain, states: List[State], num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
    states_rep: List[List[State]] = []
    for state in states:
        states_rep.append([state] * num_rand)

    states_rep_flat, split_idxs = misc_utils.flatten(states_rep)

    actions_samp_flat: List[Action] = domain.sample_state_action(states_rep_flat)
    actions_samp_l: List[List[Action]] = misc_utils.unflatten(actions_samp_flat, split_idxs)

    probs_l: List[List[float]] = []
    for actions_samp_i in actions_samp_l:
        probs_l.append([1.0 / len(actions_samp_i)] * len(actions_samp_i))

    return actions_samp_l, probs_l


def _combine_nnet_with_rand(domain: Domain, actions_l: List[List[Action]], pdfs_l: List[List[float]], states: List[State],
                            num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
    actions_rand_l: List[List[Action]] = policy_fn_rand(domain, states, num_rand)[0]

    actions_comb_l: List[List[Action]] = []
    pdfs_comb_l: List[List[float]] = []
    # get nnet actions
    for state_idx in range(len(states)):
        actions_comb_l.append(actions_l[state_idx] + actions_rand_l[state_idx])
        pdfs_i: List[float] = pdfs_l[state_idx]
        pdfs_comb_l.append(pdfs_i + random.choices(pdfs_i, k=len(pdfs_i)))

    return actions_comb_l, pdfs_comb_l


class PolicyNNetPar(NNetPar[PolicyFn]):
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device, update_num: Optional[int]) -> PolicyFn:
        nnet.eval()
        if (update_num is not None) and (update_num == 0):
            def policy_fn(domain: Domain, states: List[State], goals: List[Goal], num_samp: int, num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
                assert len(states) == len(goals)  # to stop PyCharm from complaining
                return policy_fn_rand(domain, states, num_samp + num_rand)
        else:
            def policy_fn(domain: Domain, states: List[State], goals: List[Goal], num_samp: int, num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
                inputs_nnet_rep: List[NDArray] = self._get_nnet_inputs_rep(states, goals, num_samp)
                nnet_out_np: List[NDArray[np.float64]] = nnet_batched(nnet, inputs_nnet_rep, batch_size, device)

                actions_l, pdfs_l = self._np_to_acts_and_pdfs(nnet_out_np[0], nnet_out_np[1], len(states), num_samp)
                return _combine_nnet_with_rand(domain, actions_l, pdfs_l, states, num_rand)

        return policy_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> PolicyFn:
        if (update_num is not None) and (update_num == 0):
            def policy_fn(domain: Domain, states: List[State], goals: List[Goal], num_samp: int, num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
                assert len(states) == len(goals)  # to stop PyCharm from complaining
                return policy_fn_rand(domain, states, num_samp + num_rand)
        else:
            def policy_fn(domain: Domain, states: List[State], goals: List[Goal], num_samp: int, num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
                inputs_nnet_rep: List[NDArray] = self._get_nnet_inputs_rep(states, goals, num_samp)
                nnet_out_np: List[NDArray[np.float64]] = get_nnet_par_out(inputs_nnet_rep, nnet_par_info)

                actions_l, pdfs_l = self._np_to_acts_and_pdfs(nnet_out_np[0], nnet_out_np[1], len(states), num_samp)
                return _combine_nnet_with_rand(domain, actions_l, pdfs_l, states, num_rand)

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
    def _nnet_out_to_actions(self, nnet_out: NDArray[np.float64]) -> List[Action]:
        pass

    def _get_nnet_inputs_rep(self, states: List[State], goals: List[Goal], num_samp: int) -> List[NDArray]:
        inputs_nnet: List[NDArray] = self.to_np_fn(states, goals)
        inputs_nnet_rep_interleave: List[NDArray] = []
        for inputs_nnet_i in inputs_nnet:
            inputs_nnet_rep_interleave.append(np.repeat(inputs_nnet_i, num_samp, axis=0))
        return inputs_nnet_rep_interleave

    def _np_to_acts_and_pdfs(self, actions_np: NDArray[np.float64], pdfs_np: NDArray[np.float64], num_states: int,
                             num_samp: int) -> Tuple[List[List[Action]], List[List[float]]]:

        actions_l: List[List[Action]] = []
        pdfs_l: List[List[float]] = []
        for state_idx in range(num_states):
            start_idx: int = state_idx * num_samp
            end_idx: int = start_idx + num_samp
            actions_np_state: NDArray[np.float64] = actions_np[start_idx:end_idx]
            pdfs_state: List[float] = pdfs_np[start_idx:end_idx].tolist()

            actions_l.append(self._nnet_out_to_actions(actions_np_state))
            pdfs_l.append(pdfs_state)

        return actions_l, pdfs_l
