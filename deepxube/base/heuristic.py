from abc import abstractmethod, ABC
from typing import Callable, List, Any, TypeVar, Generic, cast, Tuple, Optional, Dict, Type

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Goal, Action
from deepxube.base.nnet_input import NNetInput
from deepxube.nnet.nnet_utils import NNetParInfo, nnet_batched, NNetFn, NNetPar, get_nnet_par_out
from deepxube.utils import misc_utils
import torch
from torch import nn, Tensor


In = TypeVar('In', bound=NNetInput)


class HeurNNet(nn.Module, Generic[In], ABC):
    @staticmethod
    @abstractmethod
    def nnet_input_type() -> Type[In]:
        pass

    def __init__(self, nnet_input: In, out_dim: int, q_fix: bool, **kwargs: Any):
        super().__init__(**kwargs)
        self.nnet_input: In = nnet_input
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


class HeurNNetPar(NNetPar[NNetFn]):
    @abstractmethod
    def get_nnet(self) -> HeurNNet:
        pass


S = TypeVar('S', bound=State)
G = TypeVar('G', bound=Goal)
HeurFnV = Callable[[List[S], List[G]], List[float]]


class HeurNNetParV(HeurNNetPar[HeurFnV], Generic[S, G]):
    @staticmethod
    def _get_output(heurs: NDArray[np.float64], update_num: Optional[int]) -> List[float]:
        heurs = np.maximum(heurs[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            heurs = heurs * 0
        return cast(List[float], heurs.astype(np.float64).tolist())

    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnV:
        nnet.eval()

        def heuristic_fn(states: List[S], goals: List[G]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_np(states, goals)
            heurs: NDArray[np.float64] = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]

            return self._get_output(heurs, update_num)
        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnV:
        def heuristic_fn(states: List[S], goals: List[G]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_np(states, goals)
            heurs: NDArray[np.float64] = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]

            return self._get_output(heurs, update_num)

        return heuristic_fn

    @abstractmethod
    def to_np(self, states: List[S], goals: List[G]) -> List[NDArray[Any]]:
        pass


A = TypeVar('A', bound=Action)
HeurFnQ = Callable[[List[S], List[G], List[List[A]]], List[List[float]]]


class HeurNNetParQ(HeurNNetPar[HeurFnQ], Generic[S, A, G]):
    @abstractmethod
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnQ:
        pass

    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        pass

    @abstractmethod
    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        pass


class HeurNNetParQFixOut(HeurNNetParQ[S, A, G], ABC):
    """ DQN with a fixed output shape

    """
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnQ:
        nnet.eval()

        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            inputs_nnet: List[NDArray] = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray[np.float64] = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]
            return self._get_output(states, q_vals_np, update_num)

        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            inputs_nnet: List[NDArray] = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray[np.float64] = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]
            return self._get_output(states, q_vals_np, update_num)

        return heuristic_fn

    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        self._check_same_num_acts(actions_l)
        return self._to_np_fixed_acts(states, goals, actions_l)

    @abstractmethod
    def _to_np_fixed_acts(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        pass

    def _get_input(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray]:
        inputs_nnet: List[NDArray] = self.to_np(states, goals, actions_l)
        return inputs_nnet

    @staticmethod
    def _get_output(states: List[S], q_vals_np: NDArray[np.float64], update_num: Optional[int]) -> List[List[float]]:
        assert q_vals_np.shape[0] == len(states)
        q_vals_np = np.maximum(q_vals_np, 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0
        q_vals_l: List[List[float]] = [q_vals_np[state_idx].astype(np.float64).tolist() for state_idx in
                                       range(len(states))]
        return q_vals_l

    @staticmethod
    def _check_same_num_acts(actions_l: List[List[A]]) -> None:
        assert len(set(len(actions) for actions in actions_l)) == 1, "num actions should be the same for all instances"


class HeurNNetParQIn(HeurNNetParQ[S, A, G], ABC):
    """ DQN that takes a single action as input

    """
    def get_nnet_fn(self, nnet: nn.Module, batch_size: Optional[int], device: torch.device,
                    update_num: Optional[int]) -> HeurFnQ:
        nnet.eval()

        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            inputs_nnet, states_rep, split_idxs = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray = nnet_batched(nnet, inputs_nnet, batch_size, device)[0]
            return self._get_output(states_rep, q_vals_np, split_idxs, update_num)

        return heuristic_fn

    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo, update_num: Optional[int]) -> HeurFnQ:
        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            inputs_nnet, states_rep, split_idxs = self._get_input(states, goals, actions_l)
            q_vals_np: NDArray = get_nnet_par_out(inputs_nnet, nnet_par_info)[0]
            return self._get_output(states_rep, q_vals_np, split_idxs, update_num)

        return heuristic_fn

    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        assert all((len(actions) == 1) for actions in actions_l), "there should only be one action per state/goal pair"
        actions_one: List[A] = [actions[0] for actions in actions_l]
        return self._to_np_one_act(states, goals, actions_one)

    @abstractmethod
    def _to_np_one_act(self, states: List[S], goals: List[G], actions: List[A]) -> List[NDArray[Any]]:
        pass

    def _get_input(self, states: List[S], goals: List[G],
                   actions_l: List[List[A]]) -> Tuple[List[NDArray], List[S], List[int]]:
        actions_flat, split_idxs = misc_utils.flatten(actions_l)
        states_rep: List[S] = []
        goals_rep: List[G] = []
        for state, goal, actions in zip(states, goals, actions_l, strict=True):
            states_rep.extend([state] * len(actions))
            goals_rep.extend([goal] * len(actions))
        inputs_nnet: List[NDArray] = self._to_np_one_act(states_rep, goals_rep, actions_flat)

        return inputs_nnet, states_rep, split_idxs

    @staticmethod
    def _get_output(states_rep: List[S], q_vals_np: NDArray[np.float64], split_idxs: List[int],
                    update_num: Optional[int]) -> List[List[float]]:
        assert q_vals_np.shape[0] == len(states_rep)
        q_vals_np = np.maximum(q_vals_np[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0

        q_vals_flat: List[float] = q_vals_np.astype(np.float64).tolist()
        q_vals_l: List[List[float]] = misc_utils.unflatten(q_vals_flat, split_idxs)
        return q_vals_l


class HeurNNetParser(ABC):
    @abstractmethod
    def parse(self, args_str: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def help(self) -> str:
        pass
