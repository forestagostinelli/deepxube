from abc import abstractmethod, ABC
from typing import Callable, List, Any, TypeVar, Generic, cast

import numpy as np
from numpy.typing import NDArray

from deepxube.base.env import State, Goal, Action
from deepxube.nnet.nnet_utils import NNetParInfo
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd
from torch import nn


HeurFn = Callable[..., Any]


H = TypeVar('H', bound=HeurFn)


class NNetPar(ABC):
    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> Callable[..., Any]:
        pass


class HeurNNet(NNetPar, Generic[H]):
    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> H:
        pass

    @abstractmethod
    def get_nnet(self) -> nn.Module:
        pass

    @abstractmethod
    def to_np(self, *args) -> List[NDArray[Any]]:
        pass


def _get_nnet_par_out(inputs_nnet: List[NDArray], nnet_par_info: NNetParInfo) -> NDArray:
    inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i)
                                            for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

    nnet_par_info.nnet_i_q.put((nnet_par_info.proc_id, inputs_nnet_shm))

    out_shm: SharedNDArray = nnet_par_info.nnet_o_q.get()
    out: NDArray = out_shm.array.copy()

    for arr_shm in inputs_nnet_shm + [out_shm]:
        arr_shm.close()
        arr_shm.unlink()

    return out


S = TypeVar('S', bound=State)
G = TypeVar('G', bound=Goal)
HeurFnV = Callable[[List[S], List[G]], List[float]]


class HeurNNetV(HeurNNet[HeurFnV], Generic[S, G]):
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> HeurFnV:
        def heuristic_fn(states: List[S], goals: List[G]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_np(states, goals)
            heurs: NDArray = _get_nnet_par_out(inputs_nnet, nnet_par_info)

            return cast(List[float], heurs[:, 0].astype(np.float64).tolist())

        return heuristic_fn

    @abstractmethod
    def to_np(self, states: List[S], goals: List[G]) -> List[NDArray[Any]]:
        pass


A = TypeVar('A', bound=Action)
HeurFnQ = Callable[[List[S], List[G], List[List[A]]], List[List[float]]]


class HeurNNetQ(HeurNNet[HeurFnQ], Generic[S, A, G]):
    @abstractmethod
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> HeurFnQ:
        pass

    @abstractmethod
    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        pass


def _check_same_num_acts(actions_l: List[List[A]]):
    assert len(set(len(actions) for actions in actions_l)) == 1, "num actions should be the same for all instances"


class HeurNNetQFixOut(HeurNNetQ[S, A, G], ABC):
    """ DQN with a fixed output shape

    """
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> HeurFnQ:
        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            _check_same_num_acts(actions_l)
            inputs_nnet: List[NDArray] = self.to_np(states, goals, actions_l)
            q_vals_np: NDArray = _get_nnet_par_out(inputs_nnet, nnet_par_info)

            assert q_vals_np.shape[0] == len(states)
            q_vals_l: List[List[float]] = [q_vals_np[state_idx].astype(np.float64).tolist() for state_idx in
                                           range(len(states))]

            return q_vals_l

        return heuristic_fn

    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        _check_same_num_acts(actions_l)
        return self._to_np_fixed_acts(states, goals, actions_l)

    @abstractmethod
    def _to_np_fixed_acts(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        pass


class HeurNNetQIn(HeurNNetQ[S, A, G], ABC):
    """ DQN that takes a single action as input

    """
    def get_nnet_par_fn(self, nnet_par_info: NNetParInfo) -> HeurFnQ:
        def heuristic_fn(states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[List[float]]:
            action_lens_np: NDArray = np.array([len(actions) for actions in actions_l]).astype(int)
            max_num_acts: int = int(action_lens_np.max())
            q_vals_l: List[List[float]] = [[] for _ in states]
            for act_idx in range(0, max_num_acts):
                idxs_get: List[int] = np.where(action_lens_np >= (act_idx + 1))[0].tolist()
                states_idx: List[S] = [states[idx] for idx in idxs_get]
                goals_idx: List[G] = [goals[idx] for idx in idxs_get]
                actions_idx: List[List[A]] = [[actions_l[idx][act_idx]] for idx in idxs_get]

                inputs_nnet: List[NDArray] = self.to_np(states_idx, goals_idx, actions_idx)
                q_vals_np_idx: NDArray = _get_nnet_par_out(inputs_nnet, nnet_par_info)[:, 0]
                assert q_vals_np_idx.shape[0] == len(states_idx)
                q_vals_idx: List[float] = q_vals_np_idx.astype(np.float64).tolist()
                for idx, q_val_idx in enumerate(q_vals_idx):
                    q_vals_l[idx].append(q_val_idx)

            return q_vals_l

        return heuristic_fn

    def to_np(self, states: List[S], goals: List[G], actions_l: List[List[A]]) -> List[NDArray[Any]]:
        assert all((len(actions) == 1) for actions in actions_l), "there should only be one action per state/goal pair"
        actions_one: List[A] = [actions[0] for actions in actions_l]
        return self._to_np_one_act(states, goals, actions_one)

    @abstractmethod
    def _to_np_one_act(self, states: List[S], goals: List[G], actions: List[A]) -> List[NDArray[Any]]:
        pass
