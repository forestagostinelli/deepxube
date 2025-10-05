from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Any

import numpy as np
from numpy.typing import NDArray

from deepxube.base.environment import State, Goal, Action
from deepxube.nnet.nnet_utils import NNetPar
from deepxube.utils.data_utils import SharedNDArray, np_to_shnd


class NNetType(Enum):
    V = 1
    Q = 2


class NNetQType(Enum):
    FIXED = 1
    DYNAMIC = 2


HeurFnV = Callable[[List[State], List[Goal]], List[float]]


class NNetParV(NNetPar):
    def get_nnet_par_fn(self) -> HeurFnV:
        assert self.nnet_fn_i_q is not None
        assert self.nnet_fn_o_q is not None
        assert self.proc_id is not None
        def heuristic_fn(states: List[State], goals: List[Goal]) -> List[float]:
            inputs_nnet: List[NDArray] = self.to_nnet(states, goals)
            inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i)
                                                    for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

            self.nnet_fn_i_q.put((self.proc_id, inputs_nnet_shm))

            heurs_shm: SharedNDArray = self.nnet_fn_o_q.get()
            heurs: NDArray = heurs_shm.array.copy()

            for arr_shm in inputs_nnet_shm + [heurs_shm]:
                arr_shm.close()
                arr_shm.unlink()

            return heurs[:, 0].astype(np.float64).tolist()

        return heuristic_fn

    @abstractmethod
    def to_nnet(self, states: List[State], goals: List[Goal]) -> List[NDArray[Any]]:
        pass


HeurFnQ = Callable[[List[State], List[Goal], List[List[Action]]], List[List[float]]]


class NNetParQ(NNetPar):
    def get_nnet_par_fn(self) -> HeurFnQ:
        assert self.nnet_fn_i_q is not None
        assert self.nnet_fn_o_q is not None
        assert self.proc_id is not None
        def heuristic_fn(states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
            q_vals_l: List[List[float]]
            if self.nnet_q_type is NNetQType.FIXED:
                assert len(set(len(actions) for actions in actions_l)) == 1, "action size should be fixed"
                inputs_nnet: List[NDArray] = self.to_nnet(states, goals, actions_l)
                inputs_nnet_shm: List[SharedNDArray] = [np_to_shnd(inputs_nnet_i)
                                                        for input_idx, inputs_nnet_i in enumerate(inputs_nnet)]

                self.nnet_fn_i_q.put((self.proc_id, inputs_nnet_shm))

                heurs_shm: SharedNDArray = self.nnet_fn_o_q.get()
                q_vals: NDArray = heurs_shm.array.copy()
                assert q_vals.shape[0] == len(states)
                q_vals_l = [q_vals[state_idx].astype(np.float64).tolist() for state_idx in range(len(states))]

                for arr_shm in inputs_nnet_shm + [heurs_shm]:
                    arr_shm.close()
                    arr_shm.unlink()
            elif self.nnet_q_type is NNetQType.DYNAMIC:
                breakpoint()

            return q_vals_l

        return heuristic_fn

    @abstractmethod
    def to_nnet(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[NDArray[Any]]:
        pass

    @property
    @abstractmethod
    def nnet_q_type(self) -> NNetQType:
        pass
