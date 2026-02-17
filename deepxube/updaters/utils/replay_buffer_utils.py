from abc import ABC, abstractmethod
from typing import Deque, Tuple, List, Optional, Generic, TypeVar
from collections import deque
from deepxube.base.domain import State, Action, Goal
import numpy as np


ReplayVElem = Tuple[State, Goal, bool]
ReplayQElem = Tuple[State, Goal, bool, Action, float, State]

ReplayVRet = Tuple[List[State], List[Goal], List[bool]]
ReplayQRet = Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]

Elem = TypeVar('Elem')
SampRet = TypeVar('SampRet')


class ReplayBuffer(Generic[Elem, SampRet], ABC):
    def __init__(self, max_size: int):
        self.deque: Deque[Elem] = deque([], max_size)

    def add(self, data: List[Elem]) -> None:
        self.deque.extend(data)

    def sample(self, num: int) -> SampRet:
        assert self.size() > 0, f"Replay buffer size should not be {self.size()}"
        idxs: List[int] = np.random.randint(0, len(self.deque), size=num).tolist()
        elems: List[Elem] = [self.deque[idx] for idx in idxs]
        return self._elems_to_ret(elems)

    def size(self) -> int:
        return len(self.deque)

    def max_size(self) -> int:
        maxlen: Optional[int] = self.deque.maxlen
        assert maxlen is not None

        return maxlen

    @abstractmethod
    def _elems_to_ret(self, elems: List[Elem]) -> SampRet:
        pass


class ReplayBufferV(ReplayBuffer[ReplayVElem, ReplayVRet]):
    def _elems_to_ret(self, elems: List[ReplayVElem]) -> ReplayVRet:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        is_solved_l: List[bool] = [replay_q_elem[2] for replay_q_elem in elems]

        return states, goals, is_solved_l


class ReplayBufferQ(ReplayBuffer[ReplayQElem, ReplayQRet]):
    def _elems_to_ret(self, elems: List[ReplayQElem]) -> ReplayQRet:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        is_solved_l: List[bool] = [replay_q_elem[2] for replay_q_elem in elems]
        actions: List[Action] = [replay_q_elem[3] for replay_q_elem in elems]
        tcs: List[float] = [replay_q_elem[4] for replay_q_elem in elems]
        states_next: List[State] = [replay_q_elem[5] for replay_q_elem in elems]

        return states, goals, is_solved_l, actions, tcs, states_next
