from abc import ABC, abstractmethod
from typing import Deque, Tuple, List, Optional, Generic, TypeVar, Any
from collections import deque
from deepxube.base.domain import State, Action, Goal
import numpy as np


ReplayVElem = Tuple[State, Goal, Any, bool]
ReplayQElem = Tuple[State, Goal, Action, Any, bool, float, State]
ReplayPElem = Tuple[State, Goal, Action, Any]

ReplayVLabElem = Tuple[State, Goal, Any, float]

InputV = Tuple[List[State], List[Goal], List[Any]]
InputQ = Tuple[List[State], List[Goal], List[Action], List[Any]]
InputP = Tuple[List[State], List[Goal], List[Action], List[Any]]

ReplayV = List[bool]
ReplayQ = Tuple[List[bool], List[float], List[State]]
ReplayP = Optional[None]

ReplayVLab = List[float]

Elem = TypeVar('Elem')
ID_T = TypeVar('ID_T')
RD_T = TypeVar('RD_T')


class ReplayBuffer(Generic[Elem, ID_T, RD_T], ABC):
    def __init__(self, max_size: int):
        self.deque: Deque[Elem] = deque([], max_size)

    @abstractmethod
    def add(self, input_data: ID_T, replay_data: RD_T) -> None:
        pass

    def sample(self, num: int) -> Tuple[ID_T, RD_T]:
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
    def _elems_to_ret(self, elems: List[Elem]) -> Tuple[ID_T, RD_T]:
        pass


class ReplayBufferV(ReplayBuffer[ReplayVElem, InputV, ReplayV]):
    def add(self, input_data: InputV, replay_data: ReplayV) -> None:
        data: List[ReplayVElem] = list(zip(*input_data, replay_data, strict=True))
        self.deque.extend(data)

    def _elems_to_ret(self, elems: List[ReplayVElem]) -> Tuple[InputV, ReplayV]:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        contexts: List[Any] = [replay_q_elem[2] for replay_q_elem in elems]
        is_solved_l: List[bool] = [replay_q_elem[3] for replay_q_elem in elems]

        return (states, goals, contexts), is_solved_l


class ReplayBufferQ(ReplayBuffer[ReplayQElem, InputQ, ReplayQ]):
    def add(self, input_data: InputQ, replay_data: ReplayQ) -> None:
        data: List[ReplayQElem] = list(zip(*input_data, *replay_data, strict=True))
        self.deque.extend(data)

    def _elems_to_ret(self, elems: List[ReplayQElem]) -> Tuple[InputQ, ReplayQ]:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        actions: List[Action] = [replay_q_elem[2] for replay_q_elem in elems]
        contexts: List[Any] = [replay_q_elem[3] for replay_q_elem in elems]
        is_solved_l: List[bool] = [replay_q_elem[4] for replay_q_elem in elems]
        tcs: List[float] = [replay_q_elem[5] for replay_q_elem in elems]
        states_next: List[State] = [replay_q_elem[6] for replay_q_elem in elems]

        return (states, goals, actions, contexts), (is_solved_l, tcs, states_next)


class ReplayBufferP(ReplayBuffer[ReplayPElem, InputP, ReplayP]):
    def add(self, input_data: InputP, replay_data: ReplayP) -> None:
        data: List[ReplayPElem] = list(zip(*input_data, strict=True))
        self.deque.extend(data)

    def _elems_to_ret(self, elems: List[ReplayPElem]) -> Tuple[InputP, ReplayP]:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        actions: List[Action] = [replay_q_elem[2] for replay_q_elem in elems]
        contexts: List[Any] = [replay_q_elem[3] for replay_q_elem in elems]

        return (states, goals, actions, contexts), None


class ReplayBufferVLab(ReplayBuffer[ReplayVLabElem, InputV, ReplayVLab]):
    def add(self, input_data: InputV, replay_data: ReplayVLab) -> None:
        data: List[ReplayVLabElem] = list(zip(*input_data, replay_data, strict=True))
        self.deque.extend(data)

    def _elems_to_ret(self, elems: List[ReplayVLabElem]) -> Tuple[InputV, ReplayVLab]:
        states: List[State] = [replay_q_elem[0] for replay_q_elem in elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in elems]
        contexts: List[Any] = [replay_q_elem[2] for replay_q_elem in elems]
        labels: List[float] = [replay_q_elem[3] for replay_q_elem in elems]

        return (states, goals, contexts), labels
