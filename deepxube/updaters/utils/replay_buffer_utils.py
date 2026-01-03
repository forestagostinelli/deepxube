from typing import Deque, Tuple, List, Optional
from collections import deque
from deepxube.base.domain import State, Action, Goal
import numpy as np

ReplayQElem = Tuple[State, Goal, bool, Action, float, State]


class ReplayBufferQ:
    def __init__(self, max_size: int):
        self.deque: Deque[ReplayQElem] = deque([], max_size)

    def add(self, data: List[ReplayQElem]) -> None:
        self.deque.extend(data)

    def sample(self, num: int) -> Tuple[List[State], List[Goal], List[bool], List[Action], List[float], List[State]]:
        assert self.size() > 0, f"Replay buffer size should not be {self.size()}"
        idxs: List[int] = np.random.randint(0, len(self.deque), size=num).tolist()
        rb_elems: List[ReplayQElem] = [self.deque[idx] for idx in idxs]
        states: List[State] = [replay_q_elem[0] for replay_q_elem in rb_elems]
        goals: List[Goal] = [replay_q_elem[1] for replay_q_elem in rb_elems]
        is_solved_l: List[bool] = [replay_q_elem[2] for replay_q_elem in rb_elems]
        actions: List[Action] = [replay_q_elem[3] for replay_q_elem in rb_elems]
        tcs: List[float] = [replay_q_elem[4] for replay_q_elem in rb_elems]
        states_next: List[State] = [replay_q_elem[5] for replay_q_elem in rb_elems]

        return states, goals, is_solved_l, actions, tcs, states_next

    def size(self) -> int:
        return len(self.deque)

    def max_size(self) -> int:
        maxlen: Optional[int] = self.deque.maxlen
        assert maxlen is not None

        return maxlen
