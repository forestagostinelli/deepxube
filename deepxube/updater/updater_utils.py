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

    def sample(self, num: int) -> List[ReplayQElem]:
        idxs: List[int] = np.random.randint(0, len(self.deque), size=num).tolist()
        return [self.deque[idx] for idx in idxs]

    def size(self) -> int:
        return len(self.deque)

    def max_size(self) -> int:
        maxlen: Optional[int] = self.deque.maxlen
        assert maxlen is not None

        return maxlen
