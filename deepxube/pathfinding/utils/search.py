import random
from typing import List

import numpy as np
from numpy._typing import NDArray

from deepxube.utils.misc_utils import boltzmann


def greedy_next_idx(q_vals_l: List[List[float]], temps: List[float], eps_l: List[float]) -> List[int]:
    next_idxs: List[int] = []

    rand_vals_eps: NDArray = np.random.random(len(q_vals_l))
    for q_vals, temp, eps, rand_val_eps in zip(q_vals_l, temps, eps_l, rand_vals_eps):
        next_idx: int
        if rand_val_eps < eps:
            next_idx = random.randrange(0, len(q_vals))
        elif temp > 0:
            probs: List[float] = boltzmann((-np.array(q_vals)).tolist(), temp)
            next_idx = int(np.random.multinomial(1, np.array(probs)).argmax())
        else:
            next_idx = int(np.argmin(q_vals))
        next_idxs.append(next_idx)

    return next_idxs
