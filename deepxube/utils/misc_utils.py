from typing import List, Tuple, Any, Union, Set
import numpy as np
import math
import re
from numpy.typing import NDArray


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: Union[List[Any], NDArray[Any]], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(list(data[start_idx:end_idx]))
        start_idx = end_idx

    data_split.append(list(data[start_idx:]))

    return data_split


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def remove_all_whitespace(val: str) -> str:
    pattern = re.compile(r'\s+')
    val = re.sub(pattern, '', val)

    return val


def random_subset(set_orig: Union[Set[Any], frozenset[Any]], keep_prob: bool) -> Set[Any]:
    rand_vals: NDArray[Any] = np.random.rand(len(set_orig))
    keep_arr: NDArray[np.bool_] = np.array(rand_vals < keep_prob)
    rand_subset: Set[Any] = set(elem for elem, keep_i in zip(set_orig, keep_arr) if keep_i)

    return rand_subset


def boltzmann(vals: List[float], temp: float) -> List[float]:
    if len(vals) == 1:
        return [1.0]
    else:
        vals_np: NDArray[np.float64] = np.array(vals)
        exp_vals_np: NDArray[np.float64] = np.exp((1.0 / temp) * (vals_np - np.max(vals_np)))
        probs_np: NDArray[np.float64] = exp_vals_np / np.sum(exp_vals_np)

        return list(probs_np)
