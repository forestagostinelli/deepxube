from typing import List, Tuple, Any, Union, Set
from collections import OrderedDict
import numpy as np
import math
import re
import torch
import time


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: Union[List[Any], np.array], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])

        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def cum_min(data: List) -> List:
    data_cum_min: List = []
    prev_min = float('inf')
    for data_i in data:
        prev_min = min(prev_min, data_i)
        data_cum_min.append(prev_min)

    return data_cum_min


def remove_all_whitespace(val: str) -> str:
    pattern = re.compile(r'\s+')
    val = re.sub(pattern, '', val)

    return val


def random_subset(set_orig: Union[Set, frozenset], keep_prob: bool) -> Set:
    rand_vals: np.array = np.random.rand(len(set_orig))
    keep_arr: np.array = np.array(rand_vals < keep_prob)
    rand_subset: Set = set(elem for elem, keep_i in zip(set_orig, keep_arr) if keep_i)

    return rand_subset


# Time profiling
def init_times(time_names: List[str]) -> OrderedDict:
    times: OrderedDict = OrderedDict()
    for time_name in time_names:
        times[time_name] = 0.0
    return times


def record_time(times: OrderedDict, time_name: str, start_time, on_gpu: bool = False):
    """ Increments time if time_name is already in times. Synchronizes if on_gpu is true.

    """
    if on_gpu:
        torch.cuda.synchronize()

    time_elapsed = time.time() - start_time
    record_time_elapsed(times, time_name, time_elapsed)


def record_time_elapsed(times: OrderedDict, time_name: str, time_elapsed):
    """ Increments time if time_name is already in times. Synchronizes if on_gpu is true.

    """
    if time_name in times.keys():
        times[time_name] += time_elapsed
    else:
        times[time_name] = time_elapsed


def add_times(times: OrderedDict, times_to_add: OrderedDict):
    for key, value in times_to_add.items():
        if key not in times:
            times[key] = 0.0
        times[key] += value


def reset_times(times: OrderedDict):
    for key in times.keys():
        times[key] = 0.0


def get_time_str(times: OrderedDict) -> str:
    time_str_l: List[str] = []
    for key, val in times.items():
        time_str_i = "%s: %.2f" % (key, val)
        time_str_l.append(time_str_i)
    time_str: str = ", ".join(time_str_l)

    return time_str


def boltzmann(qvals: List[float], temp: float) -> List[float]:
    if len(qvals) == 1:
        return [1.0]
    else:
        qvals_np: np.array = np.array(qvals)
        exp_vals_np: np.array = np.exp((1.0 / temp) * (-(qvals_np - np.max(qvals_np))))
        probs_np: np.array = exp_vals_np / np.sum(exp_vals_np)

    return list(probs_np)
