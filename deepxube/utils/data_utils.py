from typing import List, Any, Tuple, Optional, Type, cast

import sys

from multiprocessing import Queue
import queue
import os
import shutil
import numpy as np
from numpy.typing import NDArray, ArrayLike

from multiprocessing import shared_memory, resource_tracker
from multiprocessing.shared_memory import SharedMemory


class Logger(object):
    def __init__(self, filename: str, mode: str = "a", echo: bool = True):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
        self.echo: bool = echo

    def write(self, message: str) -> None:
        if self.echo:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        pass


def get_nowait_noerr(q: Queue) -> Any:
    try:
        q_ret: Any = q.get_nowait()
        return q_ret
    except queue.Empty:
        return None


def get_while_not_empty(q: Queue) -> List[Any]:
    q_rets: List[Any] = []

    while not q.empty():
        try:
            q_ret: Any = q.get_nowait()
            q_rets.append(q_ret)
        except queue.Empty:
            break

    return q_rets


def get_in_order(q: Queue, num: int) -> List[Any]:
    ret_vals: List[Any] = [None for _ in range(num)]
    for _ in range(num):
        idx, val = q.get()
        ret_vals[idx] = val
    return ret_vals


def copy_dir_files(src_dir: str, dest_dir: str) -> None:
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)


def sel_l(data_l: List[NDArray], idxs: NDArray) -> List[NDArray]:
    data_l_sel: List[NDArray] = []
    for np_idx in range(len(data_l)):
        data_l_sel.append(data_l[np_idx][idxs])

    return data_l_sel


def combine_l_l(l_l: List[List[NDArray]], comb: str) -> List[NDArray]:
    l_l_comb: List[NDArray] = []
    for np_idx in range(len(l_l[0])):
        l_l_idx: List[NDArray] = [x[np_idx] for x in l_l]

        l_l_idx_comb: NDArray
        if comb == "concat":
            l_l_idx_comb = np.concatenate(l_l_idx, axis=0)
        elif comb == "stack":
            l_l_idx_comb = np.stack(l_l_idx, axis=0)
        else:
            raise ValueError(f"Unknown comb method {comb}")

        l_l_comb.append(l_l_idx_comb)

    return l_l_comb


class SharedNDArray:
    """
    Wraps a numpy array in multiprocessing shared memory.
    Pickleable: can be sent through multiprocessing.Queue.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, name: Optional[str], create: bool):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

        self.shm: SharedMemory
        if create:
            # create new shared block
            assert name is None, "Let SharedMemory do name creation"
            nbytes: int = int(np.prod(self.shape)) * self.dtype.itemsize
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
            resource_tracker.unregister(self.shm._name, "shared_memory")  # TODO hacky
        else:
            # attach to existing shared block
            self.shm = shared_memory.SharedMemory(name=name)
            resource_tracker.unregister(self.shm._name, "shared_memory")  # TODO hacky

        # numpy view backed by shared memory
        self.array: NDArray = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self) -> None:
        """Close this process's handle."""
        self.shm.close()

    def unlink(self) -> None:
        """Free system resource (call once when all processes are done)."""
        self.shm.unlink()

    # --- Pickling support ---
    def __reduce__(self) -> Tuple[Type, Tuple[Tuple[int, ...], np.dtype, str, bool]]:
        """
        When pickled, only send (shape, dtype, name).
        Receiving process reattaches with create=False.
        """
        return self.__class__, (self.shape, self.dtype, self.shm.name, False)

    # --- Convenience ---
    def __getitem__(self, key: Any) -> NDArray:
        return cast(NDArray, self.array[key])

    def __setitem__(self, key: Any, value: ArrayLike) -> None:
        self.array[key] = value

    def __array__(self) -> NDArray:
        return self.array

    def __repr__(self) -> str:
        return f"SharedNDArray(name={self.name}, shape={self.shape}, dtype={self.dtype})"


def np_to_shnd(arr: NDArray) -> SharedNDArray:
    arr_shm: SharedNDArray = SharedNDArray(arr.shape, arr.dtype, None, True)
    arr_shm.array[:] = arr

    return arr_shm
