from typing import List, Tuple

from deepxube.base.heuristic import HeurNNet
from deepxube.base.updater import UpdateHeur
from deepxube.base.trainer import Train, update_optimizer
from deepxube.utils.timing_utils import Times
from deepxube.trainers.utils.train_utils import train_nnet_step, ctgs_summary

import numpy as np
from numpy.typing import NDArray
import time


class TrainHeur(Train[HeurNNet, UpdateHeur]):
    @staticmethod
    def data_parallel() -> bool:
        return True

    def _train_itr(self, batch: List[NDArray], first_itr_in_update: bool, times: Times) -> float:
        start_time = time.time()
        ctgs_batch_np: NDArray = batch[-1]
        ctgs_batch_np = np.expand_dims(ctgs_batch_np.astype(np.float32), 1)
        batch = batch[:-1] + [ctgs_batch_np]

        self.nnet.train()
        update_optimizer(self.optimizer, self.nnet, self.status.itr)
        fwd_tr_tensors, loss = train_nnet_step(self.nnet, batch, self.optimizer, self.device, self.status.itr, self.train_args, self.train_start_time)
        self.writer.add_scalar("train/loss", loss, self.status.itr)

        if first_itr_in_update:
            self.train_summary.itr_to_in_out[self.status.itr] = (ctgs_batch_np, fwd_tr_tensors[0])
        times.record_time("train", time.time() - start_time)
        return loss

    def _add_post_up_info(self) -> List[str]:
        ctgs_mean, ctgs_min, ctgs_max = ctgs_summary([self.db.arrays[-1]])
        self.writer.add_scalar("train/ctgs/mean", ctgs_mean, self.status.itr)
        self.writer.add_scalar("train/ctgs/min", ctgs_min, self.status.itr)
        self.writer.add_scalar("train/ctgs/max", ctgs_max, self.status.itr)
        return [f"cost-to-go (mean/min/max): {ctgs_mean:.2f}/{ctgs_min:.2f}/{ctgs_max:.2f}"]

    def _get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        return self.updater.get_heur_train_shapes_dtypes()
