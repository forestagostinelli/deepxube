from typing import List, Tuple

from deepxube.base.heuristic import PolicyNNet
from deepxube.base.updater import UpdatePolicy
from deepxube.base.trainer import Train
from deepxube.utils.timing_utils import Times

import torch.nn as nn

import numpy as np
from numpy.typing import NDArray
import time


class TrainHeur(Train[PolicyNNet, UpdatePolicy]):
    def _train_itr(self, batch: List[NDArray], first_itr_in_update: bool, times: Times) -> float:
        start_time = time.time()
        inputs_batch_np: List[NDArray] = batch[:-1]
        ctgs_batch_np: NDArray = batch[-1]
        ctgs_batch_np = np.expand_dims(ctgs_batch_np.astype(np.float32), 1)

        self.nnet.train()
        ctgs_batch_nnet, loss = train_heur_nnet_step(self.nnet, inputs_batch_np, ctgs_batch_np, self.optimizer, nn.MSELoss(), self.device, self.status.itr,
                                                     self.train_args, self.train_start_time)
        self.writer.add_scalar("train/loss", loss, self.status.itr)

        self.status.itr += 1
        times.record_time("train", time.time() - start_time)
        return loss

    def _add_post_up_info(self) -> List[str]:
        return []

    def _get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        return self.updater.get_policy_train_shapes_dtypes()
