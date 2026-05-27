from typing import List, Tuple

from deepxube.base.heuristic import PolicyNNet
from deepxube.base.updater import UpdatePolicy
from deepxube.base.trainer import Train, update_optimizer
from deepxube.trainers.utils.train_utils import train_nnet_step
from deepxube.utils.timing_utils import Times

import numpy as np
from numpy.typing import NDArray
import time


class TrainPolicy(Train[PolicyNNet, UpdatePolicy]):
    @staticmethod
    def data_parallel() -> bool:
        return True

    def _train_itr(self, batch: List[NDArray], first_itr_in_update: bool, times: Times) -> float:
        start_time = time.time()

        self.nnet.train()
        update_optimizer(self.optimizer, self.nnet, self.status.itr)
        loss = train_nnet_step(self.nnet, batch, self.optimizer, self.device, self.status.itr, self.train_args, self.train_start_time)[1]
        self.writer.add_scalar("train/loss", loss, self.status.itr)

        times.record_time("train", time.time() - start_time)
        return loss

    def _add_post_up_info(self) -> List[str]:
        return []

    def _get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        return self.updater.get_policy_train_shapes_dtypes()
