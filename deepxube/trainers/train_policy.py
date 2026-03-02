from typing import List, Tuple

from deepxube.base.heuristic import PolicyNNet
from deepxube.base.updater import UpdatePolicy
from deepxube.base.trainer import Train
from deepxube.trainers.utils.train_utils import train_policy_nnet_step
from deepxube.utils.timing_utils import Times

import numpy as np
from numpy.typing import NDArray
import time


class TrainPolicy(Train[PolicyNNet, UpdatePolicy]):
    def _train_itr(self, batch: List[NDArray], first_itr_in_update: bool, times: Times) -> float:
        start_time = time.time()
        states_goals_np: List[NDArray] = batch[:-1]
        actions_np: NDArray = batch[-1]

        self.nnet.train()
        # TODO make KL argument
        loss = train_policy_nnet_step(self.nnet, states_goals_np, actions_np, self.optimizer, self.device, self.status.itr, self.train_args, 0.1,
                                      self.train_start_time)
        self.writer.add_scalar("train/loss", loss, self.status.itr)

        self.status.itr += 1
        times.record_time("train", time.time() - start_time)
        return loss

    def _add_post_up_info(self) -> List[str]:
        return []

    def _get_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        return self.updater.get_policy_train_shapes_dtypes()
