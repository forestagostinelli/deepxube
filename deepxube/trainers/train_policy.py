from typing import List, Type

from deepxube.base.nnet import PolicyNNet
from deepxube.base.updater import UpdatePolicy
from deepxube.base.trainer import Train, update_optimizer, TrainParser
from deepxube.utils.train_utils import train_nnet_step
from deepxube.utils.timing_utils import Times
from deepxube.factories.trainer_factory import trainer_factory

from numpy.typing import NDArray
import time


@trainer_factory.register_class("tr_p")
class TrainPolicy(Train[PolicyNNet, UpdatePolicy]):
    @staticmethod
    def data_parallel() -> bool:
        return True

    @staticmethod
    def nnet_type() -> Type[PolicyNNet]:
        return PolicyNNet

    @staticmethod
    def updater_type() -> Type[UpdatePolicy]:
        return UpdatePolicy

    @staticmethod
    def get_nnet_name() -> str:
        return "policy"

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


@trainer_factory.register_parser("tr_p")
class TrainHeurParser(TrainParser):
    pass
