from typing import List

from deepxube.base.pathfinding import InstArgs
from deepxube.base.updater import UpdateHeurV, UpdateHeurQEnum
from deepxube.pathfinding.v.bwas import BWAS, InstArgsBWAS
from deepxube.pathfinding.q.bwqs import BWQS, InstArgsBWQS


class UpdateHeurBWAS(UpdateHeurV[BWAS]):
    def get_pathfind(self) -> BWAS:
        assert self.heur_fn is not None
        return BWAS(self.env, self.heur_fn)

    def _get_inst_args(self, num: int) -> List[InstArgs]:
        return [InstArgsBWAS() for _ in range(num)]


class UpdateHeurBWQS(UpdateHeurQEnum[BWQS]):
    def get_pathfind(self) -> BWQS:
        assert self.heur_fn is not None
        return BWQS(self.env, self.heur_fn)

    def _get_inst_args(self, num: int) -> List[InstArgs]:
        return [InstArgsBWQS() for _ in range(num)]
