from typing import List

from deepxube.base.pathfinding import InstArgs
from deepxube.base.updater import UpdateHeurV, UpdateHeurQ
from deepxube.pathfinding.v.bwas import BWAS, InstArgsBWAS
from deepxube.pathfinding.q.bwqs import BWQS, InstArgsBWQS


class UpdateHeurBWAS(UpdateHeurV):
    def get_pathfind(self) -> BWAS:
        return BWAS(self.env)

    def _get_inst_args(self, num: int) -> List[InstArgs]:
        return [InstArgsBWAS() for _ in range(num)]


class UpdateHeurBWQS(UpdateHeurQ):
    def get_pathfind(self) -> BWQS:
        return BWQS(self.env)

    def _get_inst_args(self, num: int) -> List[InstArgs]:
        return [InstArgsBWQS() for _ in range(num)]
