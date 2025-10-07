from typing import List

from deepxube.base.pathfinding import InstArgs
from deepxube.base.updater import UpdateHeurV, UpdateHeurQ
from deepxube.pathfinding.v.bwas import BWAS, InstArgsBWAS


class UpdateHeurBWAS(UpdateHeurV):
    def get_pathfind(self) -> BWAS:
        return BWAS(self.env)

    def _get_inst_args(self, num: int) -> List[InstArgs]:
        return [InstArgsBWAS() for _ in range(num)]
