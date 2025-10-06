from typing import Optional, List

from deepxube.base.pathfinding import InstArgs
from deepxube.base.updater import UpdateHeurV
from deepxube.pathfinding.v.bwas import BWAS


class UpdateHeurBWAS(UpdateHeurV):
    def get_pathfind(self) -> BWAS:
        return BWAS(self.env)

    def _get_inst_args(self, num: int) -> Optional[List[InstArgs]]:
        return None
