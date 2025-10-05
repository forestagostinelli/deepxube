from deepxube.base.updater import UpdateHeurV
from deepxube.pathfinding.v.bwas import BWAS


class UpdateHeurBWAS(UpdateHeurV):
    def get_pathfind(self) -> BWAS:
        return BWAS(self.env)
