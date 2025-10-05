from typing import List, Tuple
from deepxube.base.environment import State, Goal
from deepxube.base.updater import UpdateHeur
from deepxube.base.heuristic import HeurNNetV
from deepxube.pathfinding.v.bwas import BWAS
from numpy.typing import NDArray


class UpdateHeurBWAS(UpdateHeur[HeurNNetV, BWAS]):
    def get_pathfind(self) -> BWAS:
        return BWAS(self.env)

    def get_input_output_np(self, search_ret: Tuple[List[State], List[Goal], List[float]]) -> Tuple[List[NDArray], List[float]]:
        inputs_np: List[NDArray] = self.heur_nnet.to_np(search_ret[0], search_ret[1])
        return inputs_np, search_ret[2]

