from typing import List, Tuple, Any

import numpy as np

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

    def get_input_shapes_dtypes(self) -> List[Tuple[Tuple[int, ...], np.dtype]]:
        states, goals = self.env.get_start_goal_pairs([0])
        inputs_nnet: List[NDArray[Any]] = self.heur_nnet.to_np(states, goals)

        shapes_dypes: List[Tuple[Tuple[int, ...], np.dtype]] = []
        for inputs_nnet_i in inputs_nnet:
            shapes_dypes.append((inputs_nnet_i[0].shape, inputs_nnet_i.dtype))

        return shapes_dypes


