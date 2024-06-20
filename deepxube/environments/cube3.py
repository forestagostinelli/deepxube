from typing import List, Dict, Tuple, Union, Any, Set, Optional
from deepxube.utils import misc_utils
from deepxube.nnet.pytorch_models import FullyConnectedModel, ResnetModel
from deepxube.logic.logic_objects import Atom, Model
from deepxube.visualizers.cube3_viz_simple import InteractiveCube
from .environment_abstract import EnvGrndAtoms, State, Action, Goal, HeurFnNNet

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from random import randrange

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from itertools import permutations
import re
from numpy.typing import NDArray


class Cube3ProcessStates(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, one_hot_depth: int):
        super().__init__()
        self.state_dim: int = state_dim
        self.one_hot_depth: int = one_hot_depth

    def forward(self, states_nnet: Tensor):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        return x


class Cube3FCResnet(nn.Module):
    def __init__(self, input_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int, out_dim: int,
                 batch_norm: bool, weight_norm: bool):
        super().__init__()
        self.first_fc = FullyConnectedModel(input_dim, [h1_dim, resnet_dim], [batch_norm] * 2, ["RELU"] * 2,
                                            weight_norms=[weight_norm] * 2)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, out_dim, batch_norm, weight_norm=weight_norm,
                                  layer_act="RELU")

    def forward(self, x: Tensor):
        x = self.first_fc(x)
        x = self.resnet(x)

        return x


class Cube3NNet(HeurFnNNet):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_res_blocks: int,
                 out_dim: int, batch_norm: bool, weight_norm: bool, nnet_type: str):
        super().__init__(nnet_type)
        self.state_proc = Cube3ProcessStates(state_dim, one_hot_depth)

        input_dim: int = state_dim * one_hot_depth * 2
        self.heur = Cube3FCResnet(input_dim, h1_dim, resnet_dim, num_res_blocks, out_dim, batch_norm, weight_norm)

    def forward(self, states_goals_l: List[Tensor]):
        states_proc = self.state_proc(states_goals_l[0])
        goals_proc = self.state_proc(states_goals_l[1])

        x: Tensor = self.heur(torch.cat((states_proc, goals_proc), dim=1))

        return x


class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: NDArray[np.uint8]):
        self.colors: NDArray[np.uint8] = colors
        self.hash: Optional[int] = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tobytes())
        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Cube3State):
            return np.array_equal(self.colors, other.colors)
        return NotImplemented


class Cube3Goal(Goal):
    def __init__(self, colors: NDArray[np.uint8]):
        self.colors: NDArray[np.uint8] = colors


class Cube3Action(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self):
        return self.action

    def __eq__(self, other: object):
        if isinstance(other, Cube3Action):
            return self.action == other.action
        return NotImplemented


def _get_adj() -> Dict[int, NDArray[np.int_]]:
    # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
    return {0: np.array([2, 5, 3, 4]),
            1: np.array([2, 4, 3, 5]),
            2: np.array([0, 4, 1, 5]),
            3: np.array([0, 5, 1, 4]),
            4: np.array([0, 3, 1, 2]),
            5: np.array([0, 2, 1, 3])
            }


def _cubelet_to_type(cubelet: List[int]) -> str:
    if len(cubelet) == 1:
        return "center_cbl"
    elif len(cubelet) == 2:
        return "edge_cbl"
    elif len(cubelet) == 3:
        return "corner_cbl"
    else:
        raise ValueError("Unknown cubelet type")


def _colors_to_model(colors: NDArray[np.uint8]) -> Model:
    grnd_atoms: List[Atom] = [('at_idx', x, f"{i}") for i, x in enumerate(colors) if x != 'k']
    return frozenset(grnd_atoms)


class Cube3(EnvGrndAtoms[Cube3State, Cube3Action, Cube3Goal]):
    atomic_actions: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

    def __init__(self, env_name: str):
        super().__init__(env_name)
        self.cube_len: int = 3
        self.colors: List[str] = ['white', 'yellow', 'orange', 'red', 'blue', 'green']
        self.colors_grnd_obj: List[str] = ['w', 'y', 'o', 'r', 'b', 'g']
        self.faces_grnd_obj: List[str] = ['w_f', 'y_f', 'o_f', 'r_f', 'b_f', 'g_f']

        self.color_to_int: Dict[str, int] = dict()
        for i, col_abbrev in enumerate(self.colors_grnd_obj):
            self.color_to_int[col_abbrev] = i

        self.cbs_m_idxs_l: List[List[int]] = [[4], [13], [22], [31], [40], [49]]
        self.cbs_e_idxs_l: List[List[int]] = [[1, 23], [3, 50], [5, 41], [7, 32],
                                              [10, 21], [12, 39], [14, 48], [16, 30],
                                              [19, 43], [25, 46],
                                              [28, 52], [34, 37]]
        self.cbs_c_idxs_l: List[List[int]] = [[0, 26, 47], [2, 20, 44], [6, 29, 53], [8, 35, 38],
                                              [9, 18, 42], [11, 24, 45], [15, 33, 36], [17, 27, 51]]

        self.cbls_all: List[List[int]] = self.cbs_m_idxs_l.copy()
        self.cbls_all = self.cbs_m_idxs_l + self.cbs_e_idxs_l + self.cbs_c_idxs_l

        # all actions
        # for i in range(0, len(self.atomic_actions), 2):
        #    self.action_combs.append([i, i])

        # self.action_combs = action_combs
        self.num_actions = len(self.atomic_actions)
        self.num_stickers: int = 6 * (self.cube_len ** 2)

        # solved state
        self.goal_colors: NDArray[np.uint8] = (np.arange(0, self.num_stickers, 1,
                                                         dtype=np.uint8) // (self.cube_len ** 2)).astype(np.uint8)

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, NDArray[np.int_]]
        self.rotate_idxs_old: Dict[str, NDArray[np.int_]]

        self.adj_faces: Dict[int, NDArray[np.int_]] = _get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.atomic_actions)

        self.int_to_color: NDArray[np.str_] = np.concatenate((np.array(self.colors_grnd_obj), ['k']))  # type: ignore

    def next_state(self, states: List[Cube3State], actions: List[Cube3Action]) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)

        states_next_np = np.zeros(states_np.shape, dtype=np.uint8)
        tcs_np: NDArray[np.float64] = np.zeros(len(states))
        for action in set(actions):
            action_idxs: NDArray[np.int_] = np.array([idx for idx in range(len(actions)) if actions[idx] == action])
            states_np_act = states_np[action_idxs]

            states_next_np_act = self._move_np(states_np_act, action.action)

            tcs_act: List[float] = [1.0 for _ in range(states_np_act.shape[0])]

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs

    def get_state_actions(self, states: List[Cube3State]) -> List[List[Cube3Action]]:
        return [[Cube3Action(x) for x in range(self.num_actions)] for _ in range(len(states))]

    def is_solved(self, states: List[Cube3State], goals: List[Cube3Goal]) -> List[bool]:
        states_np = np.stack([x.colors for x in states], axis=0)
        goals_np = np.stack([x.colors for x in goals], axis=0)
        is_solved_np = np.all(np.logical_or(states_np == goals_np, goals_np == 6), axis=1)
        return list(is_solved_np)

    def states_goals_to_nnet_input(self, states: List[Cube3State], goals: List[Cube3Goal]) -> List[NDArray[np.uint8]]:
        # states
        states_np: NDArray[np.uint8] = np.stack([state.colors for state in states], axis=0).astype(np.uint8)
        goals_np: NDArray[np.uint8] = np.stack([goal.colors for goal in goals], axis=0)
        return [states_np, goals_np]

    def state_to_model(self, states: List[Cube3State]) -> List[Model]:
        states_np = np.stack([x.colors for x in states], axis=0).astype(np.uint8)
        colors_l = self.int_to_color[states_np]
        models: List[Model] = [_colors_to_model(x) for x in colors_l]

        return models

    def model_to_state(self, states_m: List[Model]) -> List[Cube3State]:
        for state_m in states_m:
            assert len(state_m) == self.num_stickers, "model should be fully specified"
        return [Cube3State(x) for x in self._models_to_np(states_m)]

    def goal_to_model(self, goals: List[Cube3Goal]) -> List[Model]:
        goals_np = np.stack([x.colors for x in goals], axis=0).astype(np.uint8)
        colors_l = self.int_to_color[goals_np]
        models: List[Model] = [_colors_to_model(x) for x in colors_l]

        return models

    def model_to_goal(self, models: List[Model]) -> List[Cube3Goal]:
        return [Cube3Goal(x) for x in self._models_to_np(models)]

    def get_v_nnet(self) -> HeurFnNNet:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = Cube3NNet(state_dim, 7, 5000, 1000, 4, 1, True, False, "V")

        return nnet

    def get_q_nnet(self) -> HeurFnNNet:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = Cube3NNet(state_dim, 7, 5000, 1000, 4, self.num_actions, True, False, "Q")

        return nnet

    def get_start_states(self, num_states: int) -> List[Cube3State]:
        assert (num_states > 0)
        backwards_range: Tuple[int, int] = (100, 200)

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_atomic_moves: int = len(self.atomic_actions)

        # Get numpy goal states
        goal_np: NDArray[np.uint8] = np.expand_dims(self.goal_colors.copy(), 0)
        states_np: NDArray[np.uint8] = np.repeat(goal_np, num_states, axis=0)

        # Scrambles
        scramble_nums: NDArray[np.int_] = np.random.choice(scrambs, num_states)
        num_back_moves: NDArray[np.int_] = np.zeros(num_states, dtype=int)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: NDArray[np.int_] = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_atomic_moves, 1))
            idxs = np.random.choice(idxs, subset_size)

            move: int = randrange(num_atomic_moves)
            states_np[idxs] = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states

    def start_state_fixed(self, states: List[Cube3State]) -> List[Model]:
        return [frozenset() for _ in states]

    def get_bk(self) -> List[str]:
        bk: List[str] = []

        clockwise: Dict[str, List[str]] = dict()
        clockwise["w"] = ["b", "r", "g", "o"]
        clockwise["o"] = ["y", "b", "w", "g"]
        clockwise["g"] = ["o", "w", "r", "y"]

        opposite: Dict[str, str] = {"w": "y", "o": "r", "g": "b"}

        bk.append("% Colors")
        for color_obj in self.colors_grnd_obj:
            bk.append(f"color({color_obj})")
        bk.append("dif_col(Col1, Col2) :- color(Col1), color(Col2), Col1 != Col2")

        bk.append("\n")
        for color_idx, (color, color_obj) in enumerate(zip(self.colors, self.colors_grnd_obj)):
            bk.append(f"{color}({color_obj})")

        bk.append("\n% Cubelets")

        center_cbls = [x for x in self.cbls_all if len(x) == 1]
        for cbl_l in center_cbls:
            bk.append(f"center_cbl({self._cubelet_to_name(cbl_l)})")

        edge_cbls = [x for x in self.cbls_all if len(x) == 2]
        for cbl_l in edge_cbls:
            bk.append(f"edge_cbl({self._cubelet_to_name(cbl_l)})")

        corner_cbls = [x for x in self.cbls_all if len(x) == 3]
        for cbl_l in corner_cbls:
            bk.append(f"corner_cbl({self._cubelet_to_name(cbl_l)})")

        if len(center_cbls) > 0:
            bk.append("cubelet(Cbl) :- center_cbl(Cbl)")
        if len(edge_cbls) > 0:
            bk.append("cubelet(Cbl) :- edge_cbl(Cbl)")
        if len(corner_cbls) > 0:
            bk.append("cubelet(Cbl) :- corner_cbl(Cbl)")
        bk.append("dif_cbl(Cbl1, Cbl2) :- cubelet(Cbl1), cubelet(Cbl2), Cbl1 != Cbl2")

        bk.append("\n% has_stk_col\n")
        for cbl_l in self.cbls_all:
            cbl_name: str = self._cubelet_to_name(cbl_l)
            color_objs: List[str] = self._cubelet_to_colors_obj(cbl_l)
            for color_obj in color_objs:
                bk.append(f"has_stk_col({cbl_name}, {color_obj})")

        """
        for cbl_idxs in self.cbs_m_idxs_l + self.cbs_e_idxs_l + self.cbs_c_idxs_l:
            face_names: List[str] = [f"{self.colors_abbrev[idx // (self.cube_len ** 2)]}_f" for idx in cbl_idxs]
            for idx, face_name in zip(cbl_idxs, face_names):
                bk.append(f"onface(Cbl, Col, {face_name}) :- at_idx_cbl(Cbl, Col, {idx})")

        for cbl_idxs in self.cbs_m_idxs_l + self.cbs_e_idxs_l + self.cbs_c_idxs_l:
            for idx_1 in range(len(cbl_idxs)):
                for idx_2 in range(idx_1 + 1, len(cbl_idxs)):
                    bk.append(f":- at_idx_cbl(Cbl0, _, {cbl_idxs[idx_1]}), at_idx_cbl(Cbl1, _, {cbl_idxs[idx_2]}), "
                              f"dif_cbl(Cbl0, Cbl1)")

        for cbl_idxs in self.cbs_m_idxs_l + self.cbs_e_idxs_l + self.cbs_c_idxs_l:
            if len(cbl_idxs) == 1:
                continue
            neq_str: str = ", ".join([f"I != {cbl_idx}" for cbl_idx in cbl_idxs])
            for cbl_idx in cbl_idxs:
                bk.append(f":- at_idx_cbl(Cbl, _, {cbl_idx}), at_idx_cbl(Cbl, _, I), {neq_str}")
        """

        bk.append("\n% Directions")
        for direction in ["cl", "cc", "op"]:
            bk.append(f"direction({direction})")
        bk.append("dif_dir(Dir1, Dir2) :- direction(Dir1), direction(Dir2), Dir1 != Dir2")

        bk.append("")
        bk.extend(["clockwise(cl)", "counterclockwise(cc)", "opposite(op)"])

        bk.append("\n% Faces")
        for color_obj in self.colors_grnd_obj:
            bk.append(f"face({color_obj}_f)")
        bk.append("dif_face(F1, F2) :- face(F1), face(F2), F1 != F2")

        bk.append("\n")
        for color_obj in self.colors_grnd_obj:
            bk.append(f"face_col({color_obj}_f, {color_obj})")

        bk.append("\n")
        for center in clockwise.keys():
            centers_cl: List[str] = clockwise[center]
            for idx, center_cl in enumerate(centers_cl):
                idx_next = (idx + 1) % len(centers_cl)
                bk.append(f"face_rel({center}_f, {center_cl}_f, {centers_cl[idx_next]}_f, cl)")

        for center_op in clockwise.keys():
            center = opposite[center_op]
            centers_cl = clockwise[center_op]
            for idx, center_cl in enumerate(centers_cl):
                idx_next = (idx - 1) % len(centers_cl)
                bk.append(f"face_rel({center}_f, {center_cl}_f, {centers_cl[idx_next]}_f, cl)")
        bk.append("\n")

        bk.append("face_rel(A, B, C, cc) :- face_rel(A, C, B, cl)")
        bk.append("face_rel(A, B, D, op) :- face_rel(A, B, C, cl), face_rel(A, C, D, cl)")
        bk.append("face_rel(A, D, B, op) :- face_rel(A, B, C, cl), face_rel(A, C, D, cl)")
        bk.append("face_adj(F0, F1) :- face_rel(F0, _, F1, _)")

        bk.append("\n% State dependent predicates")
        bk.append("index(0..53)")
        bk.append("-at_idx(Col, I) :- at_idx(Col2, I), color(Col), color(Col2), not Col=Col2")

        bk.append("\n% onface\n")
        for cbl_idxs in self.cbs_m_idxs_l + self.cbs_e_idxs_l + self.cbs_c_idxs_l:
            face_names: List[str] = [f"{self.colors_grnd_obj[idx // (self.cube_len ** 2)]}_f" for idx in cbl_idxs]
            for cbl_idxs2 in self.cbls_all:
                if _cubelet_to_type(cbl_idxs) != _cubelet_to_type(cbl_idxs2):
                    continue
                color_obj_perms = permutations(self._cubelet_to_colors_obj(cbl_idxs2))
                for color_objs_p in color_obj_perms:
                    at_idx_str: str = ", ".join([f"at_idx({col}, {idx})" for col, idx in zip(color_objs_p, cbl_idxs)])
                    for col, face_name in zip(color_objs_p, face_names):
                        bk.append(f"onface({self._cubelet_to_name(cbl_idxs2)}, {col}, {face_name}) :- {at_idx_str}")

        bk.append("stk_f_dir(Cbl, StkCol, FTo, Dir) :- edge_cbl(Cbl), face_rel(FRef, FFrom, FTo, Dir), "
                  "onface(Cbl, StkCol, FFrom), onface(Cbl, _, FRef)")

        bk.append("\n% in_place")
        bk.append("in_place(Cbl) :- edge_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0), "
                  "onface(Cbl, Col1, F1), face_col(F1, Col1), Col0 != Col1")
        bk.append("in_place(Cbl) :- corner_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0), "
                  "onface(Cbl, Col1, F1), face_col(F1, Col1), onface(Cbl, Col2, F2), face_col(F2, Col2), "
                  "Col0 != Col1, Col0 != Col2, Col1 != Col2")
        bk.append("in_place(Cbl) :- center_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0)")

        # bk.append("1 { onface(Cbl, Col1, F1) : face_adj(F0, F1), dif_col(Col0, Col1) } 1 :- "
        #           "edge_cbl(Cbl), onface(Cbl, Col0, F0).\n")
        # bk.append(":- corner_cbl(Cbl), onface(Cbl, Col0, F0), "
        #           "not 2 { onface(Cbl, Col1, F1) : face_adj(F0, F1), dif_col(Col0, Col1) } 2.\n")

        """
        for action in range(self.num_actions):
            bk.append(f"\n% Action {action}")
            # TODO hacky, 0-53 when usually 0-5
            state: Cube3State = Cube3State(np.arange(0, self.num_stickers, 1, dtype=self.dtype))
            state_next: Cube3State = self.next_state([state], [action])[0][0]
            if action < 12:
                act_face: str = self.faces_grnd_obj[action // 2]
                if action % 2 == 0:
                    act_dir: str = "cc"
                else:
                    act_dir: str = "cl"
            else:
                act_face: str = self.faces_grnd_obj[action - 12]
                act_dir: str = "op"
            act_lit_str: str = f"act(T, {act_face}, {act_dir})"

            for color_unique in state.colors:
                idx1: int = np.where(state.colors == color_unique)[0][0]
                idx2: int = np.where(state_next.colors == color_unique)[0][0]
                bk.append(f"at_idx(T+1, Col, {idx2}) :- time(T), at_idx(T, Col, {idx1}), {act_lit_str}")
        """

        bk.append("\n% constraints")
        """
        bk.append("% all center cubelets are always on their respective faces\n")
        for cbl_l in [x for x in self.subgoal_cbls if len(x) == 1]:
            cbl_name: str = self._cubelet_to_name(cbl_l)
            color_obj: str = self._cubelet_to_colors_obj(cbl_l)[0]
            bk.append(f":- not onface({cbl_name}, {color_obj}, {color_obj}_f)")
        """

        bk.append("% different stickers from the same cubelet cannot be on the same face")
        bk.append(":- onface(Cbl, Col0, F), onface(Cbl, Col1, F), dif_col(Col0, Col1)")

        bk.append("\n% cannot have a sticker color from same cubelet be on more than one face")
        bk.append(":- onface(Cbl, Col, F0), onface(Cbl, Col, F1), dif_face(F0, F1)")

        bk.append("\n% cannot say a color of a cubelet is on a face if it does not have the color")
        bk.append(":- onface(Cbl, Col, _), not has_stk_col(Cbl, Col)")

        bk.append("\n% cannot have two stickers from same cbl on opposite faces")
        bk.append(":- onface(Cbl, Col0, F0), onface(Cbl, Col1, F1), face_rel(_, F0, F1, op)")

        bk.append("\n% cannot have two different colors at the same index")
        bk.append(":- at_idx(Col0, I), at_idx(Col1, I), dif_col(Col0, Col1)")

        # bk.append("\n% cannot have two different cubelets at the same index")
        # bk.append(":- at_idx_cbl(Cbl0, _, I), at_idx_cbl(Cbl1, _, I), dif_cbl(Cbl0, Cbl1)")

        """
        bk.append("\n% cannot have stickers from different cubelets on same cubelet loc")
        for cbl in self.subgoal_cbls:
            for cbl_i in cbl:
                for cbl_i2 in cbl:
                    bk.append(f":- at_idx_cbl(Cbl, _, {cbl_i}), at_idx_cbl(CblOth, _, {cbl_i2}), "
                               f"dif_cbl(Cbl, CblOth)\n")

        bk.append("\n% cannot have stickers from same cubelet on different cubelet locs\n")
        for cbl in self.subgoal_cbls:
            for cbl_i in cbl:
                cbl_oths = [x for x in cbl if x != cbl_i]
                if len(cbl) == 2:
                    bk.append(f":- at_idx_cbl(Cbl, _, {cbl_i}), at_idx_cbl(Cbl, _, I), not I={cbl_i}, "
                               f"not I={cbl_oths[0]}\n")
                if len(cbl) == 3:
                    bk.append(f":- at_idx_cbl(Cbl, _, {cbl_i}), at_idx_cbl(Cbl, _, I), not I={cbl_i}, "
                               f"not I={cbl_oths[0]}, not I={cbl_oths[1]}\n")
        """

        # bk.append("count_at_idx(C) :- #count{V0, V1: at_idx(V0,V1) }=C")
        # bk.append("#minimize {C: count_at_idx(C)}")

        bk.append("#show at_idx/2")
        bk.append("#defined at_idx/2")
        bk.append("#defined center_cbl/1")
        bk.append("#defined edge_cbl/1")
        bk.append("#defined corner_cbl/1")

        return bk

    def get_ground_atoms(self) -> List[Atom]:
        ground_atoms: List[Atom] = []
        for color in self.colors_grnd_obj:
            for idx in range(self.num_stickers):
                ground_atoms.append(("at_idx", f"{color}", f"{idx}"))

        # return ["{ at_idx(Col, I) : color(Col), index(I) } 54"]
        return ground_atoms

    def on_model(self, m) -> Model:
        symbs_set: Set[str] = set(str(x) for x in m.symbols(shown=True))
        symbs: List[str] = [misc_utils.remove_all_whitespace(symb) for symb in symbs_set]

        # get atoms
        atoms: List[Atom] = []
        for symb in symbs:
            match = re.search(r"^at_idx\((\S+),(\S+)\)$", symb)
            if match is None:
                continue
            atom: Atom = ("at_idx", match.group(1), match.group(2))
            atoms.append(atom)

        model: Model = frozenset(atoms)
        return model

    def get_pddl_domain(self) -> List[str]:
        pddl_str: str = """
(define (domain cube3)
(:requirements :strips)
(:predicates (at ?c ?i) )"""

        for action in range(self.num_actions):
            action_str_l: List[str] = [f"(:action a{action}", ":effect", "(and"]
            action_name: str = self.atomic_actions[action]
            for idx_old, idx_new in zip(self.rotate_idxs_old[action_name], self.rotate_idxs_new[action_name]):
                action_str_l.append(f"(forall (?x) (when (at ?x i{idx_old}) "
                                    f"(and (not (at ?x i{idx_old})) (at ?x i{idx_new}))))")
                # action_str_l.append(":parameters (?c)")
                # action_str_l.append(f":precondition (at ?c i{idx_old})")
                # action_str_l.append(f":effect (and (not (at ?c i{idx_old})) (at ?c i{idx_new}))")
            action_str_l.append(")")
            action_str_l.append(")")
            action_str: str = '\n'.join(action_str_l)
            pddl_str = f"{pddl_str}\n{action_str}"
        # pddl_str = pddl_str + ("\n(:action A\n"
        #                       ":parameters (?x ?y)\n"
        #                       ":precondition (at ?x ?y)\n"
        #                       "\n:effect (at ?y ?x))")

        pddl_str = pddl_str + "\n)"

        return pddl_str.split("\n")

    def state_goal_to_pddl_inst(self, state: Cube3State, goal: Cube3Goal) -> List[str]:
        model: Model = self.goal_to_model([goal])[0]
        inst_l: List[str] = ["(define(problem cube3inst)", "(:domain cube3)"]
        idx_objects: str = ' '.join([f"i{idx}" for idx in range(self.num_stickers)])
        objects_str: str = f"(:objects {' '.join(self.colors_grnd_obj)} {idx_objects})"
        inst_l.append(objects_str)

        inst_l.append("(:init")
        inst_l.append(' '.join([f'(at {self.int_to_color[color]} i{idx})' for idx, color in enumerate(state.colors)]))
        inst_l.append(")")

        inst_l.append("(:goal")
        inst_l.append("(and")
        inst_l.append(' '.join([f'(at {self.int_to_color[color]} i{idx})' for idx, color
                                in enumerate(self._models_to_np([model])[0]) if color < 6]))
        inst_l.append(")")
        inst_l.append(")")

        inst_l.append(")")
        return inst_l

    def pddl_action_to_action(self, pddl_action: str) -> int:
        match = re.match(r"^a(\d+).*", pddl_action)
        assert match is not None
        return int(match.group(1))

    def visualize(self, states: Union[List[Cube3State], List[Cube3Goal]]) -> NDArray[np.float64]:
        # initialize
        fig = plt.figure(figsize=(.64, .64))
        viz = InteractiveCube(3, self.get_start_states(1)[0].colors)

        fig.add_axes(viz)
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)

        states_img: NDArray[np.float64] = np.zeros((len(states), width, height, 6))
        for state_idx, state in enumerate(states):
            # create image
            if isinstance(state, Cube3State):
                viz.new_state(state.colors)
            elif isinstance(state, Cube3Goal):
                model: Model = self.goal_to_model([state])[0]
                viz.new_state(self._models_to_np([model])[0])
            else:
                raise ValueError(f"Unknown input type {type(state)}")

            viz.set_rot(0)
            canvas.draw()
            image1 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((width, height, 3)) / 255

            viz.set_rot(1)
            canvas.draw()
            image2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((width, height, 3)) / 255

            states_img[state_idx] = np.concatenate((image1, image2), axis=2)

        plt.close(fig)

        return states_img

    def _models_to_np(self, models: List[Model]) -> NDArray[np.uint8]:
        models_np: NDArray[np.uint8] = (np.ones((len(models), self.num_stickers), dtype=np.uint8) * 6).astype(np.uint8)
        for idx, model in enumerate(models):
            for grnd_atom in model:
                models_np[idx, int(grnd_atom[2])] = self.color_to_int[grnd_atom[1]]

        return models_np

    def _cubelet_to_name(self, cubelet: List[int]) -> str:
        face_idxs = np.floor(np.sort(cubelet) // (self.cube_len ** 2)).astype(int)
        cubelet_name = "".join([self.colors_grnd_obj[idx] for idx in face_idxs])
        cubelet_name = cubelet_name + "_c"

        return cubelet_name

    def _cubelet_to_colors_obj(self, cubelet: List[int]) -> List[str]:
        face_idxs = np.floor(np.sort(cubelet) // (self.cube_len ** 2)).astype(int)
        return [self.colors_grnd_obj[idx] for idx in face_idxs]

    def _move_np(self, states_np: NDArray[np.uint8], action: int) -> NDArray[np.uint8]:
        states_next_np: NDArray[np.uint8] = states_np.copy()

        actions = [action]

        for action_part in actions:
            action_str: str = self.atomic_actions[action_part]
            states_next_np[:, self.rotate_idxs_new[action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]

        return states_next_np

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, NDArray[np.int_]], Dict[str, NDArray[np.int_]]]:
        rotate_idxs_new: Dict[str, NDArray[np.int_]] = dict()
        rotate_idxs_old: Dict[str, NDArray[np.int_]] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                            4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                        1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0], 4: [range(0, cube_len), 0],
                            5: [range(0, cube_len), 0]},
                        2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                            4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                        3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                            4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                        4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                            2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                        5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                            2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]}
                        }
            face_dict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [[0, range(0, cube_len)], [range(0, cube_len), cube_len - 1],
                          [cube_len - 1, range(cube_len - 1, -1, -1)], [range(cube_len - 1, -1, -1), 0]]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new: int = int(np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape))
                    flat_idx_old: int = int(np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape))
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))  # type: ignore
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))  # type: ignore

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to: int = int(faces_to[i])
                face_from: int = int(faces_from[i])
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = int(np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape))
                    flat_idx_old = int(np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape))
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))  # type: ignore
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))  # type: ignore

        return rotate_idxs_new, rotate_idxs_old
