from typing import List, Tuple, Union, Set, Optional
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from deepxube.utils import misc_utils
import re
from random import randrange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel
from .environment_abstract import EnvGrndAtoms, State, Action, Goal, HeurFnNNet
from deepxube.logic.logic_objects import Atom, Model
from numpy.typing import NDArray


int_t = Union[np.uint8, np.int_]


class NPState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: NDArray[int_t]):
        self.tiles: NDArray[int_t] = tiles
        self.hash: Optional[int] = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.tiles.tobytes())
        return self.hash

    def __eq__(self, other: object):
        if isinstance(other, NPState):
            return np.array_equal(self.tiles, other.tiles)
        return NotImplemented


class NPGoal(Goal):
    def __init__(self, tiles: NDArray[int_t]):
        self.tiles: NDArray[int_t] = tiles


class NPAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self):
        return self.action

    def __eq__(self, other: object):
        if isinstance(other, NPAction):
            return self.action == other.action
        return NotImplemented


class ProcessStates(nn.Module):
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


class FCResnet(nn.Module):
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


class NNet(HeurFnNNet):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_res_blocks: int,
                 out_dim: int, batch_norm: bool, weight_norm: bool, nnet_type: str):
        super().__init__(nnet_type)
        self.state_proc = ProcessStates(state_dim, one_hot_depth)

        input_dim: int = state_dim * one_hot_depth * 2
        self.heur = FCResnet(input_dim, h1_dim, resnet_dim, num_res_blocks, out_dim, batch_norm, weight_norm)

    def forward(self, states_goals_l: List[Tensor]):
        states_proc = self.state_proc(states_goals_l[0])
        goals_proc = self.state_proc(states_goals_l[1])

        x: Tensor = self.heur(torch.cat((states_proc, goals_proc), dim=1))

        return x


class NPuzzle(EnvGrndAtoms[NPState, NPAction, NPGoal]):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, env_name: str, dim: int):
        super().__init__(env_name)

        self.dim: int = dim
        self.dtype: type
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int_

        self.num_tiles: int = dim ** 2

        # Solved state
        self.goal_tiles: NDArray[int_t] = np.concatenate((np.arange(1, self.dim * self.dim),
                                                          [0])).astype(self.dtype)  # type: ignore

        # Next state ops
        self.swap_zero_idxs: NDArray[int_t] = self._get_swap_zero_idxs(self.dim)

        self.num_actions: int = 4

    def next_state(self, states: List[NPState], actions: List[NPAction]) -> Tuple[List[NPState], List[float]]:
        # initialize
        states_np: NDArray[int_t] = np.stack([x.tiles for x in states], axis=0)
        states_next_np: NDArray[int_t] = states_np.copy()

        # get zero indicies
        z_idxs: NDArray[np.int_]
        _, z_idxs = np.where(states_next_np == 0)

        tcs_np: NDArray[np.float64] = np.zeros(len(states))
        for action in set(actions):
            action_idxs: NDArray[np.int_] = np.array([idx for idx in range(len(actions)) if actions[idx] == action])
            states_np_act = states_np[action_idxs]
            z_idxs_act: NDArray[np.int_] = z_idxs[action_idxs]

            states_next_np_act, _, tcs_act = self._move_np(states_np_act, z_idxs_act, action.action)

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        # make states
        states_next: List[NPState] = [NPState(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs

    def expand(self, states: List[NPState]) -> Tuple[List[List[NPState]], List[List[NPAction]], List[List[float]]]:
        # initialize
        num_states: int = len(states)

        states_exp: List[List[NPState]] = [[] for _ in range(len(states))]
        actions_exp_l: List[List[NPAction]] = [[] for _ in range(len(states))]

        tc: NDArray[np.float64] = np.empty([num_states, self.num_actions])

        # numpy states
        states_np: NDArray[int_t] = np.stack([state.tiles for state in states])

        # Get z_idxs
        z_idxs: NDArray[np.int_]
        _, z_idxs = np.where(states_np == 0)

        # for each move, get next states, transition costs, and if solved
        for action in range(self.num_actions):
            # next state
            states_next_np: NDArray[int_t]
            tc_move: List[float]
            states_next_np, _, tc_move = self._move_np(states_np, z_idxs, action)

            # transition cost
            tc[:, action] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(NPState(states_next_np[idx]))
                actions_exp_l[idx].append(NPAction(action))

        # make lists
        tc_l: List[List[float]] = [list(tc[i]) for i in range(num_states)]

        return states_exp, actions_exp_l, tc_l

    def get_state_actions(self, states: List[NPState]) -> List[List[NPAction]]:
        return [[NPAction(x) for x in range(self.num_actions)] for _ in range(len(states))]

    def is_solved(self, states: List[NPState], goals: List[NPGoal]) -> List[bool]:
        states_np = np.stack([x.tiles for x in states], axis=0)
        goals_np = np.stack([x.tiles for x in goals], axis=0)
        is_solved_np = np.all(np.logical_or(states_np == goals_np, goals_np == self.num_tiles), axis=1)
        return list(is_solved_np)

    def states_goals_to_nnet_input(self, states: List[NPState], goals: List[NPGoal]) -> List[NDArray[int_t]]:
        states_np: NDArray[int_t] = np.stack([x.tiles for x in states], axis=0)
        goals_np: NDArray[int_t] = np.stack([x.tiles for x in goals], axis=0)

        return [states_np.astype(self.dtype), goals_np]

    def state_to_model(self, states: List[NPState]) -> List[Model]:
        states_np: NDArray[int_t] = np.stack([state.tiles for state in states], axis=0).astype(self.dtype)
        states_np = states_np.reshape((-1, self.dim, self.dim))
        models: List[Model] = [self._sqr_tiles_to_model(x) for x in states_np]
        return models

    def model_to_state(self, states_m: List[Model]) -> List[NPState]:
        for state_m in states_m:
            assert len(state_m) == self.num_tiles, "model should be fully specified"
        return [NPState(x) for x in self._models_to_np(states_m)]

    def goal_to_model(self, goals: List[NPGoal]) -> List[Model]:
        goals_np: NDArray[int_t] = np.stack([goal.tiles for goal in goals], axis=0).astype(self.dtype)
        goals_np = goals_np.reshape((-1, self.dim, self.dim))
        models: List[Model] = [self._sqr_tiles_to_model(x) for x in goals_np]
        return models

    def model_to_goal(self, models: List[Model]) -> List[NPGoal]:
        return [NPGoal(x) for x in self._models_to_np(models)]

    def get_v_nnet(self) -> HeurFnNNet:
        nnet = NNet(self.num_tiles, self.num_tiles + 1, 5000, 1000, 4, 1, True, False, "V")
        return nnet

    def get_q_nnet(self) -> HeurFnNNet:
        nnet = NNet(self.num_tiles, self.num_tiles + 1, 5000, 1000, 4, self.num_actions, True, False, "V")
        return nnet

    def get_start_states(self, num_states: int) -> List[NPState]:
        assert (num_states > 0)
        states: List[NPState] = []
        while len(states) < num_states:
            states_np: NDArray[int_t] = np.stack([np.random.permutation(self.num_tiles)
                                                  for _ in range(num_states - len(states))], axis=0).astype(self.dtype)
            is_solvable: NDArray[np.bool_] = self._is_solvable(states_np)

            states.extend([NPState(x) for x in states_np[is_solvable]])

        return states

    def start_state_fixed(self, states: List[NPState]) -> List[Model]:
        return [frozenset() for _ in states]

    def get_pddl_domain(self) -> List[str]:
        pddl_str: str = """
        (define (domain strips-sliding-tile)
  (:requirements :strips)
  (:predicates
   (tile ?x) (position ?x)
   (at ?t ?x ?y) (blank ?x ?y)
   (inc ?p ?pp) (dec ?p ?pp)
   (up ?t) (down ?t) (left ?t) (right ?t)
   )

  (:action move-up
    :parameters (?omf ?px ?py ?by)
    :precondition (and (up ?omf)
        (tile ?omf) (position ?px) (position ?py) (position ?by)
        (dec ?by ?py) (blank ?px ?by) (at ?omf ?px ?py))
    :effect (and (not (blank ?px ?by)) (not (at ?omf ?px ?py))
    (blank ?px ?py) (at ?omf ?px ?by)))

  (:action move-down
    :parameters (?omf ?px ?py ?by)
    :precondition (and (down ?omf)
        (tile ?omf) (position ?px) (position ?py) (position ?by)
        (inc ?by ?py) (blank ?px ?by) (at ?omf ?px ?py))
    :effect (and (not (blank ?px ?by)) (not (at ?omf ?px ?py))
    (blank ?px ?py) (at ?omf ?px ?by)))

  (:action move-left
    :parameters (?omf ?px ?py ?bx)
    :precondition (and (left ?omf)
        (tile ?omf) (position ?px) (position ?py) (position ?bx)
        (dec ?bx ?px) (blank ?bx ?py) (at ?omf ?px ?py))
    :effect (and (not (blank ?bx ?py)) (not (at ?omf ?px ?py))
        (blank ?px ?py) (at ?omf ?bx ?py)))

  (:action move-right
    :parameters (?omf ?px ?py ?bx)
    :precondition (and (right ?omf)
        (tile ?omf) (position ?px) (position ?py) (position ?bx)
        (inc ?bx ?px) (blank ?bx ?py) (at ?omf ?px ?py))
    :effect (and (not (blank ?bx ?py)) (not (at ?omf ?px ?py))
    (blank ?px ?py) (at ?omf ?bx ?py)))
  ) """

        return pddl_str.split("\n")

    def state_goal_to_pddl_inst(self, state: NPState, goal: NPGoal) -> List[str]:
        model: Model = self.goal_to_model([goal])[0]

        # objects
        inst_l: List[str] = ["(define(problem slidingtile)", "(:domain strips-sliding-tile)"]
        tile_names = [f"t{i}" for i in range(1, self.num_tiles)]
        positions = [f"x{i + 1}" for i in range(0, self.dim)] + [f"y{i + 1}" for i in range(0, self.dim)]
        objects: List[str] = tile_names.copy() + positions.copy()

        inst_l.append(f"(:objects {' '.join(objects)})")

        # tiles and positions
        inst_l.append("(:init")
        tile_grnd_atoms: List[str] = [f"(tile {x})" for x in tile_names]
        position_grnd_atoms: List[str] = [f"(position {x})" for x in positions]
        inst_l.append(f"{' '.join(tile_grnd_atoms)}")
        inst_l.append(f"{' '.join(position_grnd_atoms)}")

        # inc and dec
        inc_grnd_atoms: List[str] = []
        for idx in range(self.dim - 1):
            inc_grnd_atoms.append(f"(inc x{idx + 1} x{idx + 2})")
            inc_grnd_atoms.append(f"(inc y{idx + 1} y{idx + 2})")

        dec_grnd_atoms: List[str] = []
        for idx in range(self.dim - 1):
            dec_grnd_atoms.append(f"(dec x{idx + 2} x{idx + 1})")
            dec_grnd_atoms.append(f"(dec y{idx + 2} y{idx + 1})")

        inst_l.append(f"{' '.join(inc_grnd_atoms)}")
        inst_l.append(f"{' '.join(dec_grnd_atoms)}")

        # state
        inst_l.append("")
        tiles_mat = state.tiles.reshape((self.dim, self.dim))
        for idx_y in range(tiles_mat.shape[0]):
            state_lits_row: List[str] = []
            for idx_x in range(tiles_mat.shape[1]):
                tile = tiles_mat[idx_y, idx_x]
                if tile == 0:
                    state_lits_row.append(f"(blank x{idx_x + 1} y{idx_y + 1})")
                else:
                    state_lits_row.append(f"(at t{tile} x{idx_x + 1} y{idx_y + 1})")
            inst_l.append(f"{' '.join(state_lits_row)}")
        inst_l.append("")

        # up, down, left, right
        for direction_name in ["up", "down", "left", "right"]:
            direction_pred_names: List[str] = [f"({direction_name} {x})" for x in tile_names]
            inst_l.append(f"{' '.join(direction_pred_names)}")
        inst_l.append(")")

        # goal
        inst_l.append("(:goal")
        if len(model) > 0:
            inst_l.append("(and")
            tiles_goal_mat = self._models_to_np([model])[0].reshape((self.dim, self.dim))
            for idx_y in range(tiles_goal_mat.shape[0]):
                goal_lits_row: List[str] = []
                for idx_x in range(tiles_goal_mat.shape[1]):
                    tile = tiles_goal_mat[idx_y, idx_x]
                    if tile == self.num_tiles:
                        goal_lits_row.append("                                     ")
                    elif tile == 0:
                        goal_lits_row.append(f"(blank x{idx_x + 1} y{idx_y + 1})")
                    else:
                        goal_lits_row.append(f"(at t{tile} x{idx_x + 1} y{idx_y + 1})")
                inst_l.append(f"{' '.join(goal_lits_row)}")

            inst_l.append(")")
        else:
            inst_l.append("(tile t1)")  # TODO hacky, how to do empty goal in PDDL?

        inst_l.append(")")
        inst_l.append(")")

        return inst_l

    def pddl_action_to_action(self, pddl_action: str) -> int:
        if re.match("^move-up", pddl_action):
            return 1
        elif re.match("^move-down", pddl_action):
            return 0
        elif re.match("^move-left", pddl_action):
            return 3
        elif re.match("^move-right", pddl_action):
            return 2

        raise ValueError(f"Unknown action {pddl_action}")

    def visualize(self, states: Union[List[NPState], List[NPGoal]]) -> NDArray[np.float64]:
        fig = plt.figure(figsize=(.64, .64))
        ax = fig.add_axes((0, 0, 1., 1.))
        # fig = plt.figure(figsize=(.64, .64))
        # ax = fig.gca()
        # fig.add_axes(ax)
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)
        states_img: NDArray[np.float64] = np.zeros((len(states), width, height, 3))
        for state_idx, state in enumerate(states):
            ax.clear()

            state_np: NDArray[int_t]
            if isinstance(state, NPState):
                state_np = state.tiles
            elif isinstance(state, NPGoal):
                model: Model = self.goal_to_model([state])[0]
                state_np = self._models_to_np([model])[0]
            else:
                raise ValueError(f"Unknown input type {type(state)}")

            for square_idx, square in enumerate(state_np):
                color = 'white'
                x_pos = int(np.floor(square_idx / self.dim))
                yPos = square_idx % self.dim

                left = yPos / float(self.dim)
                right = left + 1.0 / float(self.dim)
                top = (self.dim - x_pos - 1) / float(self.dim)
                bottom = top + 1.0 / float(self.dim)

                ax.add_patch(patches.Rectangle((left, top), 1.0 / self.dim, 1.0 / self.dim, linewidth=1,
                                               edgecolor='k', facecolor=color))

                if square != 0:
                    sqr_txt: str
                    if square == (self.dim ** 2):
                        sqr_txt = "-"
                    else:
                        sqr_txt = str(square)
                    ax.text(0.5 * (left + right), 0.5 * (bottom + top), sqr_txt, horizontalalignment='center',
                            verticalalignment='center', fontsize=6, color='black', transform=ax.transAxes)

            canvas.draw()
            states_img[state_idx] = np.frombuffer(canvas.tostring_rgb(),
                                                  dtype='uint8').reshape((width, height, 3)) / 255

        plt.close(fig)
        return states_img

    def get_ground_atoms(self) -> List[Atom]:
        ground_atoms: List[Atom] = []
        for tile_num in range(self.num_tiles):
            for idx_x in range(self.dim):
                for idx_y in range(self.dim):
                    ground_atoms.append(("at_idx", f"t{tile_num}", f"r{idx_x}", f"c{idx_y}"))

        return ground_atoms

    def on_model(self, m) -> Model:
        symbs_set: Set[str] = set(str(x) for x in m.symbols(shown=True))
        symbs: List[str] = [misc_utils.remove_all_whitespace(symb) for symb in symbs_set]

        # get atoms
        atoms: List[Atom] = []
        for symb in symbs:
            match = re.search(r"^at_idx\((\S+),(\S+),(\S+)\)$", symb)
            if match is None:
                continue
            atom: Atom = ("at_idx", match.group(1), match.group(2), match.group(3))
            atoms.append(atom)

        model: Model = frozenset(atoms)
        return model

    def get_bk(self) -> List[str]:
        bk: List[str] = ["%tiles and blanks"]

        for tile_num in range(1, self.num_tiles):
            bk.append(f"tile(t{tile_num})")
        bk.append("blank(t0)")
        bk.append("t_or_b(X) :- tile(X)")
        bk.append("t_or_b(X) :- blank(X)")

        for tile_num in range(0, self.num_tiles):
            bk.append(f"val(t{tile_num}, {tile_num})")
        bk.append(f"num(0..{sum(range(self.num_tiles + 1))})")

        bk.append("")
        bk.append("%rows and columns")
        for idx in range(self.dim):
            bk.append(f"row(r{idx})")
        for idx in range(self.dim):
            bk.append(f"col(c{idx})")
        bk.append("at_row(X, R) :- t_or_b(X), row(R), at_idx(X, R, _)")
        bk.append("at_col(X, C) :- t_or_b(X), col(C), at_idx(X, _, C)")
        bk.append(f"num_row({self.dim})")
        bk.append(f"num_col({self.dim})")

        bk.append("")
        bk.append("% classical negation")
        bk.append("-at_idx(X, R, C) :- t_or_b(X), t_or_b(X2), at_idx(X2, R, C), not X=X2")
        bk.append("-at_idx(X, R, C) :- row(R), col(C), row(R2), at_idx(X, R2, _), not R=R2")
        bk.append("-at_idx(X, R, C) :- row(R), col(C), col(C2), at_idx(X, _, C2), not C=C2")

        bk.append("")
        bk.append("% constraints")
        bk.append("% location cannot have multiple tiles")
        bk.append(":- row(R), col(C), #count{X: at_idx(X, R, C)} > 1")

        bk.append("% tile or blank cannot be in more than one place at a time")
        bk.append(":- t_or_b(X), #count{R, C: at_idx(X, R, C)} > 1")

        return bk

    def _is_solvable(self, states_np: NDArray[int_t]) -> NDArray[np.bool_]:
        num_inversions: NDArray[np.int_] = self._get_num_inversions(states_np)
        num_inversions_is_even: NDArray[np.bool_] = (num_inversions % 2 == 0)
        if self.dim % 2 == 0:
            # even
            _, z_idxs = np.where(states_np == 0)
            z_row_from_bottom_1 = self.dim - np.floor(z_idxs / self.dim)
            z_from_bottom_1_is_even: NDArray[np.bool_] = (z_row_from_bottom_1 % 2 == 0)
            case_1: NDArray[np.bool_] = np.logical_and(z_from_bottom_1_is_even, np.logical_not(num_inversions_is_even))
            case_2: NDArray[np.bool_] = np.logical_and(np.logical_not(z_from_bottom_1_is_even), num_inversions_is_even)
            return np.logical_or(case_1, case_2)
        else:
            # odd
            return num_inversions_is_even

    def _get_num_inversions(self, states_np: NDArray[int_t]) -> NDArray[np.int_]:
        num_inversions: NDArray[np.int_] = np.zeros(states_np.shape[0], dtype=int)
        for idx_1 in range(self.num_tiles):
            for idx_2 in range(idx_1 + 1, self.num_tiles):
                no_zeros: NDArray[np.bool_] = np.logical_and(states_np[:, idx_1] != 0, states_np[:, idx_2] != 0)
                has_inversion: NDArray[np.bool_] = states_np[:, idx_1] > states_np[:, idx_2]
                num_inversions = num_inversions + np.logical_and(no_zeros, has_inversion)

        return num_inversions

    def _sqr_tiles_to_model(self, tiles_sqr: NDArray[int_t]):
        grnd_atoms: List[Atom] = []
        for idx_x in range(tiles_sqr.shape[0]):
            for idx_y in range(tiles_sqr.shape[1]):
                val = tiles_sqr[idx_x, idx_y]
                if val != self.num_tiles:
                    grnd_atoms.append(('at_idx', f"t{tiles_sqr[idx_x, idx_y]}", f"r{str(idx_x)}", f"c{str(idx_y)}"))

        return frozenset(grnd_atoms)

    def _models_to_np(self, models: List[Model]) -> NDArray[int_t]:
        models_np: NDArray[int_t] = (np.ones((len(models), self.dim, self.dim)) * self.num_tiles).astype(self.dtype)
        for idx, model in enumerate(models):
            for grnd_atom in model:
                models_np[idx, int(grnd_atom[2][1:]), int(grnd_atom[3][1:])] = int(grnd_atom[1][1:])

        return models_np.reshape((len(models), -1))

    def _random_walk(self, states: List[NPState], num_steps_l: List[int]) -> List[NPState]:
        states_np = np.stack([x.tiles for x in states], axis=0)

        # Get z_idxs
        z_idxs: NDArray[np.int_]
        _, z_idxs = np.where(states_np == 0)

        # Scrambles
        num_steps_np: NDArray[np.int_] = np.array(num_steps_l)
        num_actions: NDArray[np.int_] = np.zeros(len(states), dtype=int)

        # go backward from goal state
        while int(np.max(num_actions < num_steps_np)) > 0:
            idxs: NDArray[np.int_] = np.where((num_actions < num_steps_np))[0]
            subset_size: int = int(max(len(idxs) / self.num_actions, 1))
            idxs = np.random.choice(idxs, subset_size)

            move: int = randrange(self.num_actions)
            states_np[idxs], z_idxs[idxs], _ = self._move_np(states_np[idxs], z_idxs[idxs], move)

            num_actions[idxs] = num_actions[idxs] + 1

        return [NPState(x) for x in states_np]

    def _get_swap_zero_idxs(self, n: int) -> NDArray[int_t]:
        swap_zero_idxs: NDArray[int_t] = np.zeros((n ** 2, len(NPuzzle.moves)), dtype=self.dtype)
        for moveIdx, move in enumerate(NPuzzle.moves):
            for i in range(n):
                for j in range(n):
                    z_idx = np.ravel_multi_index((i, j), (n, n))

                    state = np.ones((n, n), dtype=int)
                    state[i, j] = 0

                    is_eligible: bool = False
                    if move == 'U':
                        is_eligible = i < (n - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (n - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i: int = -1
                        swap_j: int = -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        swap_zero_idxs[z_idx, moveIdx] = np.ravel_multi_index((swap_i, swap_j), (n, n))
                    else:
                        swap_zero_idxs[z_idx, moveIdx] = z_idx

        return swap_zero_idxs

    def _move_np(self, states_np: NDArray[int_t], z_idxs: NDArray[np.int_],
                 action: int) -> Tuple[NDArray[int_t], NDArray[int_t], List[float]]:
        states_next_np: NDArray[int_t] = states_np.copy()

        # get index to swap with zero
        state_idxs: NDArray[np.int_] = np.arange(0, states_next_np.shape[0]).astype(int)
        swap_z_idxs: NDArray[int_t] = self.swap_zero_idxs[z_idxs, action]

        # swap zero with adjacent tile
        states_next_np[state_idxs, z_idxs] = states_np[state_idxs, swap_z_idxs]
        states_next_np[state_idxs, swap_z_idxs] = 0

        # transition costs
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, swap_z_idxs, transition_costs
