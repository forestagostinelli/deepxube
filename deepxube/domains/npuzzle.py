from typing import List, Tuple, Union, Optional, Dict, Any

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, GoalStartRevWalkableActsRev, StateGoalVizable, StringToAct
from deepxube.factories.domain_factory import domain_factory
from deepxube.base.nnet_input import HasFlatSGIn
import numpy as np
from random import randrange
import matplotlib.patches as patches
from matplotlib.figure import Figure

from numpy.typing import NDArray


int_t = Union[np.uint8, np.int_]


class NPState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: NDArray[int_t]):
        self.tiles: NDArray[int_t] = tiles
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(self.tiles.tobytes())
        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NPState):
            return np.array_equal(self.tiles, other.tiles)
        return NotImplemented


class NPGoal(Goal):
    def __init__(self, tiles: NDArray[int_t]):
        self.tiles: NDArray[int_t] = tiles


class NPAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NPAction):
            return self.action == other.action
        return NotImplemented


@domain_factory.register_class("npuzzle")
class NPuzzle(ActsEnumFixed[NPState, NPAction, NPGoal], GoalStartRevWalkableActsRev[NPState, NPAction, NPGoal], HasFlatSGIn[NPState, NPAction, NPGoal],
              StateGoalVizable[NPState, NPAction, NPGoal], StringToAct[NPState, NPAction, NPGoal]):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, dim: int = 4):
        super().__init__()

        self.dim: int = dim
        self.dtype: type
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int_

        self.num_tiles: int = dim ** 2

        # Solved state
        self.goal_tiles: NDArray[int_t] = np.concatenate((np.arange(1, self.dim * self.dim), [0])).astype(self.dtype)

        # Next state ops
        self.swap_zero_idxs: NDArray[int_t] = self._get_swap_zero_idxs(self.dim)

        self.num_actions: int = 4
        self.actions: List[NPAction] = [NPAction(x) for x in range(self.num_actions)]

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[NPState], List[NPGoal]]:
        states_goal: List[NPState] = [NPState(self.goal_tiles.copy())] * num
        goals: List[NPGoal] = [NPGoal(self.goal_tiles.copy())] * num

        return states_goal, goals

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

    def get_actions_fixed(self) -> List[NPAction]:
        return self.actions.copy()

    def rev_action(self, states: List[NPState], actions: List[NPAction]) -> List[NPAction]:
        actions_rev: List[NPAction] = []
        for action in actions:
            action_val: int = action.action
            action_val_rev: int
            if action_val % 2 == 0:
                action_val_rev = action_val + 1
            else:
                action_val_rev = action_val - 1
            actions_rev.append(NPAction(action_val_rev))

        return actions_rev

    def is_solved(self, states: List[NPState], goals: List[NPGoal]) -> List[bool]:
        states_np = np.stack([x.tiles for x in states], axis=0)
        goals_np = np.stack([x.tiles for x in goals], axis=0)
        is_solved_np = np.all(np.logical_or(states_np == goals_np, goals_np == self.num_tiles), axis=1)
        return list(is_solved_np)

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [self.num_tiles], [self.num_tiles]

    def to_np_flat_sg(self, states: List[NPState], goals: List[NPGoal]) -> List[NDArray]:
        return [np.stack([x.tiles for x in states], axis=0).astype(self.dtype)]

    def visualize_state_goal(self, state: NPState, goal: NPGoal, fig: Figure) -> None:
        ax = fig.add_subplot(111)
        # fig = plt.figure(figsize=(.64, .64))
        # ax = fig.gca()
        # fig.add_axes(ax)

        state_np: NDArray[int_t] = state.tiles

        for square_idx, square in enumerate(state_np):
            color = 'white'
            x_pos = int(np.floor(square_idx / self.dim))
            y_pos = square_idx % self.dim

            left = y_pos / float(self.dim)
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
                        verticalalignment='center', fontsize=12, color='black', transform=ax.transAxes)

        fig.canvas.draw()

    def string_to_action(self, act_str: str) -> Optional[NPAction]:
        act_str_to_act: Dict[str, NPAction] = {"w": NPAction(0), "s": NPAction(1), "a": NPAction(2), "d": NPAction(3)}
        return act_str_to_act[act_str.lower()]

    def string_to_action_help(self) -> str:
        return "swap blank tile up/down/left/right: w,s,a,d"

    def _is_solvable(self, states_np: NDArray[int_t]) -> NDArray[np.bool_]:
        num_inversions: NDArray[np.int_] = self._get_num_inversions(states_np)
        num_inversions_is_even: NDArray[np.bool_] = np.array(num_inversions % 2 == 0)
        if self.dim % 2 == 0:
            # even
            _, z_idxs = np.where(states_np == 0)
            z_row_from_bottom_1 = self.dim - np.floor(z_idxs / self.dim)
            z_from_bottom_1_is_even: NDArray[np.bool_] = np.array(z_row_from_bottom_1 % 2 == 0)
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

    def random_walk(self, states: List[NPState], num_steps_l: List[int]) -> Tuple[List[NPState], List[float]]:
        states_np = np.stack([x.tiles for x in states], axis=0)
        path_costs: List[float] = [0.0 for _ in states]

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
            states_np[idxs], z_idxs[idxs], tcs = self._move_np(states_np[idxs], z_idxs[idxs], move)

            idx: int
            for move_idx, idx in enumerate(idxs):
                path_costs[idx] += tcs[move_idx]

            num_actions[idxs] = num_actions[idxs] + 1

        return [NPState(x) for x in states_np], path_costs

    def _get_swap_zero_idxs(self, n: int) -> NDArray[int_t]:
        swap_zero_idxs: NDArray[int_t] = np.zeros((n ** 2, len(self.moves)), dtype=self.dtype)
        for moveIdx, move in enumerate(self.moves):
            for i in range(n):
                for j in range(n):
                    z_idx = np.ravel_multi_index((i, j), (n, n))

                    state: NDArray = np.ones((n, n), dtype=int)
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

    def __repr__(self) -> str:
        return f"NPuzzle(dim={self.dim})"


@domain_factory.register_parser("npuzzle")
class GridParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"dim": int(args_str)}

    def help(self) -> str:
        return "An integer for the dimension. E.g. 'npuzzle.6'"
