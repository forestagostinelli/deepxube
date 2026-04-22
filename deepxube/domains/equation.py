"""Symbolic Equation Simplification domain for DeepXube.

States are polynomial expressions represented as lists of (coefficient, degree)
terms. The goal is the canonical form: terms sorted by descending degree,
no duplicate degrees, no zero coefficients, active terms packed to the front.

Actions: combine same-degree terms, swap adjacent terms, remove zero terms.
Training data: reverse walks from canonical forms by splitting/shuffling terms.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
import matplotlib.patches as patches
from matplotlib.figure import Figure

from deepxube.base.factory import Parser
from deepxube.base.domain import (
    State, Action, Goal, ActsEnumFixed,
    GoalStartRevWalkable,
    StateGoalVizable, StringToAct,
)
from deepxube.base.nnet_input import HasFlatSGIn
from deepxube.factories.domain_factory import domain_factory


COEFF_MIN, COEFF_MAX = -9, 9
COEFF_SHIFT = -COEFF_MIN
COEFF_OH_DEPTH = COEFF_MAX - COEFF_MIN + 1

ACT_COMBINE = 0
ACT_SWAP = 1
ACT_REMOVE = 2

GARNET = "#73000A"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
HORSESHOE = "#65780B"
HONEYCOMB = "#A49137"
ROSE = "#CC2E40"
BLACK_90 = "#363636"
BLACK_70 = "#5C5C5C"
BLACK_50 = "#A2A2A2"
BLACK_30 = "#C7C7C7"
BLACK_10 = "#ECECEC"
GRASS = "#CED318"
SANDSTORM = "#FFF2E3"

TERM_COLORS = [GARNET, ATLANTIC, CONGAREE, HORSESHOE, HONEYCOMB, ROSE, BLACK_70, "#676156"]


def _is_active(terms: NDArray[np.int8], i: int, empty_deg: int) -> bool:
    return int(terms[i, 1]) != empty_deg


def _term_label(c: int, d: int) -> str:
    if d == 0:
        return str(c)
    var = "x" if d == 1 else f"x^{d}"
    if c == 1:
        return var
    if c == -1:
        return f"-{var}"
    return f"{c}{var}"


def _terms_to_string(terms: NDArray[np.int8], max_terms: int, empty_deg: int) -> str:
    parts: List[str] = []
    for i in range(max_terms):
        if not _is_active(terms, i, empty_deg):
            continue
        c, d = int(terms[i, 0]), int(terms[i, 1])
        parts.append(_term_label(c, d))
    if not parts:
        return "0"
    result = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            result += f" - {p[1:]}"
        else:
            result += f" + {p}"
    return result


class EqState(State):
    __slots__ = ["terms", "_hash"]

    def __init__(self, terms: NDArray[np.int8]):
        self.terms: NDArray[np.int8] = terms
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.terms.tobytes())
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EqState):
            return np.array_equal(self.terms, other.terms)
        return NotImplemented


class EqGoal(Goal):
    def __init__(self, terms: NDArray[np.int8]):
        self.terms: NDArray[np.int8] = terms


class EqAction(Action):
    def __init__(self, action_id: int, action_type: int, i: int, j: int = 0):
        self.action_id = action_id
        self.action_type = action_type
        self.i = i
        self.j = j

    def __hash__(self) -> int:
        return self.action_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EqAction):
            return self.action_id == other.action_id
        return NotImplemented

    def __repr__(self) -> str:
        if self.action_type == ACT_COMBINE:
            return f"combine({self.i},{self.j})"
        if self.action_type == ACT_SWAP:
            return f"swap({self.i})"
        return f"remove({self.i})"


@domain_factory.register_class("equation")
class EquationSimplification(
    ActsEnumFixed[EqState, EqAction, EqGoal],
    GoalStartRevWalkable[EqState, EqAction, EqGoal],
    HasFlatSGIn[EqState, EqAction, EqGoal],
    StateGoalVizable[EqState, EqAction, EqGoal],
    StringToAct[EqState, EqAction, EqGoal],
):

    def __init__(self, max_degree: int = 4, max_terms: int = 8):
        super().__init__()
        assert max_terms >= max_degree + 1
        self.max_degree: int = max_degree
        self.max_terms: int = max_terms
        self.empty_deg: int = max_degree + 1
        self.deg_oh_depth: int = self.empty_deg + 1

        self.actions_fixed: List[EqAction] = []
        aid = 0
        for i in range(max_terms):
            for j in range(i + 1, max_terms):
                self.actions_fixed.append(EqAction(aid, ACT_COMBINE, i, j))
                aid += 1
        self.num_combine = aid
        for i in range(max_terms - 1):
            self.actions_fixed.append(EqAction(aid, ACT_SWAP, i))
            aid += 1
        for i in range(max_terms):
            self.actions_fixed.append(EqAction(aid, ACT_REMOVE, i))
            aid += 1

    def _empty_terms(self) -> NDArray[np.int8]:
        t = np.zeros((self.max_terms, 2), dtype=np.int8)
        t[:, 1] = self.empty_deg
        return t

    def _random_canonical(self) -> NDArray[np.int8]:
        t = self._empty_terms()
        deg = np.random.randint(0, self.max_degree + 1)
        for i in range(deg + 1):
            c = 0
            while c == 0:
                c = np.random.randint(COEFF_MIN, COEFF_MAX + 1)
            t[i] = [c, deg - i]
        return t

    # ── ActsEnumFixed ────────────────────────────────────────────────────

    def get_actions_fixed(self) -> List[EqAction]:
        return self.actions_fixed.copy()

    def get_state_actions(self, states: List[EqState]) -> List[List[EqAction]]:
        result: List[List[EqAction]] = []
        ed = self.empty_deg
        for state in states:
            valid: List[EqAction] = []
            t = state.terms
            for act in self.actions_fixed:
                if act.action_type == ACT_COMBINE:
                    i, j = act.i, act.j
                    if (_is_active(t, i, ed) and _is_active(t, j, ed)
                            and t[i, 1] == t[j, 1]):
                        s = int(t[i, 0]) + int(t[j, 0])
                        if COEFF_MIN <= s <= COEFF_MAX:
                            valid.append(act)
                elif act.action_type == ACT_SWAP:
                    if not np.array_equal(t[act.i], t[act.i + 1]):
                        valid.append(act)
                elif act.action_type == ACT_REMOVE:
                    if _is_active(t, act.i, ed) and t[act.i, 0] == 0:
                        valid.append(act)
            result.append(valid if valid else [self.actions_fixed[0]])
        return result

    # ── Core domain methods ──────────────────────────────────────────────

    def next_state(self, states: List[EqState], actions: List[EqAction]) -> Tuple[List[EqState], List[float]]:
        new_states: List[EqState] = []
        tcs: List[float] = []
        ed = self.empty_deg
        for state, act in zip(states, actions):
            t = state.terms.copy()
            if act.action_type == ACT_COMBINE:
                i, j = act.i, act.j
                if (_is_active(t, i, ed) and _is_active(t, j, ed)
                        and t[i, 1] == t[j, 1]):
                    s = int(t[i, 0]) + int(t[j, 0])
                    if COEFF_MIN <= s <= COEFF_MAX:
                        t[i, 0] = s
                        t[j] = [0, ed]
            elif act.action_type == ACT_SWAP:
                tmp = t[act.i].copy()
                t[act.i] = t[act.i + 1]
                t[act.i + 1] = tmp
            elif act.action_type == ACT_REMOVE:
                if _is_active(t, act.i, ed) and t[act.i, 0] == 0:
                    t[act.i] = [0, ed]
            new_states.append(EqState(t))
            tcs.append(1.0)
        return new_states, tcs

    def is_solved(self, states: List[EqState], goals: List[EqGoal]) -> List[bool]:
        return [np.array_equal(s.terms, g.terms) for s, g in zip(states, goals)]

    # ── Problem instance generation ──────────────────────────────────────

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[EqState], List[EqGoal]]:
        states: List[EqState] = []
        goals: List[EqGoal] = []
        for _ in range(num):
            t = self._random_canonical()
            states.append(EqState(t.copy()))
            goals.append(EqGoal(t.copy()))
        return states, goals

    def random_walk_rev(self, states: List[EqState], num_steps_l: List[int]) -> List[EqState]:
        result: List[EqState] = []
        ed = self.empty_deg
        for state, n_steps in zip(states, num_steps_l):
            t = state.terms.copy()
            for _ in range(n_steps):
                op = np.random.choice(3, p=[0.5, 0.3, 0.2])
                if op == 0:
                    active = [i for i in range(self.max_terms)
                              if _is_active(t, i, ed) and abs(int(t[i, 0])) > 1]
                    empty = [i for i in range(self.max_terms)
                             if not _is_active(t, i, ed)]
                    if active and empty:
                        idx = int(np.random.choice(active))
                        slot = int(np.random.choice(empty))
                        c = int(t[idx, 0])
                        if c > 0:
                            sc = np.random.randint(1, c)
                        else:
                            sc = np.random.randint(c + 1, 0)
                        t[slot] = [c - sc, t[idx, 1]]
                        t[idx, 0] = sc
                elif op == 1:
                    i = np.random.randint(0, self.max_terms - 1)
                    tmp = t[i].copy()
                    t[i] = t[i + 1]
                    t[i + 1] = tmp
                else:
                    empty = [i for i in range(self.max_terms)
                             if not _is_active(t, i, ed)]
                    if empty:
                        slot = int(np.random.choice(empty))
                        deg = np.random.randint(0, self.max_degree + 1)
                        t[slot] = [0, deg]
            result.append(EqState(t))
        return result

    # ── Neural network input ─────────────────────────────────────────────

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        mt = self.max_terms
        doh = self.deg_oh_depth
        return ([mt, mt, mt, mt], [COEFF_OH_DEPTH, doh, COEFF_OH_DEPTH, doh])

    def to_np_flat_sg(self, states: List[EqState], goals: List[EqGoal]) -> List[NDArray]:
        s_c = np.stack([s.terms[:, 0] for s in states]).astype(np.int64) + COEFF_SHIFT
        s_d = np.stack([s.terms[:, 1] for s in states]).astype(np.int64)
        g_c = np.stack([g.terms[:, 0] for g in goals]).astype(np.int64) + COEFF_SHIFT
        g_d = np.stack([g.terms[:, 1] for g in goals]).astype(np.int64)
        return [s_c, s_d, g_c, g_d]

    # ── Visualization ────────────────────────────────────────────────────

    def visualize_state_goal(self, state: EqState, goal: EqGoal, fig: Figure) -> None:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SANDSTORM)

        mt, ed = self.max_terms, self.empty_deg
        bw = min(0.10, 0.85 / mt)
        bh = 0.10
        gap = 0.005
        sx = 0.5 - mt * (bw + gap) / 2

        ax.text(0.5, 0.93, "Current Expression", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLACK_90)
        for i in range(mt):
            x = sx + i * (bw + gap)
            y = 0.75
            if _is_active(state.terms, i, ed):
                c, d = int(state.terms[i, 0]), int(state.terms[i, 1])
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=TERM_COLORS[i % len(TERM_COLORS)],
                    edgecolor="k", linewidth=0.8))
                ax.text(x + bw / 2, y + bh / 2, _term_label(c, d),
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            else:
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=BLACK_10,
                    edgecolor=BLACK_30, linewidth=0.5))
            ax.text(x + bw / 2, y - 0.025, str(i), ha="center", va="center",
                    fontsize=5, color=BLACK_50)

        expr = _terms_to_string(state.terms, mt, ed)
        ax.text(0.5, 0.64, f"= {expr}", ha="center", va="center",
                fontsize=10, color=BLACK_90, style="italic")

        ax.text(0.5, 0.50, "Goal (Canonical Form)", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLACK_90)
        for i in range(mt):
            x = sx + i * (bw + gap)
            y = 0.32
            if _is_active(goal.terms, i, ed):
                c, d = int(goal.terms[i, 0]), int(goal.terms[i, 1])
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=GRASS,
                    edgecolor="k", linewidth=0.8))
                ax.text(x + bw / 2, y + bh / 2, _term_label(c, d),
                        ha="center", va="center", fontsize=6.5,
                        color=BLACK_90, fontweight="bold")
            else:
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=BLACK_10,
                    edgecolor=BLACK_30, linewidth=0.5))

        goal_str = _terms_to_string(goal.terms, mt, ed)
        ax.text(0.5, 0.21, f"= {goal_str}", ha="center", va="center",
                fontsize=10, color=BLACK_90, style="italic")

        solved = np.array_equal(state.terms, goal.terms)
        ax.text(0.5, 0.07, "SOLVED" if solved else "UNSOLVED",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=GRASS if solved else GARNET)
        fig.canvas.draw()

    # ── StringToAct ──────────────────────────────────────────────────────

    def string_to_action(self, act_str: str) -> Optional[EqAction]:
        try:
            parts = act_str.strip().split()
            cmd = parts[0].lower()
            if cmd == "c" and len(parts) == 3:
                i, j = int(parts[1]), int(parts[2])
                if i > j:
                    i, j = j, i
                for a in self.actions_fixed:
                    if a.action_type == ACT_COMBINE and a.i == i and a.j == j:
                        return a
            elif cmd == "s" and len(parts) == 2:
                i = int(parts[1])
                for a in self.actions_fixed:
                    if a.action_type == ACT_SWAP and a.i == i:
                        return a
            elif cmd == "r" and len(parts) == 2:
                i = int(parts[1])
                for a in self.actions_fixed:
                    if a.action_type == ACT_REMOVE and a.i == i:
                        return a
        except (ValueError, IndexError):
            pass
        return None

    def string_to_action_help(self) -> str:
        return ("'c i j' combine terms i,j  |  "
                "'s i' swap i,i+1  |  "
                "'r i' remove zero term i  (0-indexed)")

    def __repr__(self) -> str:
        return f"EquationSimplification(max_degree={self.max_degree}, max_terms={self.max_terms})"


@domain_factory.register_parser("equation")
class EqParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        parts = args_str.split(".")
        if len(parts) == 1:
            return {"max_degree": int(parts[0])}
        if len(parts) == 2:
            return {"max_degree": int(parts[0]), "max_terms": int(parts[1])}
        raise ValueError(f"Expected 'max_degree' or 'max_degree.max_terms', got '{args_str}'")

    def help(self) -> str:
        return "max_degree[.max_terms]. E.g. 'equation.4' or 'equation.4.8'"
