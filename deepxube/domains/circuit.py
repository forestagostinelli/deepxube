"""Boolean Circuit Minimization domain for DeepXube.

States are Sum-of-Products (SOP) boolean expressions over N variables.
Each product term (implicant) is stored as a (mask, polarity) pair where
mask indicates which variables appear and polarity indicates their sign.
The goal is the minimal SOP form (Quine-McCluskey) for the same truth table.

Actions: combine adjacent terms, absorb redundant terms, remove constant-zero
terms, swap adjacent terms. Reverse walks split terms and insert redundancy.
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


ACT_COMBINE = 0
ACT_ABSORB = 1
ACT_REMOVE = 2
ACT_SWAP = 3

GARNET = "#73000A"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
HORSESHOE = "#65780B"
HONEYCOMB = "#A49137"
ROSE = "#CC2E40"
BLACK_90 = "#363636"
BLACK_50 = "#A2A2A2"
BLACK_30 = "#C7C7C7"
BLACK_10 = "#ECECEC"
GRASS = "#CED318"
SANDSTORM = "#FFF2E3"

TERM_COLORS = [GARNET, ATLANTIC, CONGAREE, HORSESHOE, HONEYCOMB, ROSE, "#5C5C5C", "#676156"]


def _term_truth_table(mask: int, pol: int, num_vars: int) -> int:
    tt = 0
    for assignment in range(1 << num_vars):
        match = True
        for v in range(num_vars):
            if mask & (1 << v):
                var_val = (assignment >> v) & 1
                pol_val = (pol >> v) & 1
                if var_val != pol_val:
                    match = False
                    break
        if match:
            tt |= (1 << assignment)
    return tt


def _sop_truth_table(terms: NDArray[np.int8], max_terms: int,
                     empty_mask: int, num_vars: int) -> int:
    tt = 0
    for i in range(max_terms):
        m = int(terms[i, 0])
        if m == empty_mask:
            continue
        tt |= _term_truth_table(m, int(terms[i, 1]), num_vars)
    return tt


def _term_implies(m1: int, p1: int, m2: int, p2: int) -> bool:
    if m1 & m2 != m1:
        return False
    return (p1 & m1) == (p2 & m1)


def _term_label(mask: int, pol: int, num_vars: int) -> str:
    if mask == 0:
        return "1"
    parts: List[str] = []
    for v in range(num_vars):
        if mask & (1 << v):
            name = chr(ord("a") + v)
            if pol & (1 << v):
                parts.append(name)
            else:
                parts.append(f"{name}'")
    return "".join(parts)


def _sop_to_string(terms: NDArray[np.int8], max_terms: int,
                   empty_mask: int, num_vars: int) -> str:
    parts: List[str] = []
    for i in range(max_terms):
        m = int(terms[i, 0])
        if m == empty_mask:
            continue
        parts.append(_term_label(m, int(terms[i, 1]), num_vars))
    if not parts:
        return "0"
    return " + ".join(parts)


def _quine_mccluskey(tt: int, num_vars: int) -> List[Tuple[int, int]]:
    if tt == 0:
        return []
    if tt == (1 << (1 << num_vars)) - 1:
        return [(0, 0)]

    minterms = [i for i in range(1 << num_vars) if tt & (1 << i)]

    implicants = set()
    for mt in minterms:
        implicants.add((mt, 0))

    changed = True
    while changed:
        changed = False
        new_impl = set()
        used = set()
        impl_list = sorted(implicants)
        for idx1 in range(len(impl_list)):
            for idx2 in range(idx1 + 1, len(impl_list)):
                val1, dc1 = impl_list[idx1]
                val2, dc2 = impl_list[idx2]
                if dc1 != dc2:
                    continue
                diff = val1 ^ val2
                if diff and (diff & (diff - 1)) == 0 and (diff & dc1) == 0:
                    new_impl.add((val1 & val2, dc1 | diff))
                    used.add(impl_list[idx1])
                    used.add(impl_list[idx2])
                    changed = True
        for imp in impl_list:
            if imp not in used:
                new_impl.add(imp)
        implicants = new_impl

    prime_list = sorted(implicants)
    coverage: Dict[int, List[int]] = {mt: [] for mt in minterms}
    for pi_idx, (val, dc) in enumerate(prime_list):
        for mt in minterms:
            if (mt & ~dc) == (val & ~dc):
                coverage[mt].append(pi_idx)

    selected = set()
    remaining = set(minterms)
    while remaining:
        for mt in list(remaining):
            covers = [pi for pi in coverage[mt] if pi not in selected]
            if len(covers) == 1:
                selected.add(covers[0])
                val, dc = prime_list[covers[0]]
                to_remove = []
                for rmt in remaining:
                    if (rmt & ~dc) == (val & ~dc):
                        to_remove.append(rmt)
                remaining -= set(to_remove)
                break
        else:
            best_pi = -1
            best_count = -1
            for pi_idx, (val, dc) in enumerate(prime_list):
                if pi_idx in selected:
                    continue
                count = sum(1 for rmt in remaining if (rmt & ~dc) == (val & ~dc))
                if count > best_count:
                    best_count = count
                    best_pi = pi_idx
            if best_pi >= 0:
                selected.add(best_pi)
                val, dc = prime_list[best_pi]
                to_remove = []
                for rmt in remaining:
                    if (rmt & ~dc) == (val & ~dc):
                        to_remove.append(rmt)
                remaining -= set(to_remove)

    result: List[Tuple[int, int]] = []
    for pi_idx in sorted(selected):
        val, dc = prime_list[pi_idx]
        mask = ((1 << num_vars) - 1) & ~dc
        pol = val & mask
        result.append((mask, pol))
    return result


class CircState(State):
    __slots__ = ["terms", "_hash"]

    def __init__(self, terms: NDArray[np.int8]):
        self.terms: NDArray[np.int8] = terms
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.terms.tobytes())
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircState):
            return np.array_equal(self.terms, other.terms)
        return NotImplemented


class CircGoal(Goal):
    def __init__(self, terms: NDArray[np.int8]):
        self.terms: NDArray[np.int8] = terms


class CircAction(Action):
    def __init__(self, action_id: int, action_type: int, i: int, j: int = 0):
        self.action_id = action_id
        self.action_type = action_type
        self.i = i
        self.j = j

    def __hash__(self) -> int:
        return self.action_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircAction):
            return self.action_id == other.action_id
        return NotImplemented

    def __repr__(self) -> str:
        if self.action_type == ACT_COMBINE:
            return f"combine({self.i},{self.j})"
        if self.action_type == ACT_ABSORB:
            return f"absorb({self.i},{self.j})"
        if self.action_type == ACT_REMOVE:
            return f"remove({self.i})"
        return f"swap({self.i})"


@domain_factory.register_class("circuit")
class CircuitMinimization(
    ActsEnumFixed[CircState, CircAction, CircGoal],
    GoalStartRevWalkable[CircState, CircAction, CircGoal],
    HasFlatSGIn[CircState, CircAction, CircGoal],
    StateGoalVizable[CircState, CircAction, CircGoal],
    StringToAct[CircState, CircAction, CircGoal],
):

    def __init__(self, num_vars: int = 3, max_terms: int = 8):
        super().__init__()
        self.num_vars: int = num_vars
        self.max_terms: int = max_terms
        self.empty_mask: int = (1 << num_vars)
        self.mask_oh_depth: int = self.empty_mask + 1
        self.pol_oh_depth: int = self.empty_mask

        self.actions_fixed: List[CircAction] = []
        aid = 0
        for i in range(max_terms):
            for j in range(i + 1, max_terms):
                self.actions_fixed.append(CircAction(aid, ACT_COMBINE, i, j))
                aid += 1
        for i in range(max_terms):
            for j in range(max_terms):
                if i != j:
                    self.actions_fixed.append(CircAction(aid, ACT_ABSORB, i, j))
                    aid += 1
        for i in range(max_terms):
            self.actions_fixed.append(CircAction(aid, ACT_REMOVE, i))
            aid += 1
        for i in range(max_terms - 1):
            self.actions_fixed.append(CircAction(aid, ACT_SWAP, i))
            aid += 1

    def _empty_terms(self) -> NDArray[np.int8]:
        t = np.zeros((self.max_terms, 2), dtype=np.int8)
        t[:, 0] = self.empty_mask
        return t

    def _is_active(self, terms: NDArray[np.int8], i: int) -> bool:
        return int(terms[i, 0]) != self.empty_mask

    def _random_canonical(self) -> NDArray[np.int8]:
        nv = self.num_vars
        while True:
            tt = np.random.randint(1, (1 << (1 << nv)))
            if tt == (1 << (1 << nv)) - 1:
                continue
            primes = _quine_mccluskey(tt, nv)
            if 1 <= len(primes) <= self.max_terms:
                t = self._empty_terms()
                for idx, (mask, pol) in enumerate(primes):
                    t[idx] = [mask, pol]
                return t

    # ── ActsEnumFixed ────────────────────────────────────────────────────

    def get_actions_fixed(self) -> List[CircAction]:
        return self.actions_fixed.copy()

    def get_state_actions(self, states: List[CircState]) -> List[List[CircAction]]:
        result: List[List[CircAction]] = []
        nv = self.num_vars
        em = self.empty_mask
        for state in states:
            valid: List[CircAction] = []
            t = state.terms
            for act in self.actions_fixed:
                if act.action_type == ACT_COMBINE:
                    i, j = act.i, act.j
                    mi, mj = int(t[i, 0]), int(t[j, 0])
                    if mi == em or mj == em:
                        continue
                    pi, pj = int(t[i, 1]), int(t[j, 1])
                    if mi != mj:
                        continue
                    diff = pi ^ pj
                    if diff and (diff & (diff - 1)) == 0:
                        valid.append(act)
                elif act.action_type == ACT_ABSORB:
                    i, j = act.i, act.j
                    mi, mj = int(t[i, 0]), int(t[j, 0])
                    if mi == em or mj == em:
                        continue
                    pi, pj = int(t[i, 1]), int(t[j, 1])
                    if _term_implies(mi, pi, mj, pj) and mi != mj:
                        valid.append(act)
                elif act.action_type == ACT_REMOVE:
                    if not self._is_active(t, act.i):
                        continue
                    test = t.copy()
                    test[act.i] = [em, 0]
                    if _sop_truth_table(test, self.max_terms, em, nv) == \
                       _sop_truth_table(t, self.max_terms, em, nv):
                        valid.append(act)
                elif act.action_type == ACT_SWAP:
                    if not np.array_equal(t[act.i], t[act.i + 1]):
                        valid.append(act)
            result.append(valid if valid else [self.actions_fixed[0]])
        return result

    # ── Core domain methods ──────────────────────────────────────────────

    def next_state(self, states: List[CircState], actions: List[CircAction]) -> Tuple[List[CircState], List[float]]:
        new_states: List[CircState] = []
        tcs: List[float] = []
        nv = self.num_vars
        em = self.empty_mask
        for state, act in zip(states, actions):
            t = state.terms.copy()
            if act.action_type == ACT_COMBINE:
                i, j = act.i, act.j
                mi, mj = int(t[i, 0]), int(t[j, 0])
                if mi != em and mj != em and mi == mj:
                    pi, pj = int(t[i, 1]), int(t[j, 1])
                    diff = pi ^ pj
                    if diff and (diff & (diff - 1)) == 0:
                        new_mask = mi & ~diff
                        new_pol = pi & new_mask
                        t[i] = [new_mask, new_pol]
                        t[j] = [em, 0]
            elif act.action_type == ACT_ABSORB:
                i, j = act.i, act.j
                mi, mj = int(t[i, 0]), int(t[j, 0])
                if mi != em and mj != em:
                    pi, pj = int(t[i, 1]), int(t[j, 1])
                    if _term_implies(mi, pi, mj, pj) and mi != mj:
                        t[j] = [em, 0]
            elif act.action_type == ACT_REMOVE:
                if self._is_active(t, act.i):
                    test = t.copy()
                    test[act.i] = [em, 0]
                    if _sop_truth_table(test, self.max_terms, em, nv) == \
                       _sop_truth_table(t, self.max_terms, em, nv):
                        t[act.i] = [em, 0]
            elif act.action_type == ACT_SWAP:
                tmp = t[act.i].copy()
                t[act.i] = t[act.i + 1]
                t[act.i + 1] = tmp
            new_states.append(CircState(t))
            tcs.append(1.0)
        return new_states, tcs

    def is_solved(self, states: List[CircState], goals: List[CircGoal]) -> List[bool]:
        return [np.array_equal(s.terms, g.terms) for s, g in zip(states, goals)]

    # ── Problem instance generation ──────────────────────────────────────

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[CircState], List[CircGoal]]:
        states: List[CircState] = []
        goals: List[CircGoal] = []
        for _ in range(num):
            t = self._random_canonical()
            states.append(CircState(t.copy()))
            goals.append(CircGoal(t.copy()))
        return states, goals

    def random_walk_rev(self, states: List[CircState], num_steps_l: List[int]) -> List[CircState]:
        result: List[CircState] = []
        nv = self.num_vars
        em = self.empty_mask
        for state, n_steps in zip(states, num_steps_l):
            t = state.terms.copy()
            for _ in range(n_steps):
                op = np.random.choice(3, p=[0.5, 0.3, 0.2])
                if op == 0:
                    active = [i for i in range(self.max_terms) if self._is_active(t, i)]
                    empty = [i for i in range(self.max_terms) if not self._is_active(t, i)]
                    expandable = [i for i in active
                                  if int(t[i, 0]) != (1 << nv) - 1]
                    if expandable and empty:
                        idx = int(np.random.choice(expandable))
                        slot = int(np.random.choice(empty))
                        mask = int(t[idx, 0])
                        pol = int(t[idx, 1])
                        absent = [v for v in range(nv) if not (mask & (1 << v))]
                        if absent:
                            v = int(np.random.choice(absent))
                            new_mask = mask | (1 << v)
                            t[idx] = [new_mask, pol | (1 << v)]
                            t[slot] = [new_mask, pol & ~(1 << v)]
                elif op == 1:
                    active = [i for i in range(self.max_terms) if self._is_active(t, i)]
                    if len(active) >= 1:
                        i = int(np.random.choice(active))
                        j_choices = [k for k in range(self.max_terms) if k != i]
                        if j_choices:
                            j = j_choices[np.random.randint(len(j_choices))]
                            tmp = t[i].copy()
                            t[i] = t[j]
                            t[j] = tmp
                else:
                    active = [i for i in range(self.max_terms) if self._is_active(t, i)]
                    empty = [i for i in range(self.max_terms) if not self._is_active(t, i)]
                    if active and empty:
                        src = int(np.random.choice(active))
                        slot = int(np.random.choice(empty))
                        mask = int(t[src, 0])
                        pol = int(t[src, 1])
                        if mask < (1 << nv) - 1:
                            absent = [v for v in range(nv) if not (mask & (1 << v))]
                            v = int(np.random.choice(absent))
                            t[slot] = [mask | (1 << v),
                                       pol | (int(np.random.choice([0, 1])) << v)]
            result.append(CircState(t))
        return result

    # ── Neural network input ─────────────────────────────────────────────

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        mt = self.max_terms
        return ([mt, mt, mt, mt],
                [self.mask_oh_depth, self.pol_oh_depth, self.mask_oh_depth, self.pol_oh_depth])

    def to_np_flat_sg(self, states: List[CircState], goals: List[CircGoal]) -> List[NDArray]:
        s_m = np.stack([s.terms[:, 0] for s in states]).astype(np.int64)
        s_p = np.stack([s.terms[:, 1] for s in states]).astype(np.int64)
        g_m = np.stack([g.terms[:, 0] for g in goals]).astype(np.int64)
        g_p = np.stack([g.terms[:, 1] for g in goals]).astype(np.int64)
        return [s_m, s_p, g_m, g_p]

    # ── Visualization ────────────────────────────────────────────────────

    def visualize_state_goal(self, state: CircState, goal: CircGoal, fig: Figure) -> None:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SANDSTORM)

        mt = self.max_terms
        nv = self.num_vars
        em = self.empty_mask
        bw = min(0.10, 0.85 / mt)
        bh = 0.10
        gap = 0.005
        sx = 0.5 - mt * (bw + gap) / 2

        ax.text(0.5, 0.93, "Current SOP", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLACK_90)
        for i in range(mt):
            x = sx + i * (bw + gap)
            y = 0.75
            if self._is_active(state.terms, i):
                m, p = int(state.terms[i, 0]), int(state.terms[i, 1])
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=TERM_COLORS[i % len(TERM_COLORS)],
                    edgecolor="k", linewidth=0.8))
                ax.text(x + bw / 2, y + bh / 2, _term_label(m, p, nv),
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            else:
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=BLACK_10,
                    edgecolor=BLACK_30, linewidth=0.5))

        expr = _sop_to_string(state.terms, mt, em, nv)
        ax.text(0.5, 0.64, f"f = {expr}", ha="center", va="center",
                fontsize=10, color=BLACK_90, style="italic")

        ax.text(0.5, 0.50, "Goal (Minimal SOP)", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLACK_90)
        for i in range(mt):
            x = sx + i * (bw + gap)
            y = 0.32
            if self._is_active(goal.terms, i):
                m, p = int(goal.terms[i, 0]), int(goal.terms[i, 1])
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=GRASS,
                    edgecolor="k", linewidth=0.8))
                ax.text(x + bw / 2, y + bh / 2, _term_label(m, p, nv),
                        ha="center", va="center", fontsize=6.5,
                        color=BLACK_90, fontweight="bold")
            else:
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, facecolor=BLACK_10,
                    edgecolor=BLACK_30, linewidth=0.5))

        goal_str = _sop_to_string(goal.terms, mt, em, nv)
        ax.text(0.5, 0.21, f"f = {goal_str}", ha="center", va="center",
                fontsize=10, color=BLACK_90, style="italic")

        solved = np.array_equal(state.terms, goal.terms)
        status = "SOLVED" if solved else "UNSOLVED"
        ax.text(0.5, 0.07, status, ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=GRASS if solved else GARNET)
        fig.canvas.draw()

    # ── StringToAct ──────────────────────────────────────────────────────

    def string_to_action(self, act_str: str) -> Optional[CircAction]:
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
            elif cmd == "a" and len(parts) == 3:
                i, j = int(parts[1]), int(parts[2])
                for a in self.actions_fixed:
                    if a.action_type == ACT_ABSORB and a.i == i and a.j == j:
                        return a
            elif cmd == "r" and len(parts) == 2:
                i = int(parts[1])
                for a in self.actions_fixed:
                    if a.action_type == ACT_REMOVE and a.i == i:
                        return a
            elif cmd == "s" and len(parts) == 2:
                i = int(parts[1])
                for a in self.actions_fixed:
                    if a.action_type == ACT_SWAP and a.i == i:
                        return a
        except (ValueError, IndexError):
            pass
        return None

    def string_to_action_help(self) -> str:
        return ("'c i j' combine  |  'a i j' absorb j via i  |  "
                "'r i' remove  |  's i' swap i,i+1  (0-indexed)")

    def __repr__(self) -> str:
        return f"CircuitMinimization(num_vars={self.num_vars}, max_terms={self.max_terms})"


@domain_factory.register_parser("circuit")
class CircParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        parts = args_str.split(".")
        if len(parts) == 1:
            return {"num_vars": int(parts[0])}
        if len(parts) == 2:
            return {"num_vars": int(parts[0]), "max_terms": int(parts[1])}
        raise ValueError(f"Expected 'num_vars' or 'num_vars.max_terms', got '{args_str}'")

    def help(self) -> str:
        return "num_vars[.max_terms]. E.g. 'circuit.3' or 'circuit.3.8'"
