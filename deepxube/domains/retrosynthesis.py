"""Simplified Retrosynthesis domain for DeepXube.

Molecules are linear carbon chains where each position carries a functional
group. Most reactions are SITE-SELECTIVE: they act on the position with the
highest reactivity among eligible groups, not a player-chosen position.
This creates tight coupling between positions -- changing a group anywhere
can redirect where the next reaction acts.

Global reagent incompatibilities (e.g. oxidation blocked by bromides present
anywhere) and protecting group mechanics add further planning dependencies.
The result is a genuinely non-decomposable puzzle where the optimal reaction
sequence depends on the entire molecular state.

Functional groups (8):
  H(0), OH(1), C=O(2), NH2(3), COOH(4), Br(5), OPG(6), NPG(7)

Reactions (12 site-selective + 3*N position-specific):
  Site-selective reactions target the eligible position with the highest
  reactivity score. Ties broken by lowest index. Each has a global
  precondition (certain groups must be absent everywhere).

  Position-specific: radical bromination, protect, deprotect.
"""

from typing import List, Tuple, Optional, Dict, Any, FrozenSet
import numpy as np
from numpy.typing import NDArray
import matplotlib.patches as patches
from matplotlib.figure import Figure
import random as stdlib_random

from deepxube.base.factory import Parser
from deepxube.base.domain import (
    State, Action, Goal,
    ActsEnum, GoalStartRevWalkable,
    StateGoalVizable, StringToAct,
)
from deepxube.base.nnet_input import HasFlatSGIn
from deepxube.factories.domain_factory import domain_factory

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

H = 0
OH = 1
KETONE = 2
NH2 = 3
COOH = 4
BR = 5
OPG = 6
NPG = 7
NUM_GROUPS = 8
NUM_REAL_GROUPS = 6

GROUP_LABELS = ["H", "OH", "C=O", "NH₂", "COOH", "Br", "OPG", "NPG"]
GROUP_ASCII = ["H", "OH", "C=O", "NH2", "COOH", "Br", "OPG", "NPG"]

REACTIVITY = np.array([0, 3, 5, 4, 2, 6, 0, 0], dtype=np.int8)

GROUP_COLORS = [
    BLACK_30,    # H
    ATLANTIC,    # OH
    ROSE,        # C=O
    CONGAREE,    # NH2
    GARNET,      # COOH
    HORSESHOE,   # Br
    HONEYCOMB,   # OPG
    HONEYCOMB,   # NPG
]

_S = frozenset
SELECTIVE_RXNS: List[Tuple[str, FrozenSet[int], Dict[int, int], FrozenSet[int]]] = [
    ("mild_oxidation",      _S({OH}),           {OH: KETONE},               _S({BR})),
    ("strong_oxidation",    _S({OH, KETONE}),   {OH: COOH, KETONE: COOH},   _S({NH2})),
    ("mild_reduction",      _S({KETONE}),       {KETONE: OH},               _S({COOH})),
    ("strong_reduction",    _S({KETONE, COOH}), {KETONE: OH, COOH: OH},     _S({BR})),
    ("amination",           _S({BR}),           {BR: NH2},                  _S({KETONE})),
    ("hydroxylation",       _S({BR}),           {BR: OH},                   _S()),
    ("bromination",         _S({OH, NH2}),      {OH: BR, NH2: BR},          _S()),
    ("reductive_amination", _S({KETONE}),       {KETONE: NH2},              _S({NH2})),
    ("ketone_from_acid",    _S({COOH}),         {COOH: KETONE},             _S({OH, NH2})),
    ("carboxylation",       _S({BR}),           {BR: COOH},                 _S({KETONE})),
    ("hydrogenolysis",      _S({BR}),           {BR: H},                    _S({KETONE})),
    ("defunctionalize",     _S({OH, NH2, COOH}), {OH: H, NH2: H, COOH: H}, _S()),
]
NUM_SELECTIVE = len(SELECTIVE_RXNS)
RXN_RADICAL_BR = NUM_SELECTIVE
RXN_PROTECT = NUM_SELECTIVE + 1
RXN_DEPROTECT = NUM_SELECTIVE + 2
RXN_NAMES = [r[0] for r in SELECTIVE_RXNS] + ["radical_br", "protect", "deprotect"]


def _select_site(mol: NDArray[np.int8], eligible: FrozenSet[int], n: int) -> int:
    best_idx = -1
    best_react = -1
    for i in range(n):
        g = int(mol[i])
        if g in eligible and REACTIVITY[g] > best_react:
            best_react = REACTIVITY[g]
            best_idx = i
    return best_idx


def _has_any(mol: NDArray[np.int8], groups: FrozenSet[int], n: int) -> bool:
    for i in range(n):
        if int(mol[i]) in groups:
            return True
    return False


def _mol_to_string(mol: NDArray[np.int8]) -> str:
    return "-".join(GROUP_ASCII[g] for g in mol)


class RetroState(State):
    __slots__ = ["mol", "_hash"]

    def __init__(self, mol: NDArray[np.int8]):
        self.mol: NDArray[np.int8] = mol
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.mol.tobytes())
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RetroState):
            return np.array_equal(self.mol, other.mol)
        return NotImplemented

    def __repr__(self) -> str:
        return _mol_to_string(self.mol)


class RetroGoal(Goal):
    def __init__(self, mol: NDArray[np.int8]):
        self.mol: NDArray[np.int8] = mol


class RetroAction(Action):
    __slots__ = ["rxn_id", "position", "_hash"]

    def __init__(self, rxn_id: int, position: int = -1):
        self.rxn_id: int = rxn_id
        self.position: int = position
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.rxn_id, self.position))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RetroAction):
            return self.rxn_id == other.rxn_id and self.position == other.position
        return NotImplemented

    def __repr__(self) -> str:
        name = RXN_NAMES[self.rxn_id]
        if self.position >= 0:
            return f"{name}@{self.position}"
        return name


@domain_factory.register_class("retro")
class Retrosynthesis(
    ActsEnum[RetroState, RetroAction, RetroGoal],
    GoalStartRevWalkable[RetroState, RetroAction, RetroGoal],
    HasFlatSGIn[RetroState, RetroAction, RetroGoal],
    StateGoalVizable[RetroState, RetroAction, RetroGoal],
    StringToAct[RetroState, RetroAction, RetroGoal],
):

    def __init__(self, chain_len: int = 5):
        super().__init__()
        self.chain_len: int = chain_len

    def get_state_actions(
        self, states: List[RetroState]
    ) -> List[List[RetroAction]]:
        n = self.chain_len
        result: List[List[RetroAction]] = []
        for state in states:
            mol = state.mol
            actions: List[RetroAction] = []

            for rxn_id in range(NUM_SELECTIVE):
                _, eligible, _, forbidden = SELECTIVE_RXNS[rxn_id]
                if forbidden and _has_any(mol, forbidden, n):
                    continue
                if _select_site(mol, eligible, n) >= 0:
                    actions.append(RetroAction(rxn_id))

            d7_ok = not _has_any(mol, _S({OH, NH2}), n)
            if d7_ok:
                for pos in range(n):
                    if mol[pos] == H:
                        actions.append(RetroAction(RXN_RADICAL_BR, pos))

            for pos in range(n):
                if mol[pos] == OH or mol[pos] == NH2:
                    actions.append(RetroAction(RXN_PROTECT, pos))

            c2_ok = not _has_any(mol, _S({BR}), n)
            if c2_ok:
                for pos in range(n):
                    if mol[pos] == OPG or mol[pos] == NPG:
                        actions.append(RetroAction(RXN_DEPROTECT, pos))

            if not actions:
                actions = [RetroAction(5)]
            result.append(actions)
        return result

    def sample_state_action(
        self, states: List[RetroState]
    ) -> List[RetroAction]:
        actions_l = self.get_state_actions(states)
        return [stdlib_random.choice(acts) for acts in actions_l]

    def next_state(
        self, states: List[RetroState], actions: List[RetroAction]
    ) -> Tuple[List[RetroState], List[float]]:
        n = self.chain_len
        new_states: List[RetroState] = []
        tcs: List[float] = []
        for state, act in zip(states, actions):
            mol = state.mol.copy()

            if act.rxn_id < NUM_SELECTIVE:
                _, eligible, effects, forbidden = SELECTIVE_RXNS[act.rxn_id]
                if not (forbidden and _has_any(mol, forbidden, n)):
                    target = _select_site(mol, eligible, n)
                    if target >= 0:
                        mol[target] = effects[int(mol[target])]
            elif act.rxn_id == RXN_RADICAL_BR:
                p = act.position
                if 0 <= p < n and mol[p] == H and not _has_any(mol, _S({OH, NH2}), n):
                    mol[p] = BR
            elif act.rxn_id == RXN_PROTECT:
                p = act.position
                if 0 <= p < n:
                    if mol[p] == OH:
                        mol[p] = OPG
                    elif mol[p] == NH2:
                        mol[p] = NPG
            elif act.rxn_id == RXN_DEPROTECT:
                p = act.position
                if 0 <= p < n and not _has_any(mol, _S({BR}), n):
                    if mol[p] == OPG:
                        mol[p] = OH
                    elif mol[p] == NPG:
                        mol[p] = NH2

            new_states.append(RetroState(mol))
            tcs.append(1.0)
        return new_states, tcs

    def is_solved(
        self, states: List[RetroState], goals: List[RetroGoal]
    ) -> List[bool]:
        return [np.array_equal(s.mol, g.mol) for s, g in zip(states, goals)]

    def sample_goalstate_goal_pairs(
        self, num: int
    ) -> Tuple[List[RetroState], List[RetroGoal]]:
        states: List[RetroState] = []
        goals: List[RetroGoal] = []
        for _ in range(num):
            mol = np.random.randint(0, NUM_REAL_GROUPS, size=self.chain_len, dtype=np.int8)
            states.append(RetroState(mol.copy()))
            goals.append(RetroGoal(mol.copy()))
        return states, goals

    def random_walk_rev(
        self, states: List[RetroState], num_steps_l: List[int]
    ) -> List[RetroState]:
        return self.random_walk(states, num_steps_l)[0]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return (
            [self.chain_len, self.chain_len],
            [NUM_GROUPS, NUM_GROUPS],
        )

    def to_np_flat_sg(
        self, states: List[RetroState], goals: List[RetroGoal]
    ) -> List[NDArray]:
        s = np.stack([st.mol for st in states], axis=0).astype(np.int64)
        g = np.stack([gl.mol for gl in goals], axis=0).astype(np.int64)
        return [s, g]

    def visualize_state_goal(
        self, state: RetroState, goal: RetroGoal, fig: Figure
    ) -> None:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SANDSTORM)

        self._draw_molecule(ax, state.mol, goal.mol, y_center=0.72, label="Current Molecule")
        self._draw_molecule(ax, goal.mol, goal.mol, y_center=0.28, label="Target Molecule")

        solved = np.array_equal(state.mol, goal.mol)
        diff = int(np.sum(state.mol != goal.mol))
        if solved:
            status = "SOLVED"
            color = GRASS
        else:
            status = f"UNSOLVED ({diff} position{'s' if diff != 1 else ''} differ)"
            color = GARNET
        ax.text(
            0.5, 0.03, status,
            ha="center", va="center", fontsize=12, fontweight="bold", color=color,
        )
        fig.canvas.draw()

    def _draw_molecule(
        self, ax, mol: NDArray[np.int8], goal_mol: NDArray[np.int8],
        y_center: float, label: str,
    ) -> None:
        n = self.chain_len
        ax.text(
            0.5, y_center + 0.17, label,
            ha="center", va="center", fontsize=13, fontweight="bold", color=BLACK_90,
        )

        margin = 0.10
        spacing = (1.0 - 2 * margin) / max(n - 1, 1)
        r = 0.028

        for i in range(n - 1):
            x1 = margin + i * spacing + r + 0.005
            x2 = margin + (i + 1) * spacing - r - 0.005
            ax.plot(
                [x1, x2], [y_center, y_center],
                color=BLACK_30, linewidth=2.0, zorder=1, solid_capstyle="butt",
            )

        for i in range(n):
            x = margin + i * spacing
            g = int(mol[i])
            match = mol[i] == goal_mol[i]
            circle_color = HORSESHOE if match else GARNET

            is_protected = g in (OPG, NPG)
            edge_style = dict(
                edgecolor=HONEYCOMB if is_protected else BLACK_90,
                linewidth=2.0 if is_protected else 1.2,
                linestyle="--" if is_protected else "-",
            )

            circle = patches.Circle(
                (x, y_center), r,
                facecolor=circle_color, zorder=3, **edge_style,
            )
            ax.add_patch(circle)
            ax.text(
                x, y_center, "C",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", zorder=4,
            )

            fg_label = GROUP_LABELS[g]
            fg_color = GROUP_COLORS[g]
            if g == H:
                ax.text(
                    x, y_center + 0.055, "H",
                    ha="center", va="center", fontsize=8, color=BLACK_30, zorder=5,
                )
            else:
                ax.text(
                    x, y_center + 0.065, fg_label,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=fg_color,
                    bbox=dict(
                        boxstyle="square,pad=0.15", facecolor="white",
                        edgecolor=fg_color, linewidth=0.8,
                    ),
                    zorder=5,
                )

        ax.text(
            0.5, y_center - 0.10, _mol_to_string(mol),
            ha="center", va="center", fontsize=9, color=BLACK_50, style="italic",
        )

    def string_to_action(self, act_str: str) -> Optional[RetroAction]:
        try:
            s = act_str.strip().lower()
            if "@" in s:
                name, pos_str = s.rsplit("@", 1)
                pos = int(pos_str)
            else:
                parts = s.split()
                if len(parts) == 2:
                    name, pos = parts[0], int(parts[1])
                else:
                    name, pos = s, -1

            for rxn_id, rxn_name in enumerate(RXN_NAMES):
                if rxn_name == name:
                    if rxn_id < NUM_SELECTIVE:
                        return RetroAction(rxn_id)
                    elif 0 <= pos < self.chain_len:
                        return RetroAction(rxn_id, pos)
            return None
        except (ValueError, IndexError):
            return None

    def string_to_action_help(self) -> str:
        sel = ", ".join(RXN_NAMES[:NUM_SELECTIVE])
        pos = ", ".join(RXN_NAMES[NUM_SELECTIVE:])
        return (
            f"Site-selective (no position): {sel}\n"
            f"Position-specific: {pos} (e.g. 'radical_br@2', 'protect 0')"
        )

    def __repr__(self) -> str:
        return f"Retrosynthesis(chain_len={self.chain_len})"


@domain_factory.register_parser("retro")
class RetroParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"chain_len": int(args_str)}

    def help(self) -> str:
        return "Chain length. E.g. 'retro.5'"
