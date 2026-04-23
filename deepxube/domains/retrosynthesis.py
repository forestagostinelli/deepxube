"""Simplified Retrosynthesis domain for DeepXube.

States represent molecules as linear carbon chains with functional groups
and bond orders. Actions are chemical reactions that transform functional
groups or change bond orders between adjacent carbons. The goal is to
transform a source molecule into a target molecule through a sequence of
valid reactions where every intermediate state is also a valid molecule.

The reaction graph is strongly connected: any molecule can be transformed
into any other through some sequence of reactions. Multi-step dependencies
(e.g. H->OH requires H->Cl->OH) create genuine search complexity that
cannot be solved by greedy approaches.
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC
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
NH2 = 2
COOH = 3
KETONE = 4
CL = 5
CH3 = 6
NO2 = 7
NUM_GROUPS = 8

GROUP_LABELS = ["H", "OH", "NH₂", "COOH", "C=O", "Cl", "CH₃", "NO₂"]
GROUP_ASCII = ["H", "OH", "NH2", "COOH", "C=O", "Cl", "CH3", "NO2"]

SINGLE = 0
DOUBLE = 1
NUM_BOND_TYPES = 2

FG_REACTIONS: List[Tuple[str, int, int]] = [
    ("halogenate", H, CL),
    ("dehalogenate", CL, H),
    ("hydroxylate", CL, OH),
    ("chlorinate", OH, CL),
    ("aminate", CL, NH2),
    ("diazotize", NH2, CL),
    ("oxidize", OH, KETONE),
    ("reduce", KETONE, OH),
    ("nitrate", H, NO2),
    ("reduce_nitro", NO2, NH2),
    ("methylate", H, CH3),
    ("demethylate", CH3, H),
    ("carboxylate", KETONE, COOH),
    ("decarboxylate", COOH, KETONE),
]
NUM_FG_RXN = len(FG_REACTIONS)
ELIMINATE_ID = NUM_FG_RXN
HYDROGENATE_ID = NUM_FG_RXN + 1

GROUP_COLORS = {
    H: BLACK_50,
    OH: ATLANTIC,
    NH2: CONGAREE,
    COOH: GARNET,
    KETONE: ROSE,
    CL: HORSESHOE,
    CH3: HONEYCOMB,
    NO2: "#5C5C5C",
}


def _mol_to_string(mol: NDArray[np.int8], chain_len: int) -> str:
    groups = mol[:chain_len]
    bonds = mol[chain_len:]
    parts: List[str] = []
    for i in range(chain_len):
        g = GROUP_ASCII[groups[i]]
        parts.append(f"C({g})")
        if i < len(bonds):
            parts.append("=" if bonds[i] == DOUBLE else "-")
    return "".join(parts)


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
        return _mol_to_string(self.mol, (len(self.mol) + 1) // 2)


class RetroGoal(Goal):
    def __init__(self, mol: NDArray[np.int8]):
        self.mol: NDArray[np.int8] = mol


class RetroAction(Action):
    __slots__ = ["rxn_type", "position", "_hash"]

    def __init__(self, rxn_type: int, position: int):
        self.rxn_type: int = rxn_type
        self.position: int = position
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.rxn_type, self.position))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RetroAction):
            return self.rxn_type == other.rxn_type and self.position == other.position
        return NotImplemented

    def __repr__(self) -> str:
        if self.rxn_type < NUM_FG_RXN:
            name = FG_REACTIONS[self.rxn_type][0]
        elif self.rxn_type == ELIMINATE_ID:
            name = "eliminate"
        else:
            name = "hydrogenate"
        return f"{name}@{self.position}"


@domain_factory.register_class("retro")
class Retrosynthesis(
    ActsEnum[RetroState, RetroAction, RetroGoal],
    GoalStartRevWalkable[RetroState, RetroAction, RetroGoal],
    HasFlatSGIn[RetroState, RetroAction, RetroGoal],
    StateGoalVizable[RetroState, RetroAction, RetroGoal],
    StringToAct[RetroState, RetroAction, RetroGoal],
):

    def __init__(self, chain_len: int = 6):
        super().__init__()
        self.chain_len: int = chain_len
        self.num_bonds: int = chain_len - 1

    def get_state_actions(self, states: List[RetroState]) -> List[List[RetroAction]]:
        result: List[List[RetroAction]] = []
        for state in states:
            groups = state.mol[: self.chain_len]
            bonds = state.mol[self.chain_len :]
            actions: List[RetroAction] = []

            for rxn_id in range(NUM_FG_RXN):
                req = FG_REACTIONS[rxn_id][1]
                for pos in range(self.chain_len):
                    if groups[pos] == req:
                        actions.append(RetroAction(rxn_id, pos))

            for pos in range(self.num_bonds):
                if groups[pos] == CL and bonds[pos] == SINGLE:
                    actions.append(RetroAction(ELIMINATE_ID, pos))

            for pos in range(self.num_bonds):
                if bonds[pos] == DOUBLE:
                    actions.append(RetroAction(HYDROGENATE_ID, pos))

            if not actions:
                actions = [RetroAction(0, 0)]
            result.append(actions)
        return result

    def sample_state_action(self, states: List[RetroState]) -> List[RetroAction]:
        actions_l = self.get_state_actions(states)
        return [stdlib_random.choice(acts) for acts in actions_l]

    def next_state(
        self, states: List[RetroState], actions: List[RetroAction]
    ) -> Tuple[List[RetroState], List[float]]:
        new_states: List[RetroState] = []
        tcs: List[float] = []
        for state, act in zip(states, actions):
            new_mol = state.mol.copy()
            if act.rxn_type < NUM_FG_RXN:
                _, req, prod = FG_REACTIONS[act.rxn_type]
                if new_mol[act.position] == req:
                    new_mol[act.position] = prod
            elif act.rxn_type == ELIMINATE_ID:
                p = act.position
                if (
                    p < self.num_bonds
                    and new_mol[p] == CL
                    and new_mol[self.chain_len + p] == SINGLE
                ):
                    new_mol[p] = H
                    new_mol[self.chain_len + p] = DOUBLE
            elif act.rxn_type == HYDROGENATE_ID:
                p = act.position
                if p < self.num_bonds and new_mol[self.chain_len + p] == DOUBLE:
                    new_mol[self.chain_len + p] = SINGLE
            new_states.append(RetroState(new_mol))
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
            groups = np.random.randint(0, NUM_GROUPS, size=self.chain_len, dtype=np.int8)
            bonds = np.random.randint(
                0, NUM_BOND_TYPES, size=self.num_bonds, dtype=np.int8
            )
            mol = np.concatenate([groups, bonds])
            states.append(RetroState(mol.copy()))
            goals.append(RetroGoal(mol.copy()))
        return states, goals

    def random_walk_rev(
        self, states: List[RetroState], num_steps_l: List[int]
    ) -> List[RetroState]:
        return self.random_walk(states, num_steps_l)[0]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return (
            [self.chain_len, self.num_bonds, self.chain_len, self.num_bonds],
            [NUM_GROUPS, NUM_BOND_TYPES, NUM_GROUPS, NUM_BOND_TYPES],
        )

    def to_np_flat_sg(
        self, states: List[RetroState], goals: List[RetroGoal]
    ) -> List[NDArray]:
        cl = self.chain_len
        s_g = np.stack([s.mol[:cl] for s in states], axis=0).astype(np.int64)
        s_b = np.stack([s.mol[cl:] for s in states], axis=0).astype(np.int64)
        g_g = np.stack([g.mol[:cl] for g in goals], axis=0).astype(np.int64)
        g_b = np.stack([g.mol[cl:] for g in goals], axis=0).astype(np.int64)
        return [s_g, s_b, g_g, g_b]

    def visualize_state_goal(
        self, state: RetroState, goal: RetroGoal, fig: Figure
    ) -> None:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SANDSTORM)

        g_mol = goal.mol
        self._draw_molecule(ax, state.mol, g_mol, y_center=0.72, label="Current Molecule")
        self._draw_molecule(ax, g_mol, g_mol, y_center=0.28, label="Target Molecule")

        solved = np.array_equal(state.mol, g_mol)
        cl = self.chain_len
        diff_g = int(np.sum(state.mol[:cl] != g_mol[:cl]))
        diff_b = int(np.sum(state.mol[cl:] != g_mol[cl:]))
        if solved:
            status = "SOLVED"
            color = GRASS
        else:
            status = f"UNSOLVED ({diff_g} groups, {diff_b} bonds differ)"
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
        cl = self.chain_len
        groups = mol[:cl]
        bonds = mol[cl:]
        g_groups = goal_mol[:cl]
        g_bonds = goal_mol[cl:]

        ax.text(
            0.5, y_center + 0.17, label,
            ha="center", va="center", fontsize=13, fontweight="bold", color=BLACK_90,
        )

        margin = 0.10
        spacing = (1.0 - 2 * margin) / max(cl - 1, 1)
        r = 0.028

        for i in range(self.num_bonds):
            x1 = margin + i * spacing + r + 0.005
            x2 = margin + (i + 1) * spacing - r - 0.005
            match = bonds[i] == g_bonds[i]
            bond_color = HORSESHOE if match else ROSE

            if bonds[i] == SINGLE:
                ax.plot(
                    [x1, x2], [y_center, y_center],
                    color=bond_color, linewidth=2.5, zorder=1, solid_capstyle="butt",
                )
            else:
                off = 0.009
                ax.plot(
                    [x1, x2], [y_center - off, y_center - off],
                    color=bond_color, linewidth=2.0, zorder=1, solid_capstyle="butt",
                )
                ax.plot(
                    [x1, x2], [y_center + off, y_center + off],
                    color=bond_color, linewidth=2.0, zorder=1, solid_capstyle="butt",
                )

        for i in range(cl):
            x = margin + i * spacing
            match = groups[i] == g_groups[i]
            circle_color = HORSESHOE if match else GARNET
            circle = patches.Circle(
                (x, y_center), r,
                facecolor=circle_color, edgecolor=BLACK_90,
                linewidth=1.2, zorder=3,
            )
            ax.add_patch(circle)
            ax.text(
                x, y_center, "C",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", zorder=4,
            )

            if groups[i] != H:
                fg_label = GROUP_LABELS[groups[i]]
                fg_color = GROUP_COLORS.get(int(groups[i]), BLACK_90)
                fg_y = y_center + 0.065
                ax.text(
                    x, fg_y, fg_label,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=fg_color,
                    bbox=dict(
                        boxstyle="square,pad=0.15", facecolor="white",
                        edgecolor=fg_color, linewidth=0.8,
                    ),
                    zorder=5,
                )
            else:
                fg_y = y_center + 0.055
                ax.text(
                    x, fg_y, "H",
                    ha="center", va="center", fontsize=8, color=BLACK_30, zorder=5,
                )

        text_str = _mol_to_string(mol, cl)
        ax.text(
            0.5, y_center - 0.10, text_str,
            ha="center", va="center", fontsize=9, color=BLACK_50, style="italic",
        )

    def string_to_action(self, act_str: str) -> Optional[RetroAction]:
        try:
            parts = act_str.strip().lower().split()
            if len(parts) == 2:
                rxn_name = parts[0]
                pos = int(parts[1])
                for rxn_id, (name, _, _) in enumerate(FG_REACTIONS):
                    if name == rxn_name:
                        if 0 <= pos < self.chain_len:
                            return RetroAction(rxn_id, pos)
                if rxn_name == "eliminate" and 0 <= pos < self.num_bonds:
                    return RetroAction(ELIMINATE_ID, pos)
                if rxn_name == "hydrogenate" and 0 <= pos < self.num_bonds:
                    return RetroAction(HYDROGENATE_ID, pos)
        except (ValueError, IndexError):
            pass
        return None

    def string_to_action_help(self) -> str:
        return "'reaction position' (e.g. 'halogenate 2', 'eliminate 0', 'hydrogenate 3')"

    def __repr__(self) -> str:
        return f"Retrosynthesis(chain_len={self.chain_len}, vocab={NUM_GROUPS} groups)"


@domain_factory.register_parser("retro")
class RetroParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"chain_len": int(args_str)}

    def help(self) -> str:
        return "Chain length. E.g. 'retro.6'"
