"""Word Ladder domain for DeepXube.

States are English words of a fixed length. Actions substitute a single
character at a specific position, producing a new valid English word.
The goal is to reach a target word through a sequence of single-character
substitutions where every intermediate state is also a valid word.

The dictionary is filtered to the largest connected component so that
all sampled problem instances are guaranteed to be solvable.
"""

from typing import List, Tuple, Optional, Dict, Any, Set, FrozenSet
from collections import deque
import numpy as np
from numpy.typing import NDArray
import matplotlib.patches as patches
from matplotlib.figure import Figure
import random as stdlib_random
import os

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

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
DICT_PATH = "/usr/share/dict/web2"


def _load_dictionary(word_len: int, dict_path: str) -> Set[str]:
    words: Set[str] = set()
    if not os.path.exists(dict_path):
        for alt in ["/usr/share/dict/words", "/usr/share/dict/american-english"]:
            if os.path.exists(alt):
                dict_path = alt
                break
    with open(dict_path) as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == word_len and w.isascii() and w.isalpha():
                words.add(w)
    return words


def _get_neighbors(word: str, vocab: Set[str]) -> List[str]:
    neighbors: List[str] = []
    for pos in range(len(word)):
        for c in ALPHABET:
            if c != word[pos]:
                nw = word[:pos] + c + word[pos + 1:]
                if nw in vocab:
                    neighbors.append(nw)
    return neighbors


def _largest_component(vocab: Set[str]) -> Set[str]:
    visited: Set[str] = set()
    best: Set[str] = set()
    for w in vocab:
        if w in visited:
            continue
        comp: Set[str] = set()
        queue: deque = deque([w])
        while queue:
            cur = queue.popleft()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            for n in _get_neighbors(cur, vocab):
                if n not in comp:
                    queue.append(n)
        if len(comp) > len(best):
            best = comp
    return best


class WLState(State):
    __slots__ = ["word", "_hash"]

    def __init__(self, word: str):
        self.word: str = word
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.word)
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WLState):
            return self.word == other.word
        return NotImplemented

    def __repr__(self) -> str:
        return self.word


class WLGoal(Goal):
    def __init__(self, word: str):
        self.word: str = word


class WLAction(Action):
    def __init__(self, pos: int, char: str):
        self.pos = pos
        self.char = char
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.pos, self.char))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WLAction):
            return self.pos == other.pos and self.char == other.char
        return NotImplemented

    def __repr__(self) -> str:
        return f"pos{self.pos}={self.char}"


@domain_factory.register_class("wordladder")
class WordLadder(
    ActsEnum[WLState, WLAction, WLGoal],
    GoalStartRevWalkable[WLState, WLAction, WLGoal],
    HasFlatSGIn[WLState, WLAction, WLGoal],
    StateGoalVizable[WLState, WLAction, WLGoal],
    StringToAct[WLState, WLAction, WLGoal],
):

    def __init__(self, word_len: int = 4, dict_path: str = DICT_PATH):
        super().__init__()
        self.word_len: int = word_len
        all_words = _load_dictionary(word_len, dict_path)
        self.vocab: Set[str] = _largest_component(all_words)
        self.vocab_list: List[str] = sorted(self.vocab)
        self._neighbor_cache: Dict[str, List[str]] = {}

    def _neighbors(self, word: str) -> List[str]:
        if word not in self._neighbor_cache:
            self._neighbor_cache[word] = _get_neighbors(word, self.vocab)
        return self._neighbor_cache[word]

    # ── ActsEnum ─────────────────────────────────────────────────────────

    def get_state_actions(self, states: List[WLState]) -> List[List[WLAction]]:
        result: List[List[WLAction]] = []
        for state in states:
            actions: List[WLAction] = []
            for pos in range(self.word_len):
                for c in ALPHABET:
                    if c != state.word[pos]:
                        nw = state.word[:pos] + c + state.word[pos + 1:]
                        if nw in self.vocab:
                            actions.append(WLAction(pos, c))
            result.append(actions if actions else [WLAction(0, state.word[0])])
        return result

    def sample_state_action(self, states: List[WLState]) -> List[WLAction]:
        actions: List[WLAction] = []
        for state in states:
            nbrs = self._neighbors(state.word)
            if nbrs:
                nw = stdlib_random.choice(nbrs)
                for pos in range(self.word_len):
                    if nw[pos] != state.word[pos]:
                        actions.append(WLAction(pos, nw[pos]))
                        break
            else:
                actions.append(WLAction(0, state.word[0]))
        return actions

    # ── Core domain methods ──────────────────────────────────────────────

    def next_state(self, states: List[WLState], actions: List[WLAction]) -> Tuple[List[WLState], List[float]]:
        new_states: List[WLState] = []
        tcs: List[float] = []
        for state, act in zip(states, actions):
            nw = state.word[:act.pos] + act.char + state.word[act.pos + 1:]
            if nw in self.vocab:
                new_states.append(WLState(nw))
                tcs.append(1.0)
            else:
                new_states.append(state)
                tcs.append(1.0)
        return new_states, tcs

    def is_solved(self, states: List[WLState], goals: List[WLGoal]) -> List[bool]:
        return [s.word == g.word for s, g in zip(states, goals)]

    # ── Problem instance generation ──────────────────────────────────────

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[WLState], List[WLGoal]]:
        words = stdlib_random.choices(self.vocab_list, k=num)
        states = [WLState(w) for w in words]
        goals = [WLGoal(w) for w in words]
        return states, goals

    def random_walk_rev(self, states: List[WLState], num_steps_l: List[int]) -> List[WLState]:
        return self.random_walk(states, num_steps_l)[0]

    # ── Neural network input ─────────────────────────────────────────────

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return ([self.word_len, self.word_len], [26, 26])

    def to_np_flat_sg(self, states: List[WLState], goals: List[WLGoal]) -> List[NDArray]:
        s_arr = np.array([[ord(c) - ord("a") for c in s.word] for s in states], dtype=np.int64)
        g_arr = np.array([[ord(c) - ord("a") for c in g.word] for g in goals], dtype=np.int64)
        return [s_arr, g_arr]

    # ── Visualization ────────────────────────────────────────────────────

    def visualize_state_goal(self, state: WLState, goal: WLGoal, fig: Figure) -> None:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SANDSTORM)

        bw = min(0.12, 0.7 / self.word_len)
        bh = 0.14
        gap = 0.01
        sx = 0.5 - self.word_len * (bw + gap) / 2

        ax.text(0.5, 0.92, "Current Word", ha="center", va="center",
                fontsize=13, fontweight="bold", color=BLACK_90)
        for i in range(self.word_len):
            x = sx + i * (bw + gap)
            y = 0.72
            match = state.word[i] == goal.word[i]
            color = HORSESHOE if match else GARNET
            ax.add_patch(patches.Rectangle(
                (x, y), bw, bh, facecolor=color, edgecolor="k", linewidth=1.0))
            ax.text(x + bw / 2, y + bh / 2, state.word[i].upper(),
                    ha="center", va="center", fontsize=18,
                    color="white", fontweight="bold")

        ax.text(0.5, 0.58, f'"{state.word}"', ha="center", va="center",
                fontsize=14, color=BLACK_90, style="italic")

        ax.text(0.5, 0.45, "Target Word", ha="center", va="center",
                fontsize=13, fontweight="bold", color=BLACK_90)
        for i in range(self.word_len):
            x = sx + i * (bw + gap)
            y = 0.26
            ax.add_patch(patches.Rectangle(
                (x, y), bw, bh, facecolor=ATLANTIC, edgecolor="k", linewidth=1.0))
            ax.text(x + bw / 2, y + bh / 2, goal.word[i].upper(),
                    ha="center", va="center", fontsize=18,
                    color="white", fontweight="bold")

        ax.text(0.5, 0.12, f'"{goal.word}"', ha="center", va="center",
                fontsize=14, color=BLACK_90, style="italic")

        solved = state.word == goal.word
        diff = sum(1 for a, b in zip(state.word, goal.word) if a != b)
        status = "SOLVED" if solved else f"UNSOLVED ({diff} letters differ)"
        ax.text(0.5, 0.03, status, ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=GRASS if solved else GARNET)
        fig.canvas.draw()

    # ── StringToAct ──────────────────────────────────────────────────────

    def string_to_action(self, act_str: str) -> Optional[WLAction]:
        try:
            parts = act_str.strip().split()
            if len(parts) == 2:
                pos = int(parts[0])
                char = parts[1].lower()
                if 0 <= pos < self.word_len and len(char) == 1 and char in ALPHABET:
                    return WLAction(pos, char)
            elif len(parts) == 1 and len(parts[0]) == self.word_len:
                target = parts[0].lower()
                if target in self.vocab:
                    diffs = [(i, target[i]) for i in range(self.word_len)]
                    changes = [(i, c) for i, c in diffs]
                    for i, c in changes:
                        return WLAction(i, c)
        except (ValueError, IndexError):
            pass
        return None

    def string_to_action_help(self) -> str:
        return f"'pos char' to substitute (e.g. '2 a' changes position 2 to 'a', 0-indexed)"

    def __repr__(self) -> str:
        return f"WordLadder(word_len={self.word_len}, vocab_size={len(self.vocab)})"


@domain_factory.register_parser("wordladder")
class WLParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {"word_len": int(args_str)}

    def help(self) -> str:
        return "Word length. E.g. 'wordladder.4'"
