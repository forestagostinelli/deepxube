from typing import List, Tuple, Optional, Dict, Any

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, GoalStartRevWalkableActsRev, GoalFixed, StateGoalVizable, StringToAct
from deepxube.factories.domain_factory import domain_factory
from deepxube.base.nnet_input import HasFlatSGIn
import numpy as np
import matplotlib.patches as patches
from matplotlib.figure import Figure

from numpy.typing import NDArray


class HanoiState(State):
    __slots__ = ['disks', 'hash']

    def __init__(self, disks: NDArray[np.uint8]):
        self.disks: NDArray[np.uint8] = disks
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(self.disks.tobytes())
        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HanoiState):
            return np.array_equal(self.disks, other.disks)
        return NotImplemented


class HanoiGoal(Goal):
    def __init__(self, disks: NDArray[np.uint8]):
        self.disks: NDArray[np.uint8] = disks


class HanoiAction(Action):
    def __init__(self, action: int, from_peg: int, to_peg: int):
        self.action = action
        self.from_peg = from_peg
        self.to_peg = to_peg

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HanoiAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"peg {self.from_peg} -> peg {self.to_peg}"


@domain_factory.register_class("hanoi")
class Hanoi(ActsEnumFixed[HanoiState, HanoiAction, HanoiGoal], GoalStartRevWalkableActsRev[HanoiState, HanoiAction, HanoiGoal],
            GoalFixed[HanoiState, HanoiAction, HanoiGoal], HasFlatSGIn[HanoiState, HanoiAction, HanoiGoal],
            StateGoalVizable[HanoiState, HanoiAction, HanoiGoal], StringToAct[HanoiState, HanoiAction, HanoiGoal]):

    def __init__(self, num_disks: int = 4, num_pegs: int = 3):
        super().__init__()
        self.num_disks: int = num_disks
        self.num_pegs: int = num_pegs

        self.actions_fixed: List[HanoiAction] = [
            HanoiAction(f * num_pegs + t, f, t)
            for f in range(num_pegs)
            for t in range(num_pegs)
            if f != t
        ]

        self.goal_disks: NDArray[np.uint8] = np.full(num_disks, num_pegs - 1, dtype=np.uint8)

    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[HanoiState], List[HanoiGoal]]:
        states = [HanoiState(self.goal_disks.copy()) for _ in range(num)]
        goals = [HanoiGoal(self.goal_disks.copy()) for _ in range(num)]
        return states, goals

    def get_goal(self) -> HanoiGoal:
        return HanoiGoal(self.goal_disks.copy())

    def get_actions_fixed(self) -> List[HanoiAction]:
        return self.actions_fixed.copy()

    def get_state_actions(self, states: List[HanoiState]) -> List[List[HanoiAction]]:
        result: List[List[HanoiAction]] = []
        for state in states:
            top = self._top_disk_per_peg_single(state.disks)
            valid: List[HanoiAction] = []
            for action in self.actions_fixed:
                f, t = action.from_peg, action.to_peg
                if top[f] == self.num_disks:
                    continue
                if top[t] != self.num_disks and top[f] <= top[t]:
                    continue
                valid.append(action)
            result.append(valid)
        return result

    def rev_action(self, states: List[HanoiState], actions: List[HanoiAction]) -> List[HanoiAction]:
        rev: List[HanoiAction] = []
        for action in actions:
            f, t = action.from_peg, action.to_peg
            rev.append(HanoiAction(t * self.num_pegs + f, t, f))
        return rev

    def next_state(self, states: List[HanoiState], actions: List[HanoiAction]) -> Tuple[List[HanoiState], List[float]]:
        states_np = np.stack([s.disks for s in states], axis=0).copy()
        tcs = np.zeros(len(states), dtype=np.float64)

        for action in set(actions):
            f, t = action.from_peg, action.to_peg
            act_idxs = np.array([i for i, a in enumerate(actions) if a == action])
            batch = states_np[act_idxs]

            top_f, _, valid = self._validity_batch(batch, f, t)
            valid_sub = np.where(valid)[0]
            if len(valid_sub) > 0:
                states_np[act_idxs[valid_sub], top_f[valid_sub]] = t
                tcs[act_idxs[valid_sub]] = 1.0

        return [HanoiState(states_np[i]) for i in range(len(states))], list(tcs)

    def is_solved(self, states: List[HanoiState], goals: List[HanoiGoal]) -> List[bool]:
        states_np = np.stack([s.disks for s in states], axis=0)
        goals_np = np.stack([g.disks for g in goals], axis=0)
        return list(np.all(states_np == goals_np, axis=1))

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [self.num_disks], [self.num_pegs]

    def to_np_flat_sg(self, states: List[HanoiState], goals: List[HanoiGoal]) -> List[NDArray]:
        return [np.stack([s.disks for s in states], axis=0).astype(np.uint8)]

    def visualize_state_goal(self, state: HanoiState, goal: HanoiGoal, fig: Figure) -> None:
        ax = fig.add_subplot(111)

        peg_xs = [(p + 1) / (self.num_pegs + 1) for p in range(self.num_pegs)]
        base_y = 0.10
        disk_h = min(0.10, 0.65 / self.num_disks)
        disk_gap = 0.01
        max_w = 0.28
        peg_w = 0.012

        cmap = __import__('matplotlib').colormaps['tab10']
        disk_colors = [cmap(i % 10) for i in range(self.num_disks)]

        disks_per_peg: List[List[int]] = [[] for _ in range(self.num_pegs)]
        for disk_i in range(self.num_disks):
            disks_per_peg[int(state.disks[disk_i])].append(disk_i)

        # base
        ax.add_patch(patches.Rectangle((0.03, base_y - 0.04), 0.94, 0.025,
                                       facecolor='gray', edgecolor='none'))

        # peg rods
        rod_h = base_y + self.num_disks * (disk_h + disk_gap) + 0.04
        for px in peg_xs:
            ax.add_patch(patches.Rectangle((px - peg_w / 2, base_y - 0.015), peg_w, rod_h,
                                           facecolor='dimgray', edgecolor='none'))

        # disks
        for peg_idx, stack in enumerate(disks_per_peg):
            px = peg_xs[peg_idx]
            for pos, disk_i in enumerate(stack):
                frac = (self.num_disks - disk_i) / self.num_disks
                w = max_w * frac + 0.04
                y = base_y + pos * (disk_h + disk_gap)
                color = disk_colors[disk_i]
                on_goal = (peg_idx == int(goal.disks[disk_i]))

                ax.add_patch(patches.Rectangle(
                    (px - w / 2, y), w, disk_h,
                    facecolor=color, edgecolor='limegreen' if on_goal else 'k',
                    linewidth=2.0 if on_goal else 0.8,
                ))
                ax.text(px, y + disk_h / 2, str(disk_i),
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        # peg labels
        goal_peg = int(goal.disks[0]) if np.all(goal.disks == goal.disks[0]) else -1
        for p, px in enumerate(peg_xs):
            suffix = ' *' if p == goal_peg else ''
            ax.text(px, 0.03, f'Peg {p}{suffix}', ha='center', va='center', fontsize=8)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis('off')
        fig.canvas.draw()

    def string_to_action(self, act_str: str) -> Optional[HanoiAction]:
        try:
            parts = act_str.strip().split()
            if len(parts) == 2:
                f, t = int(parts[0]), int(parts[1])
            elif len(parts) == 1 and len(act_str.strip()) == 2:
                f, t = int(act_str.strip()[0]), int(act_str.strip()[1])
            else:
                return None
            if 0 <= f < self.num_pegs and 0 <= t < self.num_pegs and f != t:
                return HanoiAction(f * self.num_pegs + t, f, t)
            return None
        except (ValueError, IndexError):
            return None

    def string_to_action_help(self) -> str:
        return f"move top disk from peg F to peg T: 'F T' or 'FT' (pegs 0-{self.num_pegs - 1})"

    def __repr__(self) -> str:
        return f"Hanoi(num_disks={self.num_disks}, num_pegs={self.num_pegs})"

    def _top_disk_per_peg_single(self, disks: NDArray[np.uint8]) -> NDArray[np.intp]:
        top = np.full(self.num_pegs, self.num_disks, dtype=np.intp)
        for i in range(self.num_disks):
            top[int(disks[i])] = i
        return top

    def _validity_batch(self, batch: NDArray[np.uint8], f: int, t: int) -> Tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
        disk_indices = np.arange(1, self.num_disks + 1, dtype=np.intp)

        on_f = (batch == f)
        any_on_f = on_f.any(axis=1)
        top_f: NDArray[np.intp] = np.where(any_on_f, (on_f * disk_indices).max(axis=1) - 1, self.num_disks)

        on_t = (batch == t)
        any_on_t = on_t.any(axis=1)
        top_t: NDArray[np.intp] = np.where(any_on_t, (on_t * disk_indices).max(axis=1) - 1, self.num_disks)

        valid: NDArray[np.bool_] = any_on_f & (~any_on_t | (top_f > top_t))
        return top_f, top_t, valid


@domain_factory.register_parser("hanoi")
class HanoiParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        parts = args_str.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'num_disks.num_pegs', got '{args_str}'")
        return {"num_disks": int(parts[0]), "num_pegs": int(parts[1])}

    def help(self) -> str:
        return "num_disks.num_pegs. E.g. 'hanoi.4.3'"
