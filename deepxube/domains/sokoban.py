from typing import Tuple, List, Optional, Dict, cast

import numpy as np
from numpy.typing import NDArray
from deepxube.base.nnet_input import FlatIn, StateGoalIn
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StateGoalVizable, StringToAct
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from matplotlib.figure import Figure

import pickle

import pathlib
import tarfile
import os
import wget  # type: ignore
from filelock import FileLock


class SkState(State):
    __slots__ = ['agent', 'boxes', 'walls', 'hash']

    def __init__(self, agent: NDArray[np.int_], boxes: NDArray[np.uint8], walls: NDArray[np.uint8]):
        self.agent: NDArray[np.int_] = agent
        self.boxes: NDArray[np.uint8] = boxes
        self.walls: NDArray[np.uint8] = walls

        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            boxes_flat = self.boxes.flatten()
            walls_flat = self.walls.flatten()
            state: NDArray[np.int_] = np.concatenate((self.agent, boxes_flat.astype(int), walls_flat.astype(int)), axis=0)

            self.hash = hash(state.tobytes())

        return self.hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SkState):
            agents_eq: bool = np.array_equal(self.agent, other.agent)
            boxes_eq: bool = np.array_equal(self.boxes, other.boxes)
            walls_eq: bool = np.array_equal(self.walls, other.walls)

            return agents_eq and boxes_eq and walls_eq
        return NotImplemented


class SkGoal(Goal):
    __slots__ = ['boxes']

    def __init__(self, boxes: NDArray[np.uint8]):
        self.boxes: NDArray[np.uint8] = boxes


class SkAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SkAction):
            return self.action == other.action
        return NotImplemented


def load_states(data_dir: str) -> List[SkState]:
    states_np = pickle.load(open(f"{data_dir}/sokoban/train.pkl", "rb"))
    # t_file = tarfile.open(file_name, "r:gz")
    # states_np = pickle.load(t_file.extractfile(t_file.getmembers()[1]))

    states: List[SkState] = []

    agent_idxs: Tuple[NDArray, ...] = np.where(states_np == 1)
    box_masks: NDArray = np.array(states_np == 2)
    wall_masks: NDArray = np.array(states_np == 4)

    idx: int
    for idx in range(states_np.shape[0]):
        agent_idx = np.array([agent_idxs[1][idx], agent_idxs[2][idx]], dtype=int)

        states.append(SkState(agent_idx, box_masks[idx], wall_masks[idx]))

    return states


def _get_surfaces() -> Dict[str, NDArray]:
    import imageio.v2 as imageio
    data_dir = get_data_dir()
    img_dir = f"{data_dir}/sokoban/"

    lock = FileLock(f"{data_dir}/file.lock")
    with lock:
        # Load images, representing the corresponding situation
        surface_dict: Dict[str, NDArray] = dict()
        for img_name in ["box", "box_on_target", "box_target", "floor", "player", "player_on_target", "wall"]:
            surface_dict[img_name] = imageio.imread(f"{img_dir}/surface/{img_name}.png")

    return surface_dict


def _get_train_states() -> List[SkState]:
    data_dir = get_data_dir()
    lock = FileLock(f"{data_dir}/file.lock")

    with lock:
        states_train: List[SkState] = load_states(data_dir)

    return states_train


def get_data_dir() -> str:
    parent_dir: str = str(pathlib.Path(__file__).parent.resolve())
    data_dir: str = f"{parent_dir}/data/sokoban/"

    return data_dir


@domain_factory.register_class("sokoban")
class Sokoban(ActsEnumFixed[SkState, SkAction, SkGoal], StartGoalWalkable[SkState, SkAction, SkGoal], StateGoalVizable[SkState, SkAction, SkGoal],
              StringToAct[SkState, SkAction, SkGoal]):
    def __init__(self) -> None:
        super().__init__()

        self.dim: int = 10
        self.num_boxes: int = 4

        self.num_actions: int = 4

        self.states_train: Optional[List[SkState]] = None
        self._surfaces: Optional[Dict[str, NDArray]] = None
        self.actions: List[SkAction] = [SkAction(x) for x in range(self.num_actions)]

        # check if data needs to be downloaded
        data_dir = get_data_dir()
        data_download_link: str = "https://github.com/forestagostinelli/DeepXubeData/raw/main/sokoban.tar.gz"
        if not os.path.exists(f"{data_dir}/sokoban/"):
            valid_user_in: bool = False
            while not valid_user_in:
                user_in: str = input(f"Sokoban data needs to be downloaded from {data_download_link}. "
                                     f"Download data (about 16MB)? (y/n):")
                if user_in.upper() == "Y":
                    valid_user_in = True
                    print("Downloading compressed data")
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)

                    tar_gz_file_name = f"{data_dir}/sokoban.tar.gz"
                    wget.download(data_download_link, tar_gz_file_name, bar=None)
                    tar_gz_file = tarfile.open(tar_gz_file_name)
                    print("Uncompressing data")
                    tar_gz_file.extractall(data_dir)
                    print("Deleting compressed data")
                    os.remove(tar_gz_file_name)
                elif user_in.upper() == "N":
                    valid_user_in = True

    def next_state(self, states: List[SkState], actions: List[SkAction]) -> Tuple[List[SkState], List[float]]:
        agent = np.stack([state.agent for state in states], axis=0)
        boxes = np.stack([state.boxes for state in states], axis=0)
        walls_next = np.stack([state.walls for state in states], axis=0)

        idxs_arange = np.arange(0, len(states))
        agent_next_tmp = self._get_next_idx(agent, actions)
        agent_next = np.zeros(agent_next_tmp.shape, dtype=int)

        boxes_next = boxes.copy()

        # agent -> wall
        agent_wall = walls_next[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        agent_next[agent_wall] = agent[agent_wall]

        # agent -> box
        agent_box = boxes[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        boxes_next_tmp = self._get_next_idx(agent_next_tmp, actions)

        box_wall = walls_next[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]
        box_box = boxes[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]

        # agent -> box -> obstacle
        agent_box_obstacle = agent_box & (box_wall | box_box)
        agent_next[agent_box_obstacle] = agent[agent_box_obstacle]

        # agent -> box -> empty
        agent_box_empty = agent_box & np.logical_not(box_wall | box_box)
        agent_next[agent_box_empty] = agent_next_tmp[agent_box_empty]
        abe_idxs = np.where(agent_box_empty)[0]

        agent_next_idxs_abe = agent_next[agent_box_empty]
        boxes_next_idxs_abe = boxes_next_tmp[agent_box_empty]

        boxes_next[abe_idxs, agent_next_idxs_abe[:, 0], agent_next_idxs_abe[:, 1]] = False
        boxes_next[abe_idxs, boxes_next_idxs_abe[:, 0], boxes_next_idxs_abe[:, 1]] = True

        # agent -> empty
        agent_empty = np.logical_not(agent_wall | agent_box)
        agent_next[agent_empty] = agent_next_tmp[agent_empty]
        boxes_next[agent_empty] = boxes[agent_empty]

        states_next: List[SkState] = []
        for idx in range(len(states)):
            state_next: SkState = SkState(agent_next[idx], boxes_next[idx], walls_next[idx])
            states_next.append(state_next)

        transition_costs: List[float] = [1.0 for _ in range(len(states))]

        return states_next, transition_costs

    def get_actions_fixed(self) -> List[SkAction]:
        return self.actions.copy()

    def is_solved(self, states: List[SkState], goals: List[SkGoal]) -> List[bool]:
        boxes_states: NDArray = np.stack([state.boxes for state in states], axis=0)
        targets: NDArray = np.stack([goal.boxes for goal in goals], axis=0)
        return cast(List[bool], np.all(boxes_states == targets, axis=(1, 2)).tolist())

    def sample_start_states(self, num_states: int) -> List[SkState]:
        # get states
        if self.states_train is None:
            self.states_train = _get_train_states()
        state_idxs = np.random.randint(0, len(self.states_train), size=num_states)
        states: List[SkState] = [self.states_train[idx] for idx in state_idxs]

        # random walk
        step_range: Tuple[int, int] = (0, 100)

        steps_range: List[int] = list(range(step_range[0], step_range[1] + 1))
        step_nums: List[int] = np.random.choice(steps_range, num_states).tolist()

        return self.random_walk(states, step_nums)[0]

    def sample_goal_from_state(self, states_start: Optional[List[SkState]], states_goal: List[SkState]) -> List[SkGoal]:
        goals: List[SkGoal] = []
        for state_goal in states_goal:
            goals.append(SkGoal(state_goal.boxes))

        return goals

    def string_to_action(self, act_str: str) -> Optional[SkAction]:
        act_str_to_act: Dict[str, SkAction] = {"w": SkAction(0), "s": SkAction(1), "a": SkAction(2), "d": SkAction(3)}
        if act_str in act_str_to_act.keys():
            return act_str_to_act[act_str]
        else:
            return None

    def string_to_action_help(self) -> str:
        return "w, d, a, d: up, down, left, right"

    def visualize_state_goal(self, state: SkState, goal: SkGoal, fig: Figure) -> None:
        room_rgb = self.to_img(state, goal)

        ax = fig.add_subplot(111)
        ax.imshow(room_rgb)

    def to_img(self, state: SkState, goal: SkGoal) -> NDArray:
        if self._surfaces is None:
            self._surfaces = _get_surfaces()

        room_rgb: NDArray[np.uint8] = np.zeros(shape=(self.dim * 16, self.dim * 16, 3), dtype=np.uint8)
        for i in range(self.dim):
            x_i = i * 16

            for j in range(self.dim):
                y_j = j * 16

                surface_str: str
                if state.walls[i, j]:
                    surface_str = "wall"
                elif (state.agent[0] == i) and (state.agent[1] == j):
                    if goal.boxes[i, j]:
                        surface_str = "player_on_target"
                    else:
                        surface_str = "player"
                elif state.boxes[i, j]:
                    if goal.boxes[i, j]:
                        surface_str = "box_on_target"
                    else:
                        surface_str = "box"
                elif goal.boxes[i, j]:
                    surface_str = "box_target"
                else:
                    surface_str = "floor"
                room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = self._surfaces[surface_str]

        # img = Image.fromarray(room_rgb, 'RGB')
        # img = img.resize((self.img_dim, self.img_dim))

        return room_rgb

    def _get_next_idx(self, curr_idxs: NDArray[np.int_], actions: List[SkAction]) -> NDArray[np.int_]:
        actions_np: NDArray[np.int_] = np.array([action.action for action in actions])
        next_idxs: NDArray[np.int_] = curr_idxs.copy()

        action_idxs: NDArray[np.int_] = np.where(actions_np == 0)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] - 1

        action_idxs = np.where(actions_np == 1)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] + 1

        action_idxs = np.where(actions_np == 2)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] - 1

        action_idxs = np.where(actions_np == 3)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] + 1

        next_idxs = np.maximum(next_idxs, 0)
        next_idxs = np.minimum(next_idxs, self.dim - 1)

        return next_idxs

    def __getstate__(self) -> Dict:
        self.states_train = None
        self._surfaces = None

        return self.__dict__

    def __repr__(self) -> str:
        return "Sokoban"


@register_nnet_input("sokoban", "sokoban_nnet_input")
class SkNNetInput(FlatIn[Sokoban], StateGoalIn[Sokoban, SkState, SkGoal]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return [400], [1]

    def to_np(self, states: List[SkState], goals: List[SkGoal]) -> List[NDArray]:
        walls: NDArray = np.stack([state.walls for state in states], axis=0)
        boxes: NDArray = np.stack([state.boxes for state in states], axis=0)
        targets: NDArray = np.stack([goal.boxes for goal in goals], axis=0)
        agent_locs: NDArray = np.stack([state.agent for state in states], axis=0)
        agents: NDArray = np.zeros((len(states), self.domain.dim, self.domain.dim))
        agents[np.arange(0, len(states)), agent_locs[:, 0], agent_locs[:, 1]] = 1

        rep_np: NDArray = np.stack([walls, boxes, agents, targets], axis=1)
        rep_np = np.reshape(rep_np, (rep_np.shape[0], -1)).astype(np.uint8)
        return [rep_np]
