from typing import List, Dict, Tuple, Optional, cast
from deepxube.base.domain import State, Action, Goal, GoalStartRevWalkableActsRev, NextStateNPActsEnumFixed, StateGoalVizable, StringToAct
from deepxube.base.nnet_input import HasFlatSGActsEnumFixedIn, HasFlatSGAIn
from deepxube.factories.domain_factory import domain_factory
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


import numpy as np

from numpy.typing import NDArray


class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: NDArray[np.uint8]) -> None:
        self.colors: NDArray[np.uint8] = colors
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
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
    def __init__(self, action: int) -> None:
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
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


class Quaternion:
    """Quaternion Rotation:

    Class to aid in representing 3D rotations via quaternions.
    """

    @classmethod
    def from_v_theta(cls, v, theta) -> 'Quaternion':  # type: ignore
        """
        Construct quaternions from unit vectors v and rotation angles theta

        Parameters
        ----------
        v : array_like
            array of vectors, last dimension 3. Vectors will be normalized.
        theta : array_like
            array of rotation angles in radians, shape = v.shape[:-1].

        Returns
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x: NDArray = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x: NDArray):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self) -> str:
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        # multiplication of two quaternions.
        # we don't implement multiplication by a scalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1]
                         - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0]
                         + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3]
                         + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2]
                         - prod[2, 1] + prod[3, 0])],
                       dtype=float,
                       order='F').T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self) -> Tuple[NDArray, NDArray]:
        """Return the v, theta equivalent of the (normalized) quaternion"""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self) -> NDArray:
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points):  # type: ignore
        rot_mat = self.as_rotation_matrix()
        return np.dot(points, rot_mat.T)


def project_points(points, q: Quaternion, view, vertical) -> NDArray:  # type: ignore
    """Project points using a quaternion q and a view v

    Parameters
    ----------
    points : array_like
        array of last-dimension 3
    q : utils.viz_utils.Quaternion
        quaternion representation of the rotation
    view : array_like
        length-3 vector giving the point of view
    vertical : array_like
        direction of y-axis for view.  An error will be raised if it
        is parallel to the view.

    Returns
    -------
    proj: array_like
        array of projected points: same shape as points.
    """
    if vertical is None:
        vertical = [0, 1, 0]
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponing to vertical
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotate the points
    rot_mat = q.as_rotation_matrix()
    r_pts = np.dot(points, rot_mat.T)

    # project the points onto the view
    dpoint = r_pts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans = list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir),
                     -np.dot(dpoint, zdir)]).transpose(trans)


class InteractiveCube(plt.Axes):  # type: ignore
    # Define some attributes
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    stickerwidth = 0.9
    stickermargin = 0.5 * (1. - stickerwidth)
    stickerthickness = 0.001
    (d1, d2, d3) = (1 - stickermargin,
                    1 - 2 * stickermargin,
                    1 + stickerthickness)
    base_sticker = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_sticker_centroid = np.array([[0, 0, 1 + stickerthickness]])

    def __init__(self, n, colors: NDArray, view=(0, 0, 10), fig=None, **kwargs) -> None:  # type: ignore
        self.colors: NDArray = colors

        # Define rotation angles and axes for the six sides of the cube
        x, y, z = np.eye(3)
        self.rots = [Quaternion.from_v_theta(x, np.asarray(theta)) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(y, np.asarray(theta)) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

        rect = [0, 0.16, 1, 0.84]
        self._move_list: List = []

        self.N = n
        self._prevStates: List = []

        self._view = view
        self._start_rot: Quaternion = Quaternion.from_v_theta(np.asarray((1, -1, 0)), np.asarray(-np.pi / 6))

        self._grey_stickers: List = []
        self._black_stickers: List = []

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        # callbacks = fig.canvas.callbacks.callbacks
        # del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-1.7, 1.5)),
                           ylim=kwargs.get('ylim', (-1.5, 1.7)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())  # type: ignore
        self.yaxis.set_major_formatter(plt.NullFormatter())  # type: ignore

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._tab = False  # tab key pressed

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        self._current_rot: Quaternion = self._start_rot  # current rotation state
        self._face_polys: Optional[List] = None
        self._sticker_polys: Optional[List] = None

        self.plastic_color = 'black'

        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors = ["w", "#ffcf00",
                            "#ff6f00", "#cf0000",
                            "#00008f", "#009f0f",
                            "gray", "none"]

        self._initialize_arrays()

        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)

        self._draw_cube()
        # self._initialize_widgets()

    def set_rot(self, rot: int) -> None:
        if rot == 0:
            self._current_rot = Quaternion.from_v_theta(np.asarray((-0.53180525, 0.83020462, 0.16716299)), np.asarray(0.95063829))
        elif rot == 1:
            self._current_rot = Quaternion.from_v_theta(np.asarray((0.9248325, 0.14011997, -0.35362584)), np.asarray(2.49351394))

        self._draw_cube()

    def _initialize_arrays(self) -> None:
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, stickers
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            rot_mat = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, rot_mat.T)
            stickers_t = np.dot(factor * self.base_sticker
                                + translations, rot_mat.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, rot_mat.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid
                                         + translations, rot_mat.T)
            # colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)
            colors_i = np.arange(i * face_centroids_t.shape[0], (i + 1) * face_centroids_t.shape[0])

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)

            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)

    def rotate(self, rot) -> None:  # type: ignore
        self._current_rot = self._current_rot * rot

    def _project(self, pts):  # type: ignore
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self) -> None:
        stickers = self._project(self._stickers)[:, :, :2]
        faces = self._project(self._faces)[:, :, :2]
        face_centroids = self._project(self._face_centroids[:, :3])
        sticker_centroids = self._project(self._sticker_centroids[:, :3])

        plastic_color = self.plastic_color
        # self._colors[np.ravel_multi_index((0,1,2),(6,N,N))] = 10
        colors = np.asarray(self.face_colors)[self.colors]
        for idx in self._grey_stickers:
            colors[idx] = "grey"
        for idx in self._black_stickers:
            colors[idx] = "k"

        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color, zorder=face_zorders[i])  # type: ignore
                sp = plt.Polygon(stickers[i], facecolor=colors[i], zorder=sticker_zorders[i])  # type: ignore

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            assert self._sticker_polys is not None
            # subsequent call: updater the polygon objects
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def _mouse_press(self, event, event_x=None, event_y=None):  # type: ignore
        if event_x is not None and event_y is not None:
            self._event_xy = (event_x, event_y)
            self._button1 = True
        else:
            self._event_xy = (event.x, event.y)
            if event.button == 1:
                self._button1 = True
            elif event.button == 3:
                self._button2 = True

    def _mouse_release(self, event):  # type: ignore
        self._event_xy = None  # type: ignore
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event, event_x=None, event_y=None):  # type: ignore
        if self._button1 or self._button2:
            if event_x is not None and event_y is not None:
                dx = event_x - self._event_xy[0]
                dy = event_y - self._event_xy[1]
                self._event_xy = (event_x, event_y)
            else:
                dx = event.x - self._event_xy[0]
                dy = event.y - self._event_xy[1]
                self._event_xy = (event.x, event.y)

            if self._button1:
                if self._tab:
                    ax_lr = self._ax_LR_alt
                else:
                    ax_lr = self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD, self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_lr, self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()


@domain_factory.register_class("cube3")
class Cube3(NextStateNPActsEnumFixed[Cube3State, Cube3Action, Cube3Goal],
            GoalStartRevWalkableActsRev[Cube3State, Cube3Action, Cube3Goal],
            HasFlatSGActsEnumFixedIn[Cube3State, Cube3Action, Cube3Goal], HasFlatSGAIn[Cube3State, Cube3Action, Cube3Goal],
            StateGoalVizable[Cube3State, Cube3Action, Cube3Goal], StringToAct[Cube3State, Cube3Action, Cube3Goal]):
    atomic_actions: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

    def __init__(self) -> None:
        super().__init__()
        self.cube_len: int = 3
        self.num_colors: int = 6
        self.num_actions = len(self.atomic_actions)
        self.num_stickers: int = self.num_colors * (self.cube_len ** 2)

        # solved state
        self.goal_colors: NDArray[np.uint8] = (np.arange(0, self.num_stickers, 1,
                                                         dtype=np.uint8) // (self.cube_len ** 2)).astype(np.uint8)

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, NDArray[np.int_]]
        self.rotate_idxs_old: Dict[str, NDArray[np.int_]]

        self.adj_faces: Dict[int, NDArray[np.int_]] = _get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.atomic_actions)
        self.actions: List[Cube3Action] = [Cube3Action(x) for x in range(self.num_actions)]

    def is_solved(self, states: List[Cube3State], goals: List[Cube3Goal]) -> List[bool]:
        states_np: NDArray = np.stack([x.colors for x in states], axis=0)
        goals_np: NDArray = np.stack([x.colors for x in goals], axis=0)
        return cast(List[bool], np.all(states_np == goals_np, axis=1).tolist())

    def get_goal_states(self, num_states: int) -> List[Cube3State]:
        return [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]

    def sample_goal_state_goal_pairs(self, num: int) -> Tuple[List[Cube3State], List[Cube3Goal]]:
        states_goal: List[Cube3State] = [Cube3State(self.goal_colors.copy())] * num
        goals: List[Cube3Goal] = [Cube3Goal(self.goal_colors.copy())] * num

        return states_goal, goals

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [self.num_stickers], [self.num_colors]

    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        return [self.num_stickers, 1], [self.num_colors, self.get_num_acts()]

    def to_np_flat_sg(self, states: List[Cube3State], goals: List[Cube3Goal]) -> List[NDArray]:
        return [np.stack([x.colors for x in states], axis=0).astype(np.uint8)]

    def to_np_flat_sga(self, states: List[Cube3State], goals: List[Cube3Goal],
                       actions: List[Cube3Action]) -> List[NDArray]:
        return self.to_np_flat_sg(states, goals) + [np.expand_dims(np.array(self.actions_to_indices(actions)), 1)]

    def actions_to_indices(self, actions: List[Cube3Action]) -> List[int]:
        return [action_cube3.action for action_cube3 in actions]

    def visualize_state_goal(self, state: Cube3State, goal: Cube3Goal, fig: Figure) -> None:
        interactive_cube: InteractiveCube = InteractiveCube(3, state.colors)
        fig.add_axes(interactive_cube)

    def string_to_action(self, act_str: str) -> Optional[Cube3Action]:
        if act_str in self.atomic_actions:
            return Cube3Action(self.atomic_actions.index(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "<face><dir>. i.e. U1, U-1, faces are U, D, L, R, B, F and dirs are 1 and -1"

    def get_actions_fixed(self) -> List[Cube3Action]:
        return self.actions.copy()

    def rev_action(self, states: List[Cube3State], actions: List[Cube3Action]) -> List[Cube3Action]:
        actions_rev: List[Cube3Action] = []
        for action in actions:
            action_val: int = action.action
            action_val_rev: int
            if action_val % 2 == 0:
                action_val_rev = action_val + 1
            else:
                action_val_rev = action_val - 1
            actions_rev.append(Cube3Action(action_val_rev))

        return actions_rev

    def _states_to_np(self, states: List[Cube3State]) -> List[NDArray[np.uint8]]:
        return [np.stack([x.colors for x in states], axis=0)]

    def _np_to_states(self, states_np: List[NDArray]) -> List[Cube3State]:
        assert len(states_np) == 1
        return [Cube3State(x) for x in states_np[0]]

    def _next_state_np(self, states_np_l: List[NDArray[np.uint8]],
                       actions: List[Cube3Action]) -> Tuple[List[NDArray], List[float]]:
        assert len(states_np_l) == 1
        colors_next_np: NDArray[np.uint8] = states_np_l[0].copy()
        assert colors_next_np.shape[0] == len(actions), f"#states {colors_next_np.shape[0]} != #actions {len(actions)}"

        state_idxs: NDArray = np.arange(0, colors_next_np.shape[0])
        state_idxs = np.expand_dims(state_idxs, 1)

        rotate_idxs_new: NDArray = np.stack([self.rotate_idxs_new[self.atomic_actions[action.action]] for action in actions])
        rotate_idxs_old: NDArray = np.stack([self.rotate_idxs_old[self.atomic_actions[action.action]] for action in actions])
        colors_next_np[state_idxs, rotate_idxs_new] = colors_next_np[state_idxs, rotate_idxs_old]

        return [colors_next_np], [1.0] * len(actions)

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, NDArray[np.int_]], Dict[str, NDArray[np.int_]]]:
        rotate_idxs_new: Dict[str, NDArray[np.int_]] = dict()
        rotate_idxs_old: Dict[str, NDArray[np.int_]] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors: NDArray = np.zeros((6, cube_len, cube_len), dtype=np.int64)
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
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

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
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        return rotate_idxs_new, rotate_idxs_old

    def __repr__(self) -> str:
        return "Cube3()"
