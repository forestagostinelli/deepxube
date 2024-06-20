from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt

from deepxube.environments.environment_abstract import Environment, State, Goal


class Quaternion:
    """Quaternion Rotation:

    Class to aid in representing 3D rotations via quaternions.
    """

    @classmethod
    def from_v_theta(cls, v, theta):
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

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
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

    def as_v_theta(self):
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

    def as_rotation_matrix(self):
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

    def rotate(self, points):
        rot_mat = self.as_rotation_matrix()
        return np.dot(points, rot_mat.T)


def project_points(points, q, view, vertical):  # type: ignore
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
    points = np.asarray(points)   # type: ignore
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)   # type: ignore

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponing to vertical
    ydir = np.cross(view, xdir)   # type: ignore
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


def visualize_examples(env: Environment, states: Union[List[State], List[Goal]]):
    states_np = env.visualize(states)

    plt.ion()
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    idx_stop: int
    if states_np.shape[3] == 3:
        axs = [ax1, ax2, ax3, ax4]

        for idx in range(0, len(states_np), 4):
            for ax in axs:
                ax.cla()

            idx_stop = min(idx + 4, len(states_np))
            for idx_ax, idx_show in enumerate(range(idx, idx_stop)):
                ax = axs[idx_ax]
                ax.imshow(states_np[idx_show])
                ax.set_xticks([])
                ax.set_yticks([])

            fig.canvas.draw()
            input("Enter anything: ")

    elif states_np.shape[3] == 6:
        axs1 = [ax1, ax2]
        axs2 = [ax3, ax4]
        axs_cube = [axs1, axs2]

        for idx in range(0, len(states_np), 2):
            for axs in axs_cube:
                for ax in axs:
                    ax.cla()

            idx_stop = min(idx + 2, len(states_np))
            for idx_ax, idx_show in enumerate(range(idx, idx_stop)):
                axs = axs_cube[idx_ax]
                axs[0].imshow(states_np[idx_show, :, :, :3])
                axs[1].imshow(states_np[idx_show, :, :, 3:])

                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.canvas.draw()
            input("Enter anything: ")

    plt.ioff()
    plt.close(fig)
