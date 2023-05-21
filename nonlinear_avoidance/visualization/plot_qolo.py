from pathlib import Path
from typing import Optional, Callable

import numpy as np

from scipy import ndimage
import matplotlib.pyplot as plt


def plot_qolo(position, direction, ax):
    arr_img = plt.imread(Path("media", "Qolo_T_CB_top_bumper.png"))

    length_x = 1.2
    length_y = (1.0) * arr_img.shape[0] / arr_img.shape[1] * length_x

    rot = np.arctan2(direction[1], direction[0])
    arr_img_rotated = ndimage.rotate(
        arr_img,
        rot * 180.0 / np.pi,
        # cval=[255, 255, 255, 0]
        cval=255,
        # mode="closest",
    )

    length_x_rotated = np.abs(np.cos(rot)) * length_x + np.abs(np.sin(rot)) * length_y

    length_y_rotated = np.abs(np.sin(rot)) * length_x + np.abs(np.cos(rot)) * length_y

    ax.imshow(
        arr_img_rotated,
        extent=[
            position[0] - length_x_rotated / 2.0,
            position[0] + length_x_rotated / 2.0,
            position[1] - length_y_rotated / 2.0,
            position[1] + length_y_rotated / 2.0,
        ],
        zorder=1,
    )


def integrate_with_qolo(
    start_position,
    velocity_functor,
    it_max=70,
    dt=0.1,
    ax=None,
    attractor_position=None,
    tol_conv: float = 0.05,
    show_qolo: bool = True,
) -> np.ndarray:
    dimension = 2
    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = start_position

    for pp in range(it_max):
        velocity = velocity_functor(positions[:, pp])
        positions[:, pp + 1] = positions[:, pp] + velocity * dt

        if attractor_position is not None:
            if np.linalg.norm(attractor_position - positions[:, pp + 1]) < tol_conv:
                # Cut the positions
                positions = positions[:, : pp + 2]
                break

    if ax is not None:
        # Plot trajectory & Qolo
        ax.plot(
            positions[0, :], positions[1, :], color="#DB4914", linewidth=4.0, zorder=-2
        )
        # ax.plot(positions[0, :], positions[1, :], ".", linewidth=2.0)
        ax.plot(positions[0, 0], positions[1, 0], marker="o", color="k")

        if show_qolo:
            plot_qolo(ax=ax, position=positions[:, -1], direction=velocity)

    return positions
