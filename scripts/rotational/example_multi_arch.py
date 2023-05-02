import math
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import QuadraticAxisConvergence, LinearSystem
from vartools.dynamics import WavyRotatedDynamics

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.arch_obstacle import MultiObstacleContainer, BlockArchObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.visualization.plot_multi_obstacle import plot_multi_obstacles


def plot_qolo(position, direction, ax):
    arr_img = plt.imread(Path("media", "Qolo_T_CB_top_bumper.png"))

    length_x = 1.2
    length_y = (1.0) * arr_img.shape[0] / arr_img.shape[1] * length_x

    rot = np.arctan2(direction[1], direction[0])
    arr_img_rotated = ndimage.rotate(arr_img, rot * 180.0 / np.pi, cval=255)

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
        zorder=-2,
    )


def integrate_with_qolo(start_position, velocity_functor, it_max=70, dt=0.1, ax=None):
    dimension = 2
    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = start_position

    for pp in range(it_max):
        velocity = velocity_functor(positions[:, pp])
        positions[:, pp + 1] = positions[:, pp] + velocity * dt

    if ax is not None:
        # Plot trajectory & Qolo
        ax.plot(positions[0, :], positions[1, :], color="#DB4914", linewidth=4.0)
        # ax.plot(positions[0, :], positions[1, :], ".", linewidth=2.0)
        ax.plot(positions[0, 0], positions[1, 0], marker="o", color="k")
        plot_qolo(ax=ax, position=positions[:, -1], direction=velocity)

    return positions


def visualize_double_arch(save_figure=False):
    # x_lim = [-7, 7]
    # y_lim = [-6, 6]
    x_lim = [-6.5, 6.5]
    y_lim = [-5.5, 5.5]
    n_grid = 80
    figsize = (4, 3.5)

    margin_absolut = 0.5

    attractor = np.array([4.0, -3])
    # initial_dynamics = LinearSystem(
    #     attractor_position=attractor,
    #     maximum_velocity=1.0,
    # )

    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=attractor, orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )
    container = MultiObstacleContainer()
    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.5]),
            pose=Pose(np.array([-1.5, -3.5]), orientation=90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )

    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.0]),
            pose=Pose(np.array([1.5, 3.0]), orientation=-90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        # convergence_dynamics=rotation_projector,
        create_convergence_dynamics=True,
    )

    kwargs_quiver = {
        # "color": "red",
        "scale": 18,
        "alpha": 1,
        "width": 0.007,
    }

    collision_checker = lambda pos: (not container.is_collision_free(pos))
    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=[],
        dynamics=initial_dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        attractor_position=initial_dynamics.pose.position,
        collision_check_functor=None,
        do_quiver=False,
        show_ticks=False,
        kwargs_quiver=kwargs_quiver,
    )
    # plot_multi_obstacles(ax=ax, container=container)

    start_position = np.array([-2.5, 3])
    integrate_with_qolo(
        start_position=start_position, velocity_functor=initial_dynamics.evaluate, ax=ax
    )

    if save_figure:
        fig_name = "two_arch_avoidance_initial"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        attractor_position=initial_dynamics.pose.position,
        collision_check_functor=collision_checker,
        do_quiver=False,
        show_ticks=False,
    )
    plot_multi_obstacles(ax=ax, container=container)

    integrate_with_qolo(
        start_position=start_position, velocity_functor=avoider.evaluate, ax=ax
    )

    if save_figure:
        fig_name = "two_arch_avoidance"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    # # Test collision free value
    # position = np.array([-4.5, 0.8])
    # is_colliding = collision_checker(position)
    # assert is_colliding


if (__name__) == "__main__":
    figtype = ".pdf"
    visualize_double_arch(save_figure=True)