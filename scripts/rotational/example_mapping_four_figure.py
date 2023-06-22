""" Class to Deviate a DS based on an underlying obtacle.
"""
# %%
import sys
import math
import copy
import os
from pathlib import Path
from typing import Optional

# from enum import Enum

import numpy as np
from numpy import linalg as LA
import warnings

from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from vartools.states import Pose
from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem
from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import CircularStable

# from vartools.linalg import get_orthogonal_basis
# from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)


from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import obstacle_avoidance_rotational
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.datatypes import Vector
from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.nonlinear_rotation_avoider import NonlinearRotationalAvoider
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container

from nonlinear_avoidance.nonlinear_rotation_avoider import get_convergence_weight


class CircularEnvironment:
    def __init__(self):
        self.figsize = (5.0, 4.5)

        self.x_lim = [-3.0, 3.4]
        self.y_lim = [-3.0, 3.0]

        self.obstacle_environment = RotationContainer()
        self.center = np.array([2.2, 0.0])

        self.obstacle_environment.append(
            StarshapedFlower(
                center_position=center,
                radius_magnitude=0.2,
                number_of_edges=5,
                radius_mean=0.75,
                orientation=30 / 180 * pi,
                distance_scaling=1,
                # tail_effect=False,
                # is_boundary=True,
            )
        )

        self.attractor_position = np.array([0.0, 0])
        self.circular_ds = SimpleCircularDynamics(
            radius=2.0,
            pose=Pose(
                position=attractor_position,
            ),
        )


def get_main_obstacle():
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        # distance_scaling=1,
        distance_scaling=2.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    return obstacle


def get_initial_dynamics():
    initial_dynamics = LinearSystem(
        attractor_position=np.zeros(2),
        A_matrix=np.array([[-1, -2], [2, -1]]),
        maximum_velocity=1.0,
    )
    return initial_dynamics


def get_environment_obstacle_top_right():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = get_main_obstacle()

    # obstacle = Ellipse(
    #     center_position=center,
    #     axes_length=np.array([2.0, 1.0]),
    #     # orientation=45 * math.pi / 180.0,
    #     # margin_absolut=0.3,
    # )
    initial_dynamics = get_initial_dynamics()
    reference_velocity = np.array([2, 0.1])

    # initial_dynamics = SimpleCircularDynamics(
    #     radius=2.0,
    #     pose=Pose(
    #         position=attractor_position,
    #     ),
    # )

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
        initial_dynamics=initial_dynamics,
    )

    obstacle_environment = RotationContainer()
    obstacle_environment.append(obstacle)
    # Simple Setup
    convergence_dynamics = LinearSystem(attractor_position=attractor_position)
    obstacle_avoider_globally_straight = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
        convergence_system=convergence_dynamics,
    )

    # Convergence direction [local]
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=initial_dynamics.pose.position,
        initial_dynamics=initial_dynamics,
        # reference_velocity=lambda x: x - circular_ds.center_position,
    )

    nonlinear_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
        # Currently not working... -> rotational summing needs to be improved..
        # convergence_radius=math.pi * 3 / 4,
    )

    return dynamics, nonlinear_avoider


def _test_base_gamma(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    visualize_vectors: bool = False,
    n_vectors: int = 8,
    save_figure: bool = False,
    ax=None,
    plot_converence: bool = True,
    **kwargs,
):
    # No explicit test in here, since only getting the gamma value.
    dynamics, avoider = get_environment_obstacle_top_right()

    # if visualize:
    # x_lim = [-10, 10]
    # y_lim = [-10, 10]

    nx = ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if n_resolution > 0:
        gammas = np.zeros(positions.shape[1])
        convergence_weights = np.ones_like(gammas)
        ### Basic Obstacle Transformation ###
        for pp in range(positions.shape[1]):
            gammas[pp] = dynamics.obstacle.get_gamma(
                positions[:, pp], in_global_frame=True
            )

            if gammas[pp] <= 1:
                continue

            # Convergence direction instead
            pos_shrink = positions[:, pp]
            pos_shrink = dynamics._get_position_after_deflating_obstacle(
                pos_shrink,
                in_obstacle_frame=False,
            )
            pos_shrink = dynamics._get_folded_position_opposite_kernel_point(
                pos_shrink,
                attractor_position,
                in_obstacle_frame=False,
            )

            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                pos_shrink,
                in_obstacle_frame=False,
            )

            # convergence_weights[pp] = get_convergence_weight(
            #     dynamics.obstacle.get_gamma(positions[:, pp], in_global_frame=True)
            # )
            convergence_weights[
                pp
            ] = avoider.obstacle_convergence.evaluate_projected_weight(
                positions[:, pp], obstacle=dynamics.obstacle
            )

        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            convergence_weights.reshape(nx, ny),
            cmap="binary",
            alpha=kwargs["weights_alpha"],
            # extend="max",
            # vmin=1.0,
            levels=np.linspace(0, 1, 11),
            zorder=-1,
        )
        # cbar = fig.colorbar(cs, ticks=np.linspace(0, 1.0, 6))

    ax.plot(
        dynamics.attractor_position[0],
        dynamics.attractor_position[1],
        "*",
        color=kwargs["attractor_color"],
        linewidth=12,
        markeredgewidth=0.9,
        markersize=20,
        markeredgecolor="black",
        zorder=3,
    )

    # Opposite point
    ax.plot(
        [dynamics.attractor_position[0], x_lim[0]],
        [dynamics.attractor_position[1], dynamics.attractor_position[1]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"] / 2.0,
        zorder=2,
    )
    # ax.plot(
    #     dynamics.obstacle.center_position[0],
    #     dynamics.obstacle.center_position[1],
    #     "+",
    #     color=kwargs[]
    #     linewidth=12,
    #     markeredgewidth=3.0,
    #     markersize=14,
    #     zorder=3,
    # )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_obstacles(
        ax=ax,
        obstacle_container=[dynamics.obstacle],
        alpha_obstacle=1.0,
        draw_reference=True,
        draw_center=False,
    )

    # Plot the vectors
    nx = ny = n_vectors

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    for pp in range(positions.shape[1]):
        pos = dynamics._get_position_after_inflating_obstacle(
            positions[:, pp], in_obstacle_frame=False
        )
        velocity = dynamics.initial_dynamics.evaluate(pos)
        ax.quiver(
            pos[0],
            pos[1],
            velocity[0],
            velocity[1],
            color=kwargs["initial_color"],
            scale=kwargs["quiver_scale"],
            width=kwargs["quiver_width"],
            zorder=3,
        )

        if plot_converence:
            velocity_rotation = avoider.evaluate_weighted_dynamics(pos, velocity)
            ax.quiver(
                pos[0],
                pos[1],
                # velocity_mod[0],
                # velocity_mod[1],
                velocity_rotation[0],
                velocity_rotation[1],
                color=kwargs["final_color"],
                scale=kwargs["quiver_scale"],
                width=kwargs["quiver_width"],
                zorder=3,
            )

    ax.set_xticks([])
    ax.set_yticks([])

    if save_figure:
        fig.savefig(
            os.path.join(
                # os.path.dirname(__file__),
                "figures",
                kwargs["figure_name"] + "_original" + figtype,
            ),
            bbox_inches="tight",
        )


def _test_obstacle_inflation(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors: int = 8,
    save_figure: bool = False,
    plot_converence: bool = True,
    ax=None,
    **kwargs,
):
    dynamics, avoider = get_environment_obstacle_top_right()

    # x_lim = [-10, 10]
    # y_lim = [-10, 10]
    nx = ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    gammas = np.zeros(positions.shape[1])

    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if n_vectors > 0:
        ### Do before the trafo of the attractor###
        gammas_shrink = np.ones_like(gammas)
        convergence_weights = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                positions[:, pp],
                in_obstacle_frame=False,
            )
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

            # Convergence direction instead
            pos_shrink = positions[:, pp]
            pos_shrink = dynamics._get_folded_position_opposite_kernel_point(
                pos_shrink,
                attractor_position,
                in_obstacle_frame=False,
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                pos_shrink,
                in_obstacle_frame=False,
            )
            # convergence_weights[pp] = get_convergence_weight(
            #     dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
            # )

            convergence_weights[
                pp
            ] = avoider.obstacle_convergence.evaluate_projected_weight(
                pos_shrink, obstacle=dynamics.obstacle
            )

        # cs = ax.contourf(
        #     positions[0, :].reshape(nx, ny),
        #     positions[1, :].reshape(nx, ny),
        #     gammas_shrink.reshape(nx, ny),
        #     cmap="binary",
        #     extend="max",
        #     vmin=1.0,
        #     levels=np.linspace(1, 10, 9),
        # )
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            convergence_weights.reshape(nx, ny),
            cmap="binary",
            alpha=kwargs["weights_alpha"],
            # extend="max",
            # vmin=1.0,
            levels=np.linspace(0, 0.99, 10),
            zorder=-1,
        )
        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

    # Transpose attractor
    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
    )

    ax.plot(
        attractor_position[0],
        attractor_position[1],
        "*",
        color=kwargs["attractor_color"],
        linewidth=12,
        # markeredgewidth=1.2,
        markeredgewidth=0.9,
        markersize=20,
        markeredgecolor="black",
        zorder=3,
        label=r"$f(\xi)",
    )

    # ax.plot(
    #     dynamics.obstacle.center_position[0],
    #     dynamics.obstacle.center_position[1],
    #     "+",
    #     color=kwargs["obstacle_color"],
    #     linewidth=12,
    #     markeredgewidth=3.0,
    #     markersize=14,
    #     zorder=3,
    # )
    ax.plot(
        dynamics.obstacle.center_position[0],
        dynamics.obstacle.center_position[1],
        "k+",
        linewidth=12,
        markeredgewidth=2.4,
        markersize=8,
        zorder=3,
    )

    # Opposite point
    ax.plot(
        [attractor_position[0], x_lim[0]],
        [attractor_position[1], dynamics.attractor_position[1]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"] / 2.0,
        zorder=2,
        label=r"$f(\xi)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if n_vectors:
        # Plot the vectors
        nx = ny = n_vectors

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

        for pp in range(positions.shape[1]):
            pos = dynamics._get_position_after_inflating_obstacle(
                positions[:, pp], in_obstacle_frame=False
            )
            # velocity = dynamics._get_lyapunov_gradient(pos)
            velocity = dynamics.initial_dynamics.evaluate(pos)
            ax.quiver(
                positions[0, pp],
                positions[1, pp],
                velocity[0],
                velocity[1],
                color=kwargs["initial_color"],
                scale=kwargs["quiver_scale"],
                width=kwargs["quiver_width"],
                zorder=3,
            )

            if plot_converence:
                convergence_velocity = avoider.evaluate_weighted_dynamics(pos, velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    convergence_velocity[0],
                    convergence_velocity[1],
                    color=kwargs["final_color"],
                    scale=kwargs["quiver_scale"],
                    width=kwargs["quiver_width"],
                    zorder=3,
                )

        ax.set_xticks([])
        ax.set_yticks([])

        if save_figure:
            # figure_name = "obstacle_deflated_space"
            fig.savefig(
                os.path.join(
                    # os.path.dirname(__file__),
                    "figures",
                    kwargs["figure_name"] + "_deflated" + figtype,
                ),
                bbox_inches="tight",
            )


def _test_inverse_projection_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    plot_converence: bool = True,
    ax=None,
    **kwargs,
):
    dynamics, avoider = get_environment_obstacle_top_right()

    nx = ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    gammas = np.zeros(positions.shape[1])

    ### Do before trafo ###
    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
    )

    gammas_shrink = np.zeros_like(gammas)
    convergence_weights = np.ones_like(gammas)
    for pp in range(positions.shape[1]):
        # Do the reverse operation to obtain an 'even' grid
        pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
            positions[:, pp],
            attractor_position,
            in_obstacle_frame=False,
        )
        pos_shrink = dynamics._get_position_after_inflating_obstacle(
            pos_shrink,
            in_obstacle_frame=False,
        )
        gammas_shrink[pp] = dynamics.obstacle.get_gamma(
            pos_shrink, in_global_frame=True
        )

        # Convergence direction instead
        pos_shrink = positions[:, pp]
        pos_shrink = dynamics._get_position_after_inflating_obstacle(
            pos_shrink,
            in_obstacle_frame=False,
        )
        # convergence_weights[pp] = get_convergence_weight(
        #     dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
        # )
        convergence_weights[
            pp
        ] = avoider.obstacle_convergence.evaluate_projected_weight(
            pos_shrink, obstacle=dynamics.obstacle
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # cs = ax.contourf(
    #     positions[0, :].reshape(nx, ny),
    #     positions[1, :].reshape(nx, ny),
    #     gammas_shrink.reshape(nx, ny),
    #     cmap="binary",
    #     extend="max",
    #     vmin=1.0,
    #     levels=np.linspace(1, 10, 9),
    # )
    cs = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        convergence_weights.reshape(nx, ny),
        cmap="binary",
        alpha=kwargs["weights_alpha"],
        # extend="max",
        # vmin=1.0,
        levels=np.linspace(0, 1, 10),
        zorder=-1,
    )

    # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

    # Attractor line
    ax.plot(
        [x_lim[0], x_lim[0]],
        y_lim,
        kwargs["linestyle"],
        color=kwargs["attractor_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )
    # Split lines
    ax.plot(
        x_lim,
        [y_lim[0], y_lim[0]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )
    ax.plot(
        x_lim,
        [y_lim[1], y_lim[1]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )

    # ax.plot(
    #     dynamics.obstacle.center_position[0],
    #     dynamics.obstacle.center_position[1],
    #     "+",
    #     color=kwargs["obstacle_color"],
    #     linewidth=12,
    #     markeredgewidth=3.0,
    #     markersize=14,
    #     zorder=3,
    # )
    ax.plot(
        dynamics.obstacle.center_position[0],
        dynamics.obstacle.center_position[1],
        "k+",
        linewidth=12,
        markeredgewidth=2.4,
        markersize=8,
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Plot vectors
    if n_vectors:
        # plot the vectors
        nx = ny = n_vectors

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        for pp in range(positions.shape[1]):
            # pos = dynamics._get_position_after_deflating_obstacle(
            #     positions[:, pp], in_obstacle_frame=False
            # )
            pos_unfold = dynamics._get_unfolded_position_opposite_kernel_point(
                positions[:, pp],
                attractor_position,
                in_obstacle_frame=False,
            )

            base_rotation = VectorRotationXd.from_directions(
                pos_unfold - attractor_position,
                dynamics.obstacle.center_position - attractor_position,
            )

            pos_unfold = dynamics._get_position_after_inflating_obstacle(
                pos_unfold,
                in_obstacle_frame=False,
            )
            unrotated_velocity = dynamics.initial_dynamics.evaluate(pos_unfold)
            velocity = base_rotation.rotate(unrotated_velocity)
            # velocity = unrotated_velocity

            ax.quiver(
                positions[0, pp],
                positions[1, pp],
                velocity[0],
                velocity[1],
                color=kwargs["initial_color"],
                scale=kwargs["quiver_scale"],
                width=kwargs["quiver_width"],
                zorder=3,
            )

            if plot_converence:
                convergence_velocity = avoider.evaluate_weighted_dynamics(
                    pos_unfold, unrotated_velocity
                )
                convergence_velocity = base_rotation.rotate(convergence_velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    convergence_velocity[0],
                    convergence_velocity[1],
                    color=kwargs["final_color"],
                    scale=kwargs["quiver_scale"],
                    width=kwargs["quiver_width"],
                    zorder=3,
                )

        ax.set_xticks([])
        ax.set_yticks([])

    if save_figure:
        # figure_name = "obstacle_projection_deflation"
        fig.savefig(
            os.path.join(
                # os.path.dirname(__file__),
                "figures",
                kwargs["figure_name"] + "_unfolded" + figtype,
            ),
            bbox_inches="tight",
        )


def _test_inverse_projection_and_deflation_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    ax=None,
    plot_converence: bool = True,
    **kwargs,
):
    dynamics, avoider = get_environment_obstacle_top_right()

    nx = ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    gammas = np.zeros(positions.shape[1])

    ### do before trafo ###
    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
    )

    gammas_shrink = np.zeros_like(gammas)
    convergence_weights = np.ones_like(gammas)
    for pp in range(positions.shape[1]):
        # do the reverse operation to obtain an 'even' grid
        pos_shrink = positions[:, pp]
        pos_shrink = dynamics._get_position_after_deflating_obstacle(
            pos_shrink, in_obstacle_frame=False
        )
        if np.allclose(pos_shrink, dynamics.obstacle.center_position):
            gammas_shrink[pp] = 1
            convergence_weights[pp] = 1
            continue

        pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
            pos_shrink, attractor_position, in_obstacle_frame=False
        )
        pos_shrink = dynamics._get_position_after_inflating_obstacle(
            pos_shrink, in_obstacle_frame=False
        )
        gammas_shrink[pp] = dynamics.obstacle.get_gamma(
            pos_shrink, in_global_frame=True
        )

        # Convergence direction instead
        pos_shrink = positions[:, pp]
        # convergence_weights[pp] = get_convergence_weight(
        #     dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
        # )
        convergence_weights[
            pp
        ] = avoider.obstacle_convergence.evaluate_projected_weight(
            pos_shrink, obstacle=dynamics.obstacle
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # cs = ax.contourf(
    #     positions[0, :].reshape(nx, ny),
    #     positions[1, :].reshape(nx, ny),
    #     gammas_shrink.reshape(nx, ny),
    #     cmap="binary",
    #     extend="max",
    #     vmin=1.0,
    #     levels=np.linspace(1, 10, 9),
    #     zorder=-1,
    # )
    cs = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        convergence_weights.reshape(nx, ny),
        cmap="binary",
        alpha=kwargs["weights_alpha"],
        # extend="max",
        # vmin=1.0,
        levels=np.linspace(0, 1, 6),
        zorder=-1,
    )

    # attractor line
    ax.plot(
        [x_lim[0], x_lim[0]],
        y_lim,
        kwargs["linestyle"],
        color=kwargs["attractor_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )
    # split lines
    ax.plot(
        x_lim,
        [y_lim[0], y_lim[0]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )
    ax.plot(
        x_lim,
        [y_lim[1], y_lim[1]],
        kwargs["linestyle"],
        color=kwargs["opposite_color"],
        linewidth=kwargs["linewidth"],
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_obstacles(
        ax=ax,
        obstacle_container=[dynamics.obstacle],
        alpha_obstacle=1.0,
        draw_reference=True,
        draw_center=False,
    )

    # Plot vectors
    if n_vectors:
        # plot the vectors
        nx = ny = n_vectors

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        velocities = np.zeros_like(positions)
        for pp in range(positions.shape[1]):
            pos = dynamics._get_position_after_inflating_obstacle(
                positions[:, pp], in_obstacle_frame=False
            )
            pos_unfold = dynamics._get_unfolded_position_opposite_kernel_point(
                positions[:, pp],
                attractor_position,
                in_obstacle_frame=False,
            )

            base_rotation = VectorRotationXd.from_directions(
                pos_unfold - attractor_position,
                dynamics.obstacle.center_position - attractor_position,
            )

            pos_unfold = dynamics._get_position_after_inflating_obstacle(
                pos_unfold,
                in_obstacle_frame=False,
            )
            unrotated_velocity = dynamics.initial_dynamics.evaluate(pos_unfold)
            velocity = base_rotation.rotate(unrotated_velocity)

            ax.quiver(
                pos[0],
                pos[1],
                velocity[0],
                velocity[1],
                color=kwargs["initial_color"],
                scale=kwargs["quiver_scale"],
                width=kwargs["quiver_width"],
                zorder=3,
            )

            if plot_converence:
                convergence_velocity = avoider.evaluate_weighted_dynamics(
                    pos_unfold, unrotated_velocity
                )
                convergence_velocity = base_rotation.rotate(convergence_velocity)
                ax.quiver(
                    pos[0],
                    pos[1],
                    convergence_velocity[0],
                    convergence_velocity[1],
                    color=kwargs["final_color"],
                    scale=kwargs["quiver_scale"],
                    width=kwargs["quiver_width"],
                    zorder=3,
                )

        ax.set_xticks([])
        ax.set_yticks([])

        if save_figure:
            # figure_name = "obstacle_projection_inflated"
            fig.savefig(
                os.path.join(
                    # os.path.dirname(__file__),
                    "figures",
                    kwargs["figure_name"] + "_inflated" + figtype,
                ),
                bbox_inches="tight",
            )


def _test_obstacle_partially_rotated():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([-5.0, 4.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics, avoider = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # test projection
    dist_surf = 1e-6
    pos_close_to_center = copy.deepcopy(dynamics.obstacle.center_position)
    pos_close_to_center[0] = pos_close_to_center[0] + dist_surf

    pos = dynamics._get_position_after_deflating_obstacle(
        pos_close_to_center, in_obstacle_frame=False
    )

    assert np.allclose(pos, dynamics.obstacle.center_position, atol=dist_surf / 2.0)


def plot_mappings_single_plots(save_figure=False):
    # axs[1, 0].quiver(
    #     -10,
    #     -10,
    #     0,
    #     0,
    #     color=setup["initial_color"],
    #     scale=10.0,
    #     width=0.01,
    #     zorder=3,
    #     label="Initial Dynamics",
    # )

    # axs[1, 0].quiver(
    #     -10,
    #     -10,
    #     0,
    #     0,
    #     color=setup["final_color"],
    #     scale=10.0,
    #     width=0.01,
    #     zorder=3,
    #     label="Convergence direction",
    # )

    # axs[1, 0].legend(loc=(0.05, -0.2), ncol=2)

    # sm = ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.get_cmap("binary"))
    # sm.set_array([])
    # cbaxes = inset_axes(axs[1, 1], width="100%", height="5%", loc=1)
    # cbar = fig.colorbar(sm, ax=axs[1:], orientation="horizontal", ticks=[0, 1])
    # cbar.ax.set("Colorbar")

    # if True:
    # return
    # fig_path = Path().absolute() / "figures"
    _test_base_gamma(
        visualize=True,
        visualize_vectors=True,
        save_figure=save_figure,
        # ax=ax,
        # figsize=figsize,
        **setup,
    )

    _test_obstacle_inflation(
        visualize=True,
        **setup,
        save_figure=save_figure,
        # ax=ax,
        # figsize=figsize,
    )

    _test_inverse_projection_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        # ax=ax,
        # figsize=figsize,
    )

    _test_inverse_projection_and_deflation_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        # ax=ax,
        # figsize=figsize,
    )


def plot_four_mappings(save_figure=False):
    # Dummy image - (hidden)
    # axs = [None] * 4
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    fig.subplots_adjust(
        # left=None, bottom=None, right=None, top=None,
        wspace=0.15,
        hspace=0.35,
    )

    axs[1, 0].quiver(
        -10,
        -10,
        0,
        0,
        color=setup["initial_color"],
        scale=10.0,
        width=0.01,
        zorder=3,
        label="Initial Dynamics",
    )

    axs[1, 0].quiver(
        -10,
        -10,
        0,
        0,
        color=setup["final_color"],
        scale=10.0,
        width=0.01,
        zorder=3,
        label="Convergence direction",
    )

    axs[1, 0].legend(loc=(0.05, -0.2), ncol=2)

    # sm = ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.get_cmap("binary"))
    # sm.set_array([])
    # cbaxes = inset_axes(axs[1, 1], width="100%", height="5%", loc=1)
    # cbar = fig.colorbar(sm, ax=axs[1:], orientation="horizontal", ticks=[0, 1])
    # cbar.ax.set("Colorbar")

    # if True:
    # return

    _test_base_gamma(
        visualize=True,
        visualize_vectors=True,
        save_figure=save_figure,
        ax=axs[0, 0],
        **setup,
    )
    _test_obstacle_inflation(
        visualize=True,
        **setup,
        save_figure=save_figure,
        ax=axs[0, 1],
    )
    _test_inverse_projection_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        ax=axs[1, 1],
    )
    _test_inverse_projection_and_deflation_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        ax=axs[1, 0],
    )

    fig.savefig(
        os.path.join(
            "figures",
            setup["figure_name"] + "_subplots" + figtype,
        ),
        bbox_inches="tight",
    )


def plot_single_avoidance():
    fig, ax = plt.subplots(figsize=(5, 4))
    n_resolution = 120
    vf_color = "blue"
    dynamics, avoider = get_environment_obstacle_top_right()

    plot_obstacle_dynamics(
        obstacle_container=avoider.obstacle_environment,
        # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
        dynamics=avoider.evaluate,
        x_lim=setup["x_lim"],
        y_lim=setup["y_lim"],
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        show_ticks=False,
        # do_quiver=True,
        vectorfield_color=vf_color,
        attractor_position=dynamics.attractor_position,
    )

    plot_obstacles(
        ax=ax,
        obstacle_container=avoider.obstacle_environment,
        alpha_obstacle=1.0,
        draw_reference=True,
        draw_center=False,
    )

    fig.savefig(
        os.path.join(
            "figures",
            setup["figure_name"] + "_vectorfield" + figtype,
        ),
        bbox_inches="tight",
    )


class AnimatorConvergence(Animator):
    dimension = 2
    linewidth = 5.0

    trajectory_color = "blue"

    def setup(self, start_position):
        self.fig, self.ax = plt.subplots(figsize=setup["figsize"])

        # Properties
        self.x_lim = setup["x_lim"]
        self.y_lim = setup["y_lim"]
        self.n_vectors = setup["n_vectors"]
        # _test_base_gamma(ax=self.ax, **dynamic_setup)

        self.dynamics = get_initial_dynamics()
        self.create_multiobstacle_avoider()

        self.trajectory = np.zeros((self.dimension, self.it_max + 1))
        self.trajectory[:, 0] = start_position

    def initialize_plot(self, convergence_functor=None, baseline_avoider=None):
        self.ax.clear()
        (self.trajectory_artist,) = self.ax.plot(
            self.trajectory[0, 0],
            self.trajectory[1, 0],
            # color="blue",
            color=self.trajectory_color,
            linewidth=self.linewidth,
        )
        (self.position_artist,) = self.ax.plot(
            self.trajectory[0, 0],
            self.trajectory[1, 0],
            "ko",
            markersize=10.0,
        )

        plot_multi_obstacle_container(
            ax=self.ax,
            container=self.container,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            draw_reference=True,
            noTicks=True,
        )

        # Do initial trajectory
        self.initial_trajectory = np.zeros((self.dimension, self.it_max + 1))
        self.initial_trajectory[:, 0] = self.trajectory[:, 0]
        for ii in range(self.it_max):
            velocity = self.dynamics.evaluate(self.initial_trajectory[:, ii])
            self.initial_trajectory[:, ii + 1] = (
                self.initial_trajectory[:, ii] + velocity * self.dt_simulation
            )

        self.ax.plot(
            self.initial_trajectory[0, :],
            self.initial_trajectory[1, :],
            # "black",
            color=setup["initial_color"],
            alpha=0.5,
            linewidth=self.linewidth,
            zorder=-5,
        )

        if baseline_avoider is not None:
            # Do initial trajectory
            baseline_trajectory = np.zeros((self.dimension, self.it_max + 1))
            baseline_trajectory[:, 0] = self.trajectory[:, 0]

            for ii in range(self.it_max):
                velocity = baseline_avoider.evaluate(baseline_trajectory[:, ii])
                baseline_trajectory[:, ii + 1] = (
                    baseline_trajectory[:, ii] + velocity * self.dt_simulation
                )

            self.ax.plot(
                baseline_trajectory[0, :],
                baseline_trajectory[1, :],
                # "black",
                "--",
                color=self.trajectory_color,
                alpha=0.4,
                linewidth=self.linewidth,
                zorder=-5,
            )

        # Initial
        plot_obstacle_dynamics(
            obstacle_container=self.container,
            dynamics=self.dynamics.evaluate,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            ax=self.ax,
            n_grid=self.n_vectors,
            attractor_position=self.dynamics.attractor_position,
            vectorfield_color=setup["initial_color"],
        )

        # Convergence
        if convergence_functor is None:
            convergence_functor = self.avoider.compute_convergence_direction
        plot_obstacle_dynamics(
            obstacle_container=self.container,
            dynamics=convergence_functor,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            ax=self.ax,
            n_grid=self.n_vectors,
            attractor_position=self.dynamics.attractor_position,
            vectorfield_color=setup["final_color"],
        )

    def create_multiobstacle_avoider(self):
        self.container = MultiObstacleContainer()
        new_tree = MultiObstacle(Pose.create_trivial(self.dimension))
        new_tree.set_root(get_main_obstacle())
        self.container.append(new_tree)

        self.avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
            obstacle_container=self.container,
            initial_dynamics=self.dynamics,
            # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
            create_convergence_dynamics=True,
            # convergence_radius=0.55 * math.pi,
        )

    def update_step(self, ii: int) -> None:
        velocity = self.avoider.evaluate(self.trajectory[:, ii])
        self.trajectory[:, ii + 1] = (
            self.trajectory[:, ii] + velocity * self.dt_simulation
        )

        # self.ax.clear()

        self.position_artist.set_xdata([self.trajectory[0, ii + 1]])
        self.position_artist.set_ydata([self.trajectory[1, ii + 1]])

        self.trajectory_artist.set_xdata(self.trajectory[0, : ii + 2])
        self.trajectory_artist.set_ydata(self.trajectory[1, : ii + 2])
        # self.ax.plot(
        #     self.trajectory[0, ii + 1], self.trajectory[1, ii + 1], "ko", linewidth=2
        # )
        # self.ax.plot(
        #     self.trajectory[0, : ii + 1],
        #     self.trajectory[1, : ii + 1],
        #     color="blue",
        #     linewidth=2,
        # )

        # plot_obstacle_dynamics(
        #     obstacle_container=self.container,
        #     dynamics=self.avoider.evaluate,
        #     x_lim=self.x_lim,
        #     y_lim=self.y_lim,
        #     ax=self.ax,
        #     n_grid=self.n_vectors,
        #     attractor_position=self.dynamics.attractor_position,
        # )


def run_animation(start_position, save_animation=False):
    animator = AnimatorConvergence(
        dt_simulation=0.1,
        dt_sleep=0.1,
        it_max=300,
        animation_name="spiraling_with_local_convergence",
        file_type=".gif",
    )

    animator.setup(start_position=start_position)

    baseline_avoider = MultiObstacleAvoider(
        obstacle_container=animator.container,
        initial_dynamics=animator.dynamics,
        default_dynamics=LinearSystem(animator.dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.53 * math.pi,
    )

    animator.initialize_plot(baseline_avoider=baseline_avoider)
    animator.run(save_animation=save_animation)


def run_animation_with_global_convergence(start_position, save_animation=False):
    animator = AnimatorConvergence(
        dt_simulation=0.05,
        dt_sleep=0.1,
        it_max=300,
        animation_name="spiraling_with_global_convergence",
        file_type=".gif",
    )

    animator.setup(start_position=start_position)
    animator.avoider = MultiObstacleAvoider(
        obstacle_container=animator.container,
        initial_dynamics=animator.dynamics,
        default_dynamics=LinearSystem(animator.dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.53 * math.pi,
    )
    # breakpoint()
    # TODO: there is currently a bug which makes this collide > 2.0
    # animator.avoider.tree_list.get_tree(0).get_component(0).distance_scaling = 1.6
    animator.initialize_plot(
        convergence_functor=LinearSystem(
            attractor_position=animator.dynamics.attractor_position,
            maximum_velocity=1.0,
        ).evaluate
    )
    animator.run(save_animation=save_animation)


def evaluate_projected_velocity(save_figure=False):
    setup["figure_name"] = "mappin_initial_only"
    _test_base_gamma(
        visualize=True,
        visualize_vectors=True,
        save_figure=save_figure,
        # ax=axs[0, 0],
        plot_converence=False,
        **setup,
    )

    _test_obstacle_inflation(
        visualize=True,
        **setup,
        save_figure=save_figure,
        plot_converence=False,
        # ax=axs[0, 1],
    )

    _test_inverse_projection_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        plot_converence=False,
        # ax=axs[1, 1],
    )

    _test_inverse_projection_and_deflation_around_obstacle(
        visualize=True,
        **setup,
        save_figure=save_figure,
        plot_converence=False,
        # ax=axs[1, 0],
    )

    setup["figure_name"] = "mapping_initial_and_convergence"

    _test_base_gamma(
        visualize=True,
        visualize_vectors=True,
        save_figure=save_figure,
        # ax=axs[0, 0],
        plot_converence=True,
        **setup,
    )


if (__name__) == "__main__":
    # TODO: remove '_test_*" functions
    setup = {
        "attractor_color": "#BD5E11",
        "opposite_color": "#4E8212",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3ff",
        "final_color": "#30a0b3ff",
        "figsize": (5, 4),
        # "figsize": (16, 14),
        # "x_lim": [-3.0, 3.4],
        "x_lim": [-2.5, 4.0],
        "y_lim": [-3.0, 3.0],
        "n_resolution": 60,
        # "n_vectors": 10,
        "quiver_scale": 13,
        "quiver_width": 0.01,
        "n_vectors": 9,
        "linestyle": "--",
        "linewidth": 10,
        "figure_name": "linear_spiral_motion",
        "weights_alpha": 0.7,
    }

    # figtype = ".pdf"
    figtype = ".png"

    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    # plot_four_mappings()
    # plot_single_avoidance()
    # plot_mappings_single_plots(save_figure=True)

    # start_position = np.array([3.5, -2])
    # run_animation(start_position=start_position, save_animation=True)
    # run_animation_with_global_convergence(
    #     start_position=start_position, save_animation=True
    # )
    # evaluate_projected_velocity(save_figure=True)
