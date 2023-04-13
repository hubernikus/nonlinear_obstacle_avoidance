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

from vartools.dynamical_systems import LinearSystem
from vartools.linalg import get_orthogonal_basis
from vartools.states import ObjectPose
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import CircularStable

# from vartools.linalg import get_orthogonal_basis
# from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles
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
            pose=ObjectPose(
                position=attractor_position,
            ),
        )


def get_environment_obstacle_top_right():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        distance_scaling=1,
        # tail_effect=False,
        # is_boundary=True,
    )
    # obstacle = Ellipse(
    #     center_position=center,
    #     axes_length=np.array([2.0, 1.0]),
    #     # orientation=45 * math.pi / 180.0,
    #     # margin_absolut=0.3,
    # )

    reference_velocity = np.array([2, 0.1])

    # initial_dynamics = SimpleCircularDynamics(
    #     radius=2.0,
    #     pose=ObjectPose(
    #         position=attractor_position,
    #     ),
    # )

    initial_dynamics = LinearSystem(
        attractor_position=np.zeros(2),
        A_matrix=np.array([[-1, -2], [2, -1]]),
        maximum_velocity=1.0,
    )
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
        reference_velocity=lambda x: x - circular_ds.center_position,
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

    gammas = np.zeros(positions.shape[1])
    convergence_weights = np.ones_like(gammas)
    ### Basic Obstacle Transformation ###
    for pp in range(positions.shape[1]):
        gammas[pp] = dynamics.obstacle.get_gamma(positions[:, pp], in_global_frame=True)

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

        convergence_weights[pp] = get_convergence_weight(
            dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

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
            scale=10.0,
            width=0.01,
            zorder=3,
        )

        velocity_rotation = avoider.evaluate_weighted_dynamics(pos, velocity)
        ax.quiver(
            pos[0],
            pos[1],
            # velocity_mod[0],
            # velocity_mod[1],
            velocity_rotation[0],
            velocity_rotation[1],
            color=kwargs["final_color"],
            scale=10.0,
            width=0.01,
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

    ### Do before the trafo ###
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
        convergence_weights[pp] = get_convergence_weight(
            dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
        )

    # Transpose attractor
    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position, in_obstacle_frame=False
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
        levels=np.linspace(0, 0.99, 10),
        zorder=-1,
    )
    # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

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
                scale=10.0,
                width=0.01,
                zorder=3,
            )

            convergence_velocity = avoider.evaluate_weighted_dynamics(pos, velocity)
            ax.quiver(
                positions[0, pp],
                positions[1, pp],
                convergence_velocity[0],
                convergence_velocity[1],
                color=kwargs["final_color"],
                scale=10.0,
                width=0.01,
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
        convergence_weights[pp] = get_convergence_weight(
            dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
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
                scale=10.0,
                width=0.01,
                zorder=3,
            )

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
                scale=10.0,
                width=0.01,
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
        convergence_weights[pp] = get_convergence_weight(
            dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
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
                scale=10.0,
                width=0.01,
                zorder=3,
            )

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
                scale=10.0,
                width=0.01,
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


if (__name__) == "__main__":
    setup = {
        "attractor_color": "#BD5E11",
        "opposite_color": "#4E8212",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3ff",
        "final_color": "#30a0b3ff",
        # "figsize": (5, 4),
        "figsize": (10, 8),
        "x_lim": [-3.0, 3.4],
        "y_lim": [-3.0, 3.0],
        "n_resolution": 100,
        "n_vectors": 10,
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
    plot_mappings_single_plots(save_figure=True)
