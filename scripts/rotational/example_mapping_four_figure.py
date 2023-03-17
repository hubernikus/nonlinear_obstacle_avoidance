""" Class to Deviate a DS based on an underlying obtacle.
"""
# %%
import sys
import math
import copy
import os
from typing import Optional

# from enum import Enum

import numpy as np
from numpy import linalg as LA
import warnings

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

from roam.avoidance import obstacle_avoidance_rotational
from roam.rotation_container import RotationContainer
from roam.vector_rotation import VectorRotationXd
from roam.datatypes import Vector
from roam.dynamics.circular_dynamics import SimpleCircularDynamics
from roam.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

from roam.nonlinear_rotation_avoider import get_convergence_weight


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
                orientation=33 / 180 * pi,
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
        orientation=10 / 180 * pi,
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

    circular_ds = SimpleCircularDynamics(
        radius=2.0,
        pose=ObjectPose(
            position=attractor_position,
        ),
    )
    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
        initial_dynamics=circular_ds,
    )

    return dynamics


def _test_base_gamma(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    visualize_vectors: bool = False,
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    # No explicit test in here, since only getting the gamma value.
    dynamics = get_environment_obstacle_top_right()

    if visualize:
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

            convergence_weights[pp] = get_convergence_weight(
                dynamics.obstacle.get_gamma(pos_shrink, in_global_frame=True)
            )

        fig, ax = plt.subplots(figsize=figsize)
        # cs = ax.contourf(
        #     positions[0, :].reshape(nx, ny),
        #     positions[1, :].reshape(nx, ny),
        #     gammas.reshape(nx, ny),
        #     cmap="binary",
        #     extend="max",
        #     vmin=1.0,
        #     levels=np.linspace(1, 10, 9),
        # )
        # cs = ax.imshow(
        #     convergence_weights.reshape(nx, ny),
        #     cmap="binary",
        #     alpha=0.5,
        #     # extent=extent
        # )

        # breakpoint()
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
        cbar = fig.colorbar(cs, ticks=np.linspace(0, 1.0, 6))

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

        if visualize_vectors:
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
                    zorder=1,
                )

                # velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                #     pos, velocity
                # )
                velocity_rotation = dynamics.evaluate_convergence_around_obstacle(
                    pos, dynamics.obstacle
                )

                # velocity_mod = velocity_rotation.rotate(velocity)
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


def test_obstacle_inflation(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
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
            label=r"$w^c",
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
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
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
        return fig, ax


def test_inverse_projection_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
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
                pos = dynamics._get_position_after_deflating_obstacle(
                    positions[:, pp], in_obstacle_frame=False
                )
                velocity = dynamics._get_projected_lyapunov_gradient(pos)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                # figure_name = "obstacle_projection_deflation"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        kwargs["figure_name"] + "_deflated" + figtype,
                    ),
                    bbox_inches="tight",
                )

    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position,
        in_obstacle_frame=False,
    )

    position = np.array([-1.5, -20])
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position,
        attractor_position,
        in_obstacle_frame=False,
    )


def test_inverse_projection_and_deflation_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
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

                pos_revert = positions[:, pp]
                pos_revert = dynamics._get_unfolded_position_opposite_kernel_point(
                    pos_revert, attractor_position, in_obstacle_frame=False
                )
                pos_revert = dynamics._get_position_after_inflating_obstacle(
                    pos_revert, in_obstacle_frame=False
                )

                velocity = dynamics.initial_dynamics.evaluate(pos_revert)
                base_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos_revert, dynamics.obstacle.center_position - attractor_position
                )

                print("vel", velocity)
                velocity = base_rotation.rotate(velocity)
                print("vel a", velocity)

                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                )
                breakpoint()
                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos,
                    velocity,
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )
            breakpoint()
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


def test_obstacle_partially_rotated():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([-5.0, 4.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
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


def plot_four_mappings():
    setup = {
        "attractor_color": "#BD5E11",
        "opposite_color": "#4E8212",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3",
        "final_color": "#30a0b3",
        "figsize": (5, 4),
        "x_lim": [-3.0, 3.4],
        "y_lim": [-3.0, 3.0],
        "n_resolution": 100,
        "n_vectors": 12,
        "linestyle": "--",
        "linewidth": 10,
        "figure_name": "circular_motion",
        "weights_alpha": 0.7,
    }
    # _test_base_gamma(visualize=True, visualize_vectors=True, save_figure=True, **setup)
    # test_obstacle_inflation(visualize=True, **setup, save_figure=True)
    # test_inverse_projection_around_obstacle(visualize=True, **setup, save_figure=True)
    test_inverse_projection_and_deflation_around_obstacle(
        visualize=1, **setup, save_figure=True
    )


if (__name__) == "__main__":
    figtype = "pdf"
    figtype = "png"

    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    plot_four_mappings()
