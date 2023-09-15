"""
Multi-Body obstacles to avoid humansg
"""
from __future__ import annotations  # Self typing

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt
from numpy import linalg

from scipy.spatial.transform import Rotation

import networkx as nx

from vartools.state_filters import PositionFilter, SimpleOrientationFilter
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.rigid_body import RigidBody
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

# from nonlinear_avoidance.multi_body_human import plot_3d_cuboid
# from nonlinear_avoidance.multi_body_human import plot_3d_ellipsoid
from nonlinear_avoidance.multi_body_human import MultiBodyObstacle
from nonlinear_avoidance.multi_body_human import create_2d_human


def _test_2d_human_with_linear(visualize=False):
    # Set arm-orientation
    new_human = create_2d_human()
    # multibstacle_avoider = MultiObstacleAvoider(obstacle=new_human)

    # First with (very) simple dyn
    # velocity = np.array([1.0, 0.0])
    # linearized_velociy = np.array([1.0, 0.0])
    dynamics = LinearSystem(attractor_position=np.array([10, 0.0]))

    container = MultiObstacleContainer()
    container.append(new_human)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=0.55 * math.pi,
        smooth_continuation_power=0.7,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_lim = [-1.3, 1.3]
        y_lim = [-0.25, 1.2]
        n_grid = 20

        plot_obstacles(
            obstacle_container=new_human._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
            # reference_point_number=True,
            show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            n_grid=n_grid,
            show_ticks=False,
        )


def _test_2d_human_with_circular(visualize=False, savefig=False):
    # Set arm-orientation
    new_human = create_2d_human()
    multibstacle_avoider = MultiObstacleAvoider(obstacle=new_human)

    # First with (very) simple dynanmic
    circular_ds = SimpleCircularDynamics(
        radius=1.0,
        pose=ObjectPose(position=np.array([1, 1]), orientation=30.0 / 180 * np.pi),
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - circular_ds.pose.position,
    )

    multibstacle_avoider = MultiObstacleAvoider(
        obstacle=new_human,
        initial_dynamics=circular_ds,
        convergence_dynamics=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(7, 4))
        n_grid = 100

        x_lim = [-1.3, 1.3]
        y_lim = [-0.25, 1.2]

        plot_obstacles(
            obstacle_container=new_human._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
            # reference_point_number=True,
            # show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                new_human.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=multibstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # do_quiver=True,
            do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
            attractor_position=circular_ds.pose.position,
        )

        if savefig:
            figname = "multibody_human"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-1.0, 0.0])
    velocity = multibstacle_avoider.evaluate(position)
    assert velocity[1] > 0, "Expected o be moving upwards."

    position = np.array([1.0, 1.0])
    velocity = multibstacle_avoider.evaluate(position)
    assert np.allclose(velocity, [0, 0]), "No velocity at attractor"
    # breakpoint()


def _test_2d_human_with_circular(visualize=False, savefig=False):
    # Set arm-orientation

    new_human = create_2d_human()

    circular_ds = SimpleCircularDynamics(
        radius=1.0,
        pose=ObjectPose(position=np.array([1, 1]), orientation=30.0 / 180 * np.pi),
    )

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=0.55 * math.pi,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(7, 4))
        n_grid = 100

        x_lim = [-1.3, 1.3]
        y_lim = [-0.25, 1.2]

        plot_obstacles(
            obstacle_container=new_human._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
            # reference_point_number=True,
            # show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                new_human.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=multibstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # do_quiver=True,
            do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
            attractor_position=circular_ds.pose.position,
        )

        if savefig:
            figname = "multibody_human"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )


if (__name__) == "__main__":
    # figtype = ".png"
    figtype = ".pdf"

    import matplotlib.pyplot as plt

    from dynamic_obstacle_avoidance.visualization import plot_obstacles
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )

    # plt.close("all")
    plt.ion()

    # TODO: these tests are kind of deactivated, if needed they should be adapted

    _test_2d_human_with_linear(visualize=True)
    # _test_2d_human_with_circular(visualize=True, savefig=False)

    print("[INFO] Done.")
