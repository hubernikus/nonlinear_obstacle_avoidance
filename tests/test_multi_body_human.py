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


def test_2d_human_with_linear(visualize=False):
    # Set arm-orientation
    new_human = create_2d_human()
    multibstacle_avoider = MultiObstacleAvoider(obstacle=new_human)

    # First with (very) simple dyn
    velocity = np.array([1.0, 0.0])
    linearized_velociy = np.array([1.0, 0.0])

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
            collision_check_functor=lambda x: (
                new_human.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
            # vectorfield_color=vf_color,
        )
    position = np.array([-0.201, -0.2])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )

    # assert (
    #     averaged_direction[1] > 0 and abs(averaged_direction[0]) < 0.1
    # ), "Not tangent to surface in front of obstacle."

    position = np.array([-0.3, -0.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] > 0

    position = np.array([-0.6, 1.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0

    position = np.array([0.4, 0.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] < 0


def test_2d_human_with_circular(visualize=False, savefig=False):
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


def test_2d_blocky_arch(visualize=False):
    dimension = 2

    multi_block = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )

    multi_block.set_root(
        Cuboid(axes_length=[1.0, 4.0], center_position=np.zeros(dimension)),
        name=0,
    )

    multi_block.add_component(
        Cuboid(axes_length=[4.0, 1.0], center_position=np.zeros(dimension)),
        name=1,
        parent_name=0,
        reference_position=[1.5, 0.0],
        parent_reference_position=[0.0, 1.5],
    )

    multi_block.add_component(
        Cuboid(axes_length=[4.0, 1.0], center_position=np.zeros(dimension)),
        name=1,
        parent_name=0,
        reference_position=[1.5, 0.0],
        parent_reference_position=[0.0, -1.5],
    )

    multi_block.update()

    multibstacle_avoider = MultiObstacleAvoider(obstacle=multi_block)

    velocity = np.array([1.0, 0])
    linearized_velociy = np.array([1.0, 0])

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-7, 3.0]
        y_lim = [-5, 5.0]
        n_grid = 20

        plot_obstacles(
            obstacle_container=multi_block._obstacle_list,
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
            collision_check_functor=lambda x: (
                multi_block.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
            # vectorfield_color=vf_color,
        )

    # Test positions [which has been prone to a rounding error]
    position = np.array([-3.8, 1.6])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] > 0

    position = np.array([-2.0, -2.02])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert np.allclose(averaged_direction, [1, 0], atol=1e-2)


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

    # test_2d_blocky_arch(visualize=True)
    test_2d_human_with_linear(visualize=True)
    # test_2d_human_with_circular(visualize=True, savefig=False)

    print("[INFO] Done.")
