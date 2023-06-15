"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
from typing import Optional, Protocol

import numpy as np
from numpy import linalg

from vartools.math import get_intersection_with_circle, CircleIntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.states import Pose
from vartools.dynamical_systems import DynamicalSystem, LinearSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from nonlinear_avoidance.multi_ellipse_obstacle import MultiEllipseObstacle
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.nonlinear_rotation_avoider import ObstacleConvergenceDynamics
from nonlinear_avoidance.dynamics.constant_value import ConstantValueWithSequence

from nonlinear_avoidance.datatypes import Vector


def test_triple_ellipse_environment(visualize=False, savefig=False):
    container = MultiObstacleContainer()
    multi_obstacle = MultiObstacle(Pose(np.array([0, 0.0])))
    multi_obstacle.set_root(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    multi_obstacle.add_component(
        Ellipse(
            center_position=np.array([-3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=90 * math.pi / 180.0,
        ),
        parent_ind=0,
    )

    multi_obstacle.add_component(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        ),
        parent_ind=0,
    )

    container.append(multi_obstacle)
    dynamics = LinearSystem(attractor_position=np.array([6.0, 0.0]))

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-14, 14]
        y_lim = [-8, 16]
        fig, ax = plt.subplots(figsize=(8, 5))

        plot_obstacles(
            obstacle_container=multi_obstacle,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            show_obstacle_number=True,
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            n_grid=30,
            # vectorfield_color=vf_color,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_sidewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-3.37931034, 0.27586207])
    gamma_value = container.get_gamma(position, in_global_frame=True)
    assert gamma_value <= 1, "Is in one of the obstacles"

    # Testing various position around the obstacle
    position = np.array([-1.5, 5])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, 5])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0

    position = np.array([5.5, 5])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, -0.9])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0


def test_tripple_ellipse_in_the_face(visualize=False, savefig=False):
    velocity = np.array([1.0, 0.0])
    dynamics = ConstantValueWithSequence(velocity)

    container = MultiObstacleContainer()

    triple_ellipses = MultiObstacle(Pose.create_trivial(2))
    triple_ellipses.set_root(
        Ellipse(
            center_position=np.array([3.4, 4.0]),
            axes_length=np.array([9.0, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.add_component(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        ),
        parent_ind=0,
    )

    triple_ellipses.add_component(
        Ellipse(
            center_position=np.array([0, 7.8]),
            axes_length=np.array([8, 3.0]),
            orientation=0 * math.pi / 180.0,
        ),
        parent_ind=0,
    )
    container.append(triple_ellipses)

    # avoider = MultiObstacleAvoider(obstacle=triple_ellipses)
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    if visualize:
        figsize = (10, 5)
        x_lim = [-12, 12]
        y_lim = [-5, 12.5]

        # n_grid = 120
        n_grid = 20
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=True,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_facewards"
            plt.savefig(
                "figures/" + "obstacles_only_" + figname + figtype,
                bbox_inches="tight",
            )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_facewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-5.0, 0.5])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([6.0, 6.0])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5.0, 9.0])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0


def test_orthonormal_tangent_finding():
    # Do we really want this test (?!) ->  maybe remove or make it better
    normal = np.array([0.99306112, 0.11759934])
    reference = np.array([-0.32339489, -0.9462641])

    initial = np.array([0.93462702, 0.35562949])
    tangent_rot = RotationalAvoider.get_projected_tangent_from_vectors(
        initial,
        normal=normal,
        reference=reference,
    )

    tangent_matr = MultiEllipseObstacle.get_normalized_tangent_component(
        initial, normal=normal, reference=reference
    )
    assert np.allclose(tangent_rot, tangent_matr)

    normal = np.array([0.99306112, 0.11759934])
    reference = np.array([-0.32339489, -0.9462641])
    initial = np.array([0.11759934, -0.99306112])
    tangent_rot = RotationalAvoider.get_projected_tangent_from_vectors(
        initial,
        normal=normal,
        reference=reference,
    )

    tangent_matr = MultiEllipseObstacle.get_normalized_tangent_component(
        initial, normal=normal, reference=reference
    )
    assert np.allclose(tangent_rot, tangent_matr)


def test_tree_with_two_children(visualize=False, savefig=False):
    """This is a rather uncommon configuration as the vectorfield has to traverse back
    since the root obstacle is not at the center."""
    container = MultiObstacleContainer()
    multi_obstacle = MultiObstacle(Pose(np.array([0, 0.0])))
    multi_obstacle.set_root(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        ),
    )

    # multi_obstacle.add_component(
    #     Ellipse(
    #         center_position=np.array([-3.4, 3.4]),
    #         axes_length=np.array([8, 3.0]),
    #         orientation=90 * math.pi / 180.0,
    #     ),
    #     parent_ind=0,
    # )

    multi_obstacle.add_component(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        ),
        parent_ind=0,
    )

    container.append(multi_obstacle)
    dynamics = LinearSystem(attractor_position=np.array([6.0, 0.0]))

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        figsize = (10, 5)
        x_lim = [-12, 12]
        y_lim = [-5, 12.5]

        n_grid = 20
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacles(
            obstacle_container=multi_obstacle,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            # noTicks=False,
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            attractor_position=dynamics.attractor_position
            # vectorfield_color=vf_color,
        )

        if savefig:
            figname = "triple_ellipses_childeschild_obstacle_facewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )
    plot_convergence = True
    if plot_convergence and visualize:
        plot_obstacles(
            obstacle_container=multi_obstacle,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            n_grid=n_grid,
            show_ticks=True,
            attractor_position=dynamics.attractor_position,
            vectorfield_color="green",
        )

    position = np.array([4.41, 6.97])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0
    assert velocity[1] < 0

    position = np.array([2.153, 5.88])
    convergence = avoider.compute_convergence_direction(position)
    assert convergence[0] > 0
    assert np.isclose(convergence[1], 0, atol=1e-1)

    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0
    assert velocity[1] > 0


def test_single_ellipse(visualize=False):
    container = MultiObstacleContainer()
    multi_obstacle = MultiObstacle(Pose(np.array([0, 0.0])))
    multi_obstacle.set_root(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.0, 4.0]),
            # orientation=90 * math.pi / 180.0,
        )
    )
    container.append(multi_obstacle)

    dynamics = LinearSystem(attractor_position=np.array([4.0, 0.0]))

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        figsize = (8, 6)
        x_lim = [-5, 5]
        y_lim = [-5, 5]

        n_grid = 40
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacles(
            obstacle_container=multi_obstacle,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            # vectorfield_color=vf_color,
        )

    # position = np.array([-3.22, 0.374])
    # velocity1 = avoider.evaluate_sequence(position)

    # position = np.array([-3.22, 0.670])
    # velocity2 = avoider.evaluate_sequence(position)
    # assert np.allclose(
    #     velocity1, velocity2, atol=0.5
    # ), "Smooth velocity change with respect to position."

    # Evaluate at position[1]
    position = np.array([-3.0, 0.4])
    velocity1 = avoider.evaluate_sequence(position)
    assert velocity1[0] > 0 and velocity1[1] > 0

    # Evaluate at position[2]
    position = np.array([-2.0, 0.4])
    velocity2 = avoider.evaluate_sequence(position)
    assert velocity2[0] < velocity1[0], "Not slowing down towards obstacle."


def _test_multiple_obstacles_in_wavy_dynamics(visualize=False):
    pass


if (__name__) == "__main__":
    # figtype = ".png"
    figtype = ".pdf"

    import matplotlib.pyplot as plt

    from dynamic_obstacle_avoidance.visualization import plot_obstacles
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )

    plt.ion()
    test_single_ellipse(visualize=True)
    # test_tree_with_two_children(savefig=False, visualize=False)

    # test_orthonormal_tangent_finding()
    # test_tripple_ellipse_in_the_face(visualize=True, savefig=False)
    # test_triple_ellipse_environment(visualize=False)
