import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import LinearSystem

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.dynamics import SimpleCircularDynamics
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import create_arch_obstacle
from nonlinear_avoidance.dynamics.sequenced_dynamics import evaluate_dynamics_sequence

from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def test_simple_cube(visualize=False):
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.0]),
        ),
        radius=0.1,
    )

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-0.4, 0]),
            axes_length=np.array([0.16, 0.16]),
            margin_absolut=0.1,
            distance_scaling=50.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=math.pi * 0.5,
    )

    if visualize:
        x_lim = [-0.7, 0.1]
        y_lim = [-0.4, 0.4]

        n_resolution = 9
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        # plot_multi_obstacle_container(
        #     ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        # )

        boundary = obstacle_tree[0].get_boundary_with_margin_xy()
        ax.plot(boundary[0, :], boundary[1, :], "--", color="black")

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([-0.5, -0.1])
    velocity = avoider.evaluate_sequence(position)
    normal = obstacle_tree[-1].get_normal_direction(position, in_global_frame=True)
    assert velocity @ normal > 0, "Repulsive inside."


def test_simple_repulsive_circle(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-1.5, 0]),
            # orientation=45 * math.pi / 180,
            axes_length=np.array([2.0, 2.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=math.pi * 1.0,
    )

    if visualize:
        x_lim = [-3, 2]
        y_lim = [-2.5, 2.5]

        n_resolution = 21
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        # plot_multi_obstacle_container(
        #     ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        # )

        boundary = obstacle_tree[0].get_boundary_with_margin_xy()
        # ax.plot(boundary[0, :], boundary[1, :], "--", color="black")
        ax.plot(boundary[0], boundary[1], "--", color="black")

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    # Exactly oposoite
    position = np.array([-2.5, 0])
    velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(velocity)), "Move anywhere behind."

    # Exactly opposoite
    position = obstacle_tree[-1].global_reference_point
    velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(velocity)), "Move any direction at center."

    # Exactly opposoite
    position = np.array([1.0, 0.0])
    velocity = avoider.evaluate_sequence(position)
    assert np.allclose(velocity, [-1, 0]), "Not affected at opposite."

    # Absolut repulsion inside
    position = np.array([-1.0, 0.4])
    velocity = avoider.evaluate_sequence(position)
    normal = obstacle_tree[-1].get_normal_direction(position, in_global_frame=True)
    assert np.dot(velocity, normal) / np.linalg.norm(velocity), "Parallel to tangent"

    # Repulsion outside
    position = np.array([-1.5, 1.1])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] > 0
    assert abs(velocity[0] / velocity[1]) < 0.2


def test_ellipse_repulsion(visualize=False):
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.0]),
        ),
        radius=0.1,
    )

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-2.5, 0]),
            orientation=20 * math.pi / 180,
            axes_length=np.array([3.0, 1.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=math.pi,
    )

    if visualize:
        x_lim = [-4, 2]
        y_lim = [-3, 3]

        n_resolution = 19
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)

        boundary = obstacle_tree[0].get_boundary_with_margin_xy()
        ax.plot(boundary[0], boundary[1], "--", color="black")

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    # Absolut repulsion inside
    position = np.array([0.6, 2.7])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0
    assert velocity[1] < 0


def test_penetration_repulsion(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-1.5, 0]),
            # orientation=45 * math.pi / 180,
            axes_length=np.array([2.0, 2.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=math.pi * 0.5,
        gamma_maximum_repulsion=0.8,
    )

    if visualize:
        x_lim = [-3, 2]
        y_lim = [-2.5, 2.5]

        n_resolution = 21
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        # plot_multi_obstacle_container(
        #     ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        # )

        boundary = obstacle_tree[0].get_boundary_with_margin_xy()
        # ax.plot(boundary[0, :], boundary[1, :], "--", color="black")
        ax.plot(boundary[0], boundary[1], "--", color="black")

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([-1.5, 1.0])
    velocity = avoider.evaluate_sequence(position)
    velocity = velocity / np.linalg.norm(velocity)
    assert np.allclose(velocity, [1, 0], atol=1.0), "Tangent on surface"

    position = np.array([-1.5, 0.5])
    velocity = avoider.evaluate_sequence(position)
    velocity = velocity / np.linalg.norm(velocity)
    assert np.allclose(velocity, [0, 1]), "Repulsive inside"


if (__name__) == "__main__":
    figtype = ".pdf"
    # np.set_printoptions(precision=16)

    test_simple_cube(visualize=True)
    # test_simple_repulsive_circle(visualize=True)
    # test_ellipse_repulsion(visualize=False)

    # test_penetration_repulsion(visualize=False)
