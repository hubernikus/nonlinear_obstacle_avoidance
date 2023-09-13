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
from nonlinear_avoidance.multi_obstacle_avoider import get_limited_weights_to_max_sum
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import create_arch_obstacle

from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def test_weight_normalization():
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    weights = [0.8, 0.9]
    weights, tot_weight = get_limited_weights_to_max_sum(weights)
    assert all(0 < weights) and all(weights < 1)
    assert math.isclose(tot_weight, 1)

    weights = [1, 0.1]
    weights, tot_weight = get_limited_weights_to_max_sum(weights)
    assert np.allclose(weights, [1, 0])
    assert math.isclose(tot_weight, 1)

    weights = [0.1, 0.2]
    weights, tot_weight = get_limited_weights_to_max_sum(weights)
    assert isinstance(tot_weight, float) and tot_weight == sum(weights)


def test_convergence_sequence_single(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-5, 3]
        y_lim = [-4, 4]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([0, -4])
    direction = avoider.compute_convergence_direction(position)
    center_velocity = dynamics.evaluate(obstacle_tree[0].center_position)
    center_velocity = center_velocity / np.linalg.norm(center_velocity)
    assert not np.allclose(
        direction, center_velocity, atol=1e-1
    ), "Should differ at other positions"

    position = np.array([1, 0])
    direction = avoider.compute_convergence_direction(position)
    position_velocity = dynamics.evaluate(position)
    position_velocity = position_velocity / np.linalg.norm(position_velocity)
    assert np.allclose(direction, position_velocity)

    position = np.array([-3, 0])
    direction = avoider.compute_convergence_direction(position)
    center_velocity = dynamics.evaluate(obstacle_tree[0].center_position)
    center_velocity = center_velocity / np.linalg.norm(center_velocity)
    assert np.allclose(direction, center_velocity)


def test_convergence_sequence_single_close(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-1.01, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-5, 3]
        y_lim = [-4, 4]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([-2.05, -1.524])
    direction = avoider.compute_convergence_direction(position)
    initial_velocity = dynamics.evaluate(position)
    initial_velocity = initial_velocity / np.linalg.norm(initial_velocity)
    assert not np.allclose(
        direction, initial_velocity, atol=1e-3
    ), "Singularity almost inside obstaccle --- stay with initial"


def test_convergence_sequence_double(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-3.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([-3.0, 1.0]),
            axes_length=np.array([4.0, 1.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        ),
        reference_position=np.zeros(2),
        parent_ind=0,
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-5, 3]
        y_lim = [-4, 4]

        # x_lim = [-1, 1]
        # y_lim = [-1, 1]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([2.0, 0.1])
    direction = avoider.compute_convergence_direction(position)
    assert direction[0] < 0, "Going towards target."

    position = np.array([-0.0, -2])
    direction = avoider.compute_convergence_direction(position)
    assert direction[0] > 0, "Going towards target."


def test_convergence_sequence_arch_obstacle(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    # container.append(
    #     create_arch_obstacle(
    #         wall_width=1.0,
    #         axes_length=np.array([3, 5]),
    #         pose=Pose(position=np.array([-3, 2])),
    #     )
    # )
    container.append(
        create_arch_obstacle(
            wall_width=1.0,
            axes_length=np.array([3, 5]),
            pose=Pose(position=np.array([4.0, -1]), orientation=math.pi),
        )
    )

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-3, 8]
        y_lim = [-5, 5]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([0.23, -0.28])
    direction = avoider.compute_convergence_direction(position)
    # TODO: how should this exactly behave opposite the attractor (?)
    assert direction[0] < 0 and direction[1] > 0

    position = np.array([1.37, 1.22])
    direction1 = avoider.compute_convergence_direction(position)
    position = np.array([4.6, -3.1])
    direction2 = avoider.compute_convergence_direction(position)
    assert np.allclose(direction1, direction2, atol=1e-1)


def test_convergence_sequence_double_tree(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    container.append(
        create_arch_obstacle(
            wall_width=1.0,
            axes_length=np.array([3, 5]),
            pose=Pose(position=np.array([-3, 2])),
        )
    )

    container.append(
        create_arch_obstacle(
            wall_width=1.0,
            axes_length=np.array([3, 5]),
            pose=Pose(position=np.array([3.0, -1]), orientation=math.pi),
        )
    )

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-5, 6]
        y_lim = [-5, 5]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([4.0, 0])
    direction = avoider.compute_convergence_direction(position)
    assert not np.any(np.isnan(direction))
    assert direction[0] < 0 and direction[1] > 0, "Convergence towards center"


if (__name__) == "__main__":
    figtype = ".pdf"
    # test_convergence_sequence_arch_obstacle(visualize=True)

    # test_convergence_sequence_single(visualize=True)
    # test_convergence_sequence_single_close(visualize=True)
    # test_convergence_sequence_double(visualize=True)

    # test_weight_normalization()

    # test_convergence_sequence_single(visualize=True)

    # test_convergence_sequence_double_tree(visualize=True)
    test_convergence_sequence_arch_obstacle(visualize=False)
