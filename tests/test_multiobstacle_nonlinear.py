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

from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def test_straight_system_with_edgy_tree(visualize=False):
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
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([-2.0, 1.0]),
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

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([-4.1, 1])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[0] / velocity[1]) < 1e-1, "Parallel to surface."

    position = np.array([-2.5, -1.55])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[1] / velocity[0]) < 1e-1, "Parallel to surface."


def test_straight_system_with_arch(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = create_arch_obstacle(
        wall_width=0.5,
        axes_length=np.array([2, 4]),
        pose=Pose(position=np.array([2, 0])),
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-2, 7]
        y_lim = [-4, 4]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([3, 1])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] > 0 and velocity[0] > 0, "Avoid the obstacle"

    position = np.array([1, 1])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0 and velocity[1] < 0, "Relax behind"

    position = np.array([1, 1])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0 and velocity[1] < 0, "Relax behind"


def test_parent_occlusion_weights(visualize=False):
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
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([-2.0, 1.0]),
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

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

    # Test parent-occlusion weights
    position = np.array([0, -1])
    weights = avoider.compute_parent_occlusion_weight(position, obstacle_tree)
    assert 0 < weights[1] < 1, "Second obstacle not zero weight."

    position = np.array([1, 1])
    weights = avoider.compute_parent_occlusion_weight(position, obstacle_tree)
    assert np.allclose(weights, [1, 1]), "Second obstacle not zero weight."

    position = np.array([-2, 2])
    weights = avoider.compute_parent_occlusion_weight(position, obstacle_tree)
    assert np.allclose(weights, [1, 1]), "Second obstacle not zero weight."

    position = np.array([-2, -3])
    weights = avoider.compute_parent_occlusion_weight(position, obstacle_tree)
    assert np.allclose(weights, [1, 0]), "Second obstacle not zero weight."


def test_straight_system_single_level_tree(visualize=False):
    # TODO: reduce effect at intersection
    dynamics = LinearSystem(attractor_position=np.array([2, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 2.0]),
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
        x_lim = [-5, 5]
        y_lim = [-5, 5]

        n_resolution = 30
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([3.0, 0])
    velocity = avoider.evaluate_sequence(position)
    assert np.isclose(velocity[1], 0), "Going towards center"
    assert velocity[0] < 0, "Going towards center"

    position = np.array([0.0, 3.0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] < 0, "Going towards center"
    assert velocity[0] > 0, "Going towards center"

    position = np.array([-5.0, 0.0])
    velocity = avoider.evaluate_sequence(position)
    assert np.isclose(velocity[1], 0), "Going towards center"
    assert velocity[0] > 0, "Going towards center"

    position = np.array([-5.0, 1.0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0, "Going towards center"
    assert velocity[1] > 0, "Avoiding towards the top"


def test_straight_system_with_tree(visualize=False):
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
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([-2.0, 1.0]),
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

        n_resolution = 40
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([-0.28, 1.52])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[1] / velocity[0]) < 1e-1, "Parallel to border"

    position = np.array([0.4, 3.10])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([0.4, 3.13])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    position = np.array([-3.05, -0.67])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[1] / velocity[0]) > 1e1, "Expected to be parallel to wall."
    assert velocity[1] < 0.0

    position = np.array([-2.3, -1.55])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[0] / velocity[1]) > 1e2, "Expected to be parallel to wall."
    assert velocity[0] > 0.0

    position = np.array([-4.8, -1.8])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([-4.8, -1.795])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    # position = np.array([-2.01, -4.785])
    position = np.array([-4.3469387755, 3.0204081632])
    velocity1 = avoider.evaluate(position)
    position = np.array([-4.33, 3.0])
    velocity2 = avoider.evaluate(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    position = np.array([-4.76, -2.01])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([-4.75, -2.01])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)


def test_straight_system_with_two_trees(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree1 = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree1.set_root(
        Cuboid(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )
    obstacle_tree1.add_component(
        Cuboid(
            center_position=np.array([-2.0, 1.0]),
            axes_length=np.array([4.0, 1.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        ),
        reference_position=np.zeros(2),
        parent_ind=0,
    )
    container.append(obstacle_tree1)

    obstacle_tree2 = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree2.set_root(
        Cuboid(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,
        )
    )

    container.append(obstacle_tree1)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-5, 3]
        y_lim = [-4, 4]

        n_resolution = 40
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            container=container, ax=ax, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )


def test_limit_cycle_single_level(visualize=False):
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.0]),
        ),
        radius=2.0,
    )

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-5.0, 0]),
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
    )

    if visualize:
        x_lim = [-8, 2.5]
        y_lim = [-5, 5]

        n_resolution = 30
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    plot_initial = False
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([1.0, 0.5])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0 and velocity[1] > 0, "keep circularity."

    position = np.array([0.0, 0])
    velocity = avoider.evaluate_sequence(position)
    assert np.allclose(velocity, np.zeros_like(velocity))


def test_limit_cycle_two_obstacle(visualize=False):
    distance_scaling = 10
    margin_absolut = 0.2
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([-0.5, 0.2]),
        ),
        radius=0.3,
    )

    container = MultiObstacleContainer()

    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.0, -0.7]),
            axes_length=np.array([2.5, 1.0]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(obstacle_tree)

    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-1.0, 0.4]),
            axes_length=np.array([0.2, 0.3]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(obstacle_tree)

    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.8, 1.0]),
            axes_length=np.array([0.3, 0.3]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-1.4, 1.4]
        y_lim = [-1.5, 1.5]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    plot_initial = False
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )


if (__name__) == "__main__":
    figtype = ".pdf"

    # test_straight_system_with_edgy_tree(visualize=True)

    # test_straight_system_with_arch(visualize=True)

    # test_straight_system_with_two_trees(visualize=True)

    # test_straight_system_single_level_tree(visualize=False)

    # test_straight_system_with_tree(visualize=False)

    # test_limit_cycle_single_level(visualize=True)
    test_limit_cycle_two_obstacle(visualize=True)
