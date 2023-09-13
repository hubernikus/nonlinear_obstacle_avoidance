import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import LinearSystem

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.dynamics import SimpleCircularDynamics
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import create_arch_obstacle

from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def integrate_trajectory(
    start_positions,
    velocity_functor,
    dt=0.01,
    it_max=200,
    abs_tol=1e-1,
    reduce_acceleration: bool = False,
):
    positions = np.zeros((start_positions.shape[0], it_max + 1))

    positions[:, 0] = start_positions
    tmp_velocity = velocity_functor(positions[:, 0])
    for ii in range(it_max):
        velocity = velocity_functor(positions[:, ii])

        if np.linalg.norm(velocity) < abs_tol:
            return positions[:, : ii + 1]

        # Reduce velocity when going to far apart
        dotprod = np.dot(velocity, tmp_velocity)
        if not np.isclose(dotprod, 0):
            dotprod = dotprod / (
                np.linalg.norm(velocity) * np.linalg.norm(tmp_velocity)
            )
        scaling = (1 + dotprod) / 2.0 + abs_tol
        scaling = min(1.0, scaling)

        # velocity = velocity * scaling
        tmp_velocity = velocity

        positions[:, ii + 1] = velocity * dt + positions[:, ii]
    return positions


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
    assert abs(velocity[0] / velocity[1]) < 0.5, "Parallel to surface."

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
    )

    if visualize:
        x_lim = [-1, 1]
        y_lim = [-1, 1]

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

    plot_initial = True
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([0.25, 0.01])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0 and velocity[1] > 0, "Keep circularity."

    position = np.array([0.0, 0])
    velocity = avoider.evaluate_sequence(position)
    assert np.allclose(velocity, np.zeros_like(velocity)), "Zero at center"


def test_limit_cycle_two_obstacle(visualize=False):
    # TODO: remove jittery behvaior (analyse at different positions...)
    #     is it just due to the high (local) nonlinearities? Or is it more fundamental...
    #     Further investigation needed (!)

    distance_scaling = 10
    margin_absolut = 0.2

    dynamics = SimpleCircularDynamics(
        pose=Pose(
            # np.array([-0.5, 0.2]),
            # np.array([1.0, 0.2]),
            np.array([0.5, 0.2]),
        ),
        radius=0.3,
    )

    container = MultiObstacleContainer()

    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    # Conveyer belt
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.0, -0.6]),
            axes_length=np.array([2.0, 1.0]),
            margin_absolut=0.1,
            distance_scaling=20,
        )
    )

    container.append(obstacle_tree)

    # Simple (convex) obstacle
    obstacle_tree = MultiObstacle(Pose(np.array([-1.0, 0.4])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0, 0.0]),
            axes_length=np.array([0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    obstacle_tree[-1].set_reference_point(np.array([0, -0.08]), in_global_frame=False)
    container.append(obstacle_tree)

    # More advanced, concave obstacle
    obstacle_tree = MultiObstacle(Pose(np.array([0.2, 0.5])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.0, -0.34]),
            axes_length=np.array([0.12, 0.24]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    obstacle_tree[-1].set_reference_point(np.array([0, -0.1]), in_global_frame=False)

    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([0.0, -0.1]),
            axes_length=np.array([0.37, 0.24]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
        parent_ind=0,
        reference_position=np.array([0.0, -0.12]),
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=0.55 * math.pi,
    )

    if visualize:
        x_lim = [-1.4, 1.4]
        y_lim = [-1.5, 1.5]
        # x_lim = [-0.9, -0.60]
        # y_lim = [-0.05, 0.2]

        n_resolution = 40
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim, draw_reference=True
        )

    plot_vectorfield = True
    if plot_vectorfield and visualize:
        plot_obstacle_dynamics(
            # obstacle_container=container,
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    show_trajectory = True
    if show_trajectory and visualize:
        start_positions = np.array(
            [
                [1.5, -0.4],
                [1.5, 0.4],
                [0, 1.5],
                [-1.5, -0.4],
                [1.5, 0.4],
            ]
        ).T
        for ii in range(start_positions.shape[1]):
            trajectory = integrate_trajectory(
                start_positions[:, ii],
                avoider.evaluate_sequence,
                it_max=400,
                abs_tol=1e-3,
            )

            color = "black"
            ax.plot(trajectory[0, :], trajectory[1, :], "-")
            ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color=color)
            ax.plot(trajectory[0, -1], trajectory[1, -1], "o", color=color)

            attractor_position = dynamics.attractor_position
            ax.plot(attractor_position[0], attractor_position[1], "k*", markersize=2.0)

    plot_initial = False
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    show_convergnce = False
    if show_convergnce and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.compute_convergence_direction,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    # TODO: check inside special
    # position = np.array([-0.04, 0.10])
    # velocity = avoider.evaluate_sequence(postion)

    position = np.array([-0.1885, 0.3150])
    convergence1 = avoider.compute_convergence_direction(position)
    velocity1 = avoider.evaluate_sequence(position)
    # print("velocity", velocity1)

    # print()
    position = np.array([-0.1859, 0.3150])
    convergence2 = avoider.compute_convergence_direction(position)
    velocity2 = avoider.evaluate_sequence(position)
    # print("velocity", velocity2)

    # position = np.array([-0.1859, 0.3222])
    # position = np.array([-0.1859, 0.3150])
    # convergence1 = avoider.compute_convergence_direction(position)
    # velocity3 = avoider.evaluate_sequence(position)

    # TEST DEACTIVATED (!)
    if True:
        return

    # Close to surface
    position = np.array([0.18, 0.7])
    velocity = avoider.evaluate_sequence(position)
    norm_vel = velocity / np.linalg.norm(velocity)
    normal = obstacle_tree.get_component(0).get_normal_direction(
        position, in_global_frame=True
    )
    assert 0.3 < normal @ norm_vel < 0.7

    position = np.array([-0.75809, 0.05526])
    convergence1 = avoider.compute_convergence_direction(position)
    assert convergence1[0] > 0

    # Position [list]
    position = np.array([0.44977403207952, 0.5055226795420217])
    occlusion_weights = avoider.compute_parent_occlusion_weight(
        position, container.get_tree(-1)
    )
    assert np.allclose(occlusion_weights, [1, 1])


def test_multiobstacle_normal_and_tangent(visualize=False):
    distance_scaling = 30
    margin_absolut = 0.1

    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([-0.5, 0.2]),
        ),
        radius=0.3,
    )

    container = MultiObstacleContainer()
    # More advanced, concave obstacle
    obstacle_tree = MultiObstacle(Pose(np.array([0.2, 0.2])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.0, 0.0]),
            axes_length=np.array([0.1, 0.2]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    obstacle_tree[-1].set_reference_point(np.array([0, -0.1]), in_global_frame=False)
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([0.0, 0.2]),
            axes_length=np.array([0.3, 0.2]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
        parent_ind=0,
        reference_position=np.array([0.0, -0.1]),
    )
    container.append(obstacle_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        convergence_radius=0.55 * math.pi,
    )

    if visualize:
        # x_lim = [-1.4, 1.4]
        # y_lim = [-1.5, 1.5]
        x_lim = [-0.2, 0.8]
        y_lim = [-0.1, 0.9]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim, draw_reference=True
        )

    show_trajectory = True
    if show_trajectory and visualize:
        start_position = np.array([1.5, -0.4])
        trajectory = integrate_trajectory(
            start_position, avoider.evaluate_sequence, it_max=300
        )

        color = "black"
        ax.plot(trajectory[0, :], trajectory[1, :], "-")
        ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color=color)
        ax.plot(trajectory[0, -1], trajectory[1, -1], "o", color=color)

    position = np.array([0.44977403207952, 0.5055226795420217])
    if visualize:
        plt.plot(position[0], position[1], "go")

    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] > 0, "Moving upwards."


def test_trajectory_integration(visualize=False):
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
            margin_absolut=0.2,
            distance_scaling=8.0,
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
        x_lim = [-1, 1]
        y_lim = [-1, 1]

        n_resolution = 3
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_resolution),
            np.linspace(y_lim[0], y_lim[1], n_resolution),
        )
        start_positions = np.array([xx.flatten(), yy.flatten()])

        for ii in range(start_positions.shape[1]):
            trajectory = integrate_trajectory(
                start_positions[:, ii], avoider.evaluate_sequence
            )

            color = "black"
            ax.plot(trajectory[0, :], trajectory[1, :], "-")
            ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color=color)
            ax.plot(trajectory[0, -1], trajectory[1, -1], "o", color=color)


def test_linear_avoidance_sphere(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0.0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Ellipse(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([1.0, 1.0]),
            margin_absolut=0.0,
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
        x_lim = [-4, 2]
        y_lim = [-3, 3]

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

    position = np.array([-2.8, 0.3])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0, "Avoiding to the right"
    assert velocity[1] > 0, "Avoiding to the top"


def _test_limit_cycle_obstacle_center(visualize=False):
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
            pose=Pose(
                # np.array([-0.01, 0.02]),
                np.array([-0.2, 0.1]),
                orientation=0.2 * 180.0 / math.pi,
            ),
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
    )

    if visualize:
        x_lim = [-0.3, 0.3]
        y_lim = [-0.3, 0.3]
        # x_lim = [-0.05, 0.15]
        # y_lim = [-0.025, 0.15]

        n_resolution = 10
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

    plot_trajectories = True
    if plot_trajectories and visualize:
        n_line_grid = 2
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_line_grid),
            np.linspace(y_lim[0], y_lim[1], n_line_grid),
        )
        start_positions = np.array([xx.flatten(), yy.flatten()])
        # start_positions = np.array([[0, 1.0]]).T
        for ii in range(start_positions.shape[1]):
            trajectory = integrate_trajectory(
                start_positions[:, ii], avoider.evaluate_sequence, it_max=50
            )

            color = "black"
            ax.plot(
                trajectory[0, :], trajectory[1, :], "-", linewidth=2.5, color="orange"
            )
            ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color=color)
            ax.plot(trajectory[0, -1], trajectory[1, -1], "o", color=color)

    plot_convergence = True
    if plot_convergence and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        def get_convergence(position):
            initial_sequence = evaluate_dynamics_sequence(
                position, avoider.initial_dynamics
            )

            convergence_sequence = avoider.compute_convergence_sequence(
                position, initial_sequence
            )

            return convergence_sequence.get_end_vector()

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=get_convergence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    plot_initial = False
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([0.1055555555555556, 0.0916666666666667])
    velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(velocity))


def test_limit_cycle_double_level(visualize=False):
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
    )

    if visualize:
        x_lim = [-1, 1]
        y_lim = [-1, 1]

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

    plot_initial = True
    if plot_initial and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([0.25, 0.01])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0 and velocity[1] > 0, "Keep circularity."

    position = np.array([0.0, 0])
    velocity = avoider.evaluate_sequence(position)
    assert np.allclose(velocity, np.zeros_like(velocity)), "Zero at center"


def test_simple_tree(visualize=False):
    distance_scaling = 30
    margin_absolut = 0.1
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([-0.5, 0.2]),
        ),
        radius=0.3,
    )

    container = MultiObstacleContainer()
    # More advanced, concave obstacle
    obstacle_tree = MultiObstacle(Pose(np.array([0.2, 0.2])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([0.0, 0.0]),
            axes_length=np.array([0.1, 0.2]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    obstacle_tree[-1].set_reference_point(np.array([0, -0.1]), in_global_frame=False)
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([0.0, 0.2]),
            axes_length=np.array([0.3, 0.2]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
        parent_ind=0,
        reference_position=np.array([0.0, -0.1]),
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
        # x_lim = [-0.9, -0.60]
        # y_lim = [-0.05, 0.2]

        n_resolution = 20
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
        )

    plot_vectorfield = True
    if plot_vectorfield and visualize:
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

    show_convergnce = True
    if show_convergnce and visualize:
        fig, ax = plt.subplots(figsize=figsize)

        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
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

    # Evaluate opposite the center
    position = np.array([-0.9, 0.2])
    convergence = avoider.compute_convergence_direction(position)
    assert not np.any(np.isnan((convergence)))
    assert convergence[1] < 0, "Keeping circle rotation."


def test_circular_no_obstacles(visualize=False):
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.0]),
        ),
        radius=2.0,
    )

    container = MultiObstacleContainer()

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        x_lim = [-3, 3.0]
        y_lim = [-3, 3.0]

        n_resolution = 40
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim, draw_reference=True
        )

    plot_vectorfield = True
    if plot_vectorfield and visualize:
        plot_obstacle_dynamics(
            # obstacle_container=container,
            obstacle_container=[],
            dynamics=avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    position = np.array([1.0, 0])
    velocity = avoider.evaluate(position)
    assert velocity[0] > 0
    assert velocity[1] > 0, "Pointing outwards."


def test_circular_spiraling(visualize=False):
    dimension = 2
    dynamics = LinearSystem(
        attractor_position=np.zeros(2),
        A_matrix=np.array([[-1.0, -2], [2, -1]]),
        maximum_velocity=1.0,
    )

    container = MultiObstacleContainer()
    new_tree = MultiObstacle(Pose.create_trivial(dimension))
    new_tree.set_root(
        Ellipse(axes_length=np.array([1.0, 1.0]), center_position=np.array([2.2, 0.0]))
    )
    container.append(new_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.55 * math.pi,
    )

    if visualize:
        x_lim = [-3.0, 3.4]
        y_lim = [-3.0, 3.0]
        # x_lim = [-0.9, -0.60]
        # y_lim = [-0.05, 0.2]

        n_resolution = 20
        figsize = (10, 8)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim, draw_reference=True
        )

    plot_vectorfield = True
    if plot_vectorfield and visualize:
        plot_obstacle_dynamics(
            # obstacle_container=container,
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )


def test_starshape_spiraling(visualize=False):
    dimension = 2
    dynamics = LinearSystem(
        attractor_position=np.zeros(2),
        A_matrix=np.array([[-1.0, -2], [2, -1]]),
        maximum_velocity=1.0,
    )

    main_obstacle = StarshapedFlower(
        center_position=np.array([2.2, 0.0]),
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        distance_scaling=1.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    container = MultiObstacleContainer()
    new_tree = MultiObstacle(Pose.create_trivial(dimension))
    new_tree.set_root(main_obstacle)
    container.append(new_tree)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.55 * math.pi,
    )

    if visualize:
        x_lim = [-3.0, 3.4]
        y_lim = [-3.0, 3.0]
        # x_lim = [-0.9, -0.60]
        # y_lim = [-0.05, 0.2]

        n_resolution = 20
        # figsize = (10, 8)
        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim, draw_reference=True
        )

    plot_vectorfield = True
    if plot_vectorfield and visualize:
        plot_obstacle_dynamics(
            # obstacle_container=container,
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    show_convergence = True
    if show_convergence and visualize:
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

    position = np.array([2.38, 1.13])
    # convergence = avoider.compute_convergence_direction(position)
    dynamics = avoider.evaluate_sequence(position)
    assert dynamics[1] > 0, "Keep circle dynamics."


if (__name__) == "__main__":
    figtype = ".pdf"
    # np.set_printoptions(precision=16)
    np.set_printoptions(precision=3)

    # test_straight_system_with_two_trees(visualize=True)

    # test_straight_system_with_tree(visualize=False)

    # test_trajectory_integration(visualize=True)

    # test_limit_cycle_single_level(visualize=True)

    # _test_limit_cycle_obstacle_center(visualize=False)
    # _test_limit_cycle_obstacle_center(visualize=True)

    # test_straight_system_with_arch(visualize=False)

    # test_limit_cycle_two_obstacle(visualize=False)

    # test_linear_avoidance_sphere(visualize=False)

    # test_simple_tree(visualize=True)

    # test_straight_system_single_level_tree(visualize=False)
    # test_straight_system_with_edgy_tree(visualize=True)

    # test_multiobstacle_normal_and_tangent(visualize=True)

    test_limit_cycle_two_obstacle(visualize=True)

    # test_circular_no_obstacles(visualize=False)

    # test_starshape_spiraling(visualize=False)
    # test_circular_spiraling(visualize=True)
