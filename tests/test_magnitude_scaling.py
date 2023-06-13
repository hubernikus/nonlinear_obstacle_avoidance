import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from vartools.states import Pose
from vartools.colors import hex_to_rgba_float
from vartools.dynamics import QuadraticAxisConvergence, LinearSystem
from vartools.dynamics import WavyRotatedDynamics

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from nonlinear_avoidance.arch_obstacle import BlockArchObstacle
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.visualization.plot_multi_obstacle import plot_multi_obstacles
from nonlinear_avoidance.visualization.plot_qolo import integrate_with_qolo
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def test_single_rectangle_multiavoidance(visualize=False, save_figure=False, n_grid=40):
    margin_absolut = 0.5
    wall_width = 0.4
    axes_length = np.array([4.5, 6.5])
    attractor = np.array([4.0, -3])
    dimension = 2

    container = MultiObstacleContainer()

    corner_obstacle = MultiObstacle(
        Pose(np.array([-1.5, -3.5]), orientation=90 * np.pi / 180.0)
    )
    corner_obstacle.set_root(
        Cuboid(
            axes_length=np.array([wall_width, axes_length[1]]),
            pose=Pose(np.zeros(dimension), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
    )

    # delta_pos = (axes_length - wall_width) * 0.5
    # corner_obstacle.add_component(
    #     Cuboid(
    #         axes_length=np.array([axes_length[0], wall_width]),
    #         pose=Pose(delta_pos, orientation=0.0),
    #         margin_absolut=margin_absolut,
    #     ),
    #     reference_position=np.array([-delta_pos[0], 0.0]),
    #     parent_ind=0,
    # )

    container.append(corner_obstacle)

    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=attractor, orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )

    convergence_dynamics = LinearSystem(attractor, maximum_velocity=1.0)
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=convergence_dynamics.attractor_position,
        initial_dynamics=convergence_dynamics,
        reference_velocity=lambda x: x - attractor,
    )

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        convergence_dynamics=rotation_projector,
        # create_convergence_dynamics=True,
    )

    if visualize:
        figsize = (4, 3.5)

        x_lim = [-6.5, 6.5]
        y_lim = [-5.5, 5.5]

        fig, ax = plt.subplots(figsize=figsize)
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        normals = np.zeros_like(positions)
        for pp in range(positions.shape[1]):
            if container.get_gamma(positions[:, pp]) < 1:
                continue

            normals[:, pp], _ = avoider.compute_averaged_normal_and_gamma(
                positions[:, pp]
            )

        plot_multi_obstacles(ax=ax, container=container)

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="red",
            # scale=quiver_scale,
            # alpha=quiver_alpha,
            # width=0.007,
            zorder=-1,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        fig, ax = plt.subplots(figsize=figsize)
        collision_checker = lambda pos: (not container.is_collision_free(pos))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=initial_dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            collision_check_functor=collision_checker,
            do_quiver=True,
            show_ticks=False,
        )

        fig, ax = plt.subplots(figsize=figsize)
        collision_checker = lambda pos: (not container.is_collision_free(pos))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            collision_check_functor=collision_checker,
            do_quiver=True,
            show_ticks=False,
        )
        plot_multi_obstacles(ax=ax, container=container)

    position = np.array([0.37, -2.63])
    rotated_velocity1 = avoider.evaluate_sequence(position)
    assert rotated_velocity1[0] > 0, "Avoidance towards the right"

    position = np.array([1.0, -2.5])
    rotated_velocity1 = avoider.evaluate_sequence(position)
    assert rotated_velocity1[0] > 0, "Avoidance towards the right"
    assert abs(rotated_velocity1[1]) < 0.2, "Only little velocity towards obstacle."


def test_normals_multi_arch(visualize=False, save_figure=False, n_grid=40):
    margin_absolut = 0.5
    attractor = np.array([4.0, -3])

    container = MultiObstacleContainer()
    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.5]),
            pose=Pose(np.array([-1.5, -3.5]), orientation=90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )

    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.0]),
            pose=Pose(np.array([1.5, 3.0]), orientation=-90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )

    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=attractor, orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )

    convergence_dynamics = LinearSystem(attractor, maximum_velocity=1.0)
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=convergence_dynamics.attractor_position,
        initial_dynamics=convergence_dynamics,
        reference_velocity=lambda x: x - attractor,
    )

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        convergence_dynamics=rotation_projector,
        # create_convergence_dynamics=True,
    )

    if visualize:
        figsize = (4, 3.5)

        x_lim = [-6.5, 6.5]
        y_lim = [-5.5, 5.5]

        fig, ax = plt.subplots(figsize=figsize)
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        normals = np.zeros_like(positions)
        for pp in range(positions.shape[1]):
            if container.get_gamma(positions[:, pp]) < 1:
                continue

            normals[:, pp], _ = avoider.compute_averaged_normal_and_gamma(
                positions[:, pp]
            )

        plot_multi_obstacles(ax=ax, container=container)

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="red",
            # scale=quiver_scale,
            # alpha=quiver_alpha,
            # width=0.007,
            zorder=-1,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        fig, ax = plt.subplots(figsize=figsize)
        collision_checker = lambda pos: (not container.is_collision_free(pos))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=initial_dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            collision_check_functor=collision_checker,
            do_quiver=True,
            show_ticks=False,
        )

        fig, ax = plt.subplots(figsize=figsize)
        collision_checker = lambda pos: (not container.is_collision_free(pos))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            collision_check_functor=collision_checker,
            do_quiver=True,
            show_ticks=False,
        )
        plot_multi_obstacles(ax=ax, container=container)

        start_position = np.array([-2.5, 3])
        trajectory = integrate_with_qolo(
            start_position=start_position,
            velocity_functor=avoider.evaluate_sequence,
            ax=ax,
            it_max=500,
            show_qolo=False,
        )

    # Velocity should never get stationary
    position = np.array([-2.0000256770879297, -0.9239067600516644])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate_sequence(position)
    assert not np.allclose(rotated_velocity, [0, 0], atol=1e-2), "Keep moving!"

    # Velocity should never get stationary
    position = np.array([-2.0000924283736654, -0.24311766270808285])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate_sequence(position)
    assert not np.allclose(rotated_velocity, [0, 0], atol=1e-4), "Keep moving!"

    position = np.array([1.58, 3.98])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate_sequence(position)
    assert rotated_velocity[0] > 0
    assert rotated_velocity[1] < 0
    assert abs(rotated_velocity[0]) > abs(rotated_velocity[1])
    # assert np.linalg.norm(rotated_velocity) > 0.5, "Should still be fast!"

    position = np.array([-5.55, -2.25])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(rotated_velocity))
    assert np.linalg.norm(rotated_velocity) < 1, "Should be slowed down(!)"

    # Test slow-down
    position = np.array([-4.5, -4.9])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    modulated_velocity = avoider.evaluate_sequence(position)
    speed_magnitude = np.linalg.norm(modulated_velocity)
    assert np.isclose(speed_magnitude, 1.0), "No slow down when pointing down."

    # Test normal
    position = np.array([-3.7, 0.481])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    assert abs(normal[0]) > abs(normal[1]), "Closest wall higher influence."
    assert normal[0] > 0, "Point away from the wall"
    assert normal[1] > 0, "Pointing up"


if (__name__) == "__main__":
    test_single_rectangle_multiavoidance(visualize=True)
    # test_normals_multi_arch(visualize=True)
