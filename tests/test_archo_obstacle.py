"""
Create 'Arch'-Obstacle which might be often used
"""
from dataclasses import dataclass, field
from typing import Optional, Iterator
import numpy as np
import numpy.typing as npt

import networkx as nx

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import HierarchyObstacle
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    SingularityConvergenceDynamics,
)
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import BlockArchObstacle


def test_2d_blocky_arch(visualize=False):
    multi_block = BlockArchObstacle(
        wall_width=0.4,
        axes_length=np.array([3.0, 5]),
        pose=Pose(np.array([0.0, 0]), orientation=0.0),
    )

    multibstacle_avoider = MultiObstacleAvoider(obstacle=multi_block)

    velocity = np.array([-1.0, 0])
    linearized_velociy = np.array([-1.0, 0])

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-5, 5.0]
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
            show_ticks=True,
            # vectorfield_color=vf_color,
        )

    # Test positions [which has been prone to a rounding error]
    position = np.array([4, 1.6])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] < 0, "Velocity is not going left."
    assert averaged_direction[1] > 0, "Avoiding upwards expected."


def test_2d_blocky_arch_rotated(visualize=False):
    multi_block = BlockArchObstacle(
        wall_width=0.4,
        axes_length=np.array([3.0, 5]),
        pose=Pose(np.array([-1.0, -0.4]), orientation=-45 * np.pi / 180.0),
    )

    multibstacle_avoider = MultiObstacleAvoider(obstacle=multi_block)

    velocity = np.array([-1.0, 0])
    linearized_velociy = np.array([-1.0, 0])

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-5, 5.0]
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
            show_ticks=True,
            # vectorfield_color=vf_color,
        )

    print("Doing")
    # Test positions [which has been prone to a rounding error]
    position = np.array([0.25, -3.99])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    # Evaluate gamma
    assert averaged_direction[0] < 0, "Velocity is not going left."
    assert averaged_direction[1] < 0, "Avoiding downwards expected."

    # On the surface of a leg
    position = np.array([-1.35486, -3.01399])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0, "Expected to go backwards"
    assert np.isclose(
        averaged_direction[0], -averaged_direction[1], atol=1e-1
    ), "Expected to go backwards"


def test_multi_arch_obstacle(visualize=False):
    container = MultiObstacleContainer()
    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([3.0, 5]),
            pose=Pose(np.array([-1.0, -2.5]), orientation=90 * np.pi / 180.0),
        )
    )

    container.append(
        BlockArchObstacle(
            wall_width=0.4,
            axes_length=np.array([3.0, 5]),
            pose=Pose(np.array([1.0, 2.0]), orientation=-90 * np.pi / 180.0),
        )
    )

    attractor = np.array([-4, 4.0])
    initial_dynamics = LinearSystem(attractor_position=attractor, maximum_velocity=1.0)

    # rotation_projector = ProjectedRotationDynamics(
    #     attractor_position=initial_dynamics.attractor_position,
    #     initial_dynamics=initial_dynamics,
    #     reference_velocity=lambda x: x - attractor,
    # )
    multibstacle_avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        # convergence_dynamics=rotation_projector,
        create_convergence_dynamics=True,
    )

    velocity = np.array([-1.0, 0])
    linearized_velociy = np.array([-1.0, 0])

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-5, 5.0]
        y_lim = [-5, 5.0]
        n_grid = 20

        for multi_obs in container:
            plot_obstacles(
                obstacle_container=multi_obs._obstacle_list,
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
                container.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            # dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
            #     x, velocity, linearized_velociy
            # ),
            dynamics=multibstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
        )

    position = np.array([-0.19, -0.35])
    averaged_direction = multibstacle_avoider.get_tangent_direction(position, velocity)
    assert averaged_direction[0] < 0, "Expected to continue to the left."
    assert averaged_direction[1] < 0, "Expected to rotate down."

    position = np.array([-2.4, -0.19])
    averaged_direction = multibstacle_avoider.get_tangent_direction(position, velocity)
    assert averaged_direction[0] < 0, "Expected to continue to the left."
    assert averaged_direction[1] > 0, "Expected to rotate down."


if (__name__) == "__main__":
    # test_2d_blocky_arch(visualize=False)
    # test_2d_blocky_arch_rotated(visualize=True)
    test_multi_arch_obstacle(visualize=False)

    print("Tests done.")
