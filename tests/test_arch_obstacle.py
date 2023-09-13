"""
Create 'Arch'-Obstacle which might be often used
"""
import math

from dataclasses import dataclass, field
from typing import Optional, Iterator
import numpy as np
import numpy.typing as npt

import networkx as nx

from vartools.states import Pose
from vartools.dynamics import Dynamics
from vartools.dynamics import WavyRotatedDynamics
from vartools.dynamical_systems import LinearSystem


from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles


from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import HierarchyObstacle
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    SingularityConvergenceDynamics,
)
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)
from nonlinear_avoidance.visualization.plot_multi_obstacle import plot_multi_obstacles
from nonlinear_avoidance.visualization.plot_qolo import integrate_with_qolo

from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import BlockArchObstacle
from nonlinear_avoidance.arch_obstacle import create_arch_obstacle
from nonlinear_avoidance.vector_rotation import VectorRotationSequence
from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container
from nonlinear_avoidance.dynamics.constant_value import ConstantValueWithSequence


def test_2d_blocky_arch(visualize=False):
    # multi_block = BlockArchObstacle(
    #     wall_width=0.4,
    #     axes_length=np.array([3.0, 5]),
    #     pose=Pose(np.array([0.0, 0]), orientation=0.0),
    # )
    dynamics = ConstantValueWithSequence([-1, 0.0])

    container = MultiObstacleContainer()
    multi_block = create_arch_obstacle(
        wall_width=0.4,
        axes_length=np.array([3.0, 5]),
        pose=Pose(np.array([0.0, 0]), orientation=0.0),
    )
    container.append(multi_block)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-3, 6.0]
        y_lim = [-5, 5.0]
        n_grid = 30

        plot_obstacles(
            obstacle_container=multi_block,
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
            # vectorfield_color=vf_color,
        )

    position = np.array([1, 1])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([1, -1])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.isclose(velocity1[0], velocity2[0])
    assert np.isclose(velocity1[1], -velocity2[1])

    position = np.array([1.0, -2.0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] < 0
    assert velocity[0] > 0

    position = np.array([4, 1.6])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0, "Velocity is not going left."
    assert velocity[1] > 0, "Avoiding upwards expected."


def test_2d_blocky_arch_rotated(visualize=False):
    dynamics = ConstantValueWithSequence(np.array([-1.0, 0]))

    container = MultiObstacleContainer()
    multi_block = create_arch_obstacle(
        wall_width=0.4,
        axes_length=np.array([3.0, 5]),
        pose=Pose(np.array([-1.0, -0.4]), orientation=-45 * np.pi / 180.0),
        # pose=Pose(np.array([0.0, 0]), orientation=0.0),
    )
    container.append(multi_block)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    # multibstacle_avoider = MultiObstacleAvoider(obstacle=multi_block)
    # linearized_velociy = np.array([-1.0, 0])

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-3, 6.0]
        y_lim = [-5, 5.0]
        n_grid = 30

        plot_obstacles(
            obstacle_container=multi_block,
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
            # vectorfield_color=vf_color,
        )

    # Test positions [which has been prone to a rounding error]
    position = np.array([-1.84, -2.38])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0, "Velocity is not going right."
    assert averaged_direction[1] < 0, "Avoiding downwards expected."

    # On the surface of a leg
    position = np.array([-1.35486, -3.01399])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] > 0, "Expected to go backwards"
    assert np.isclose(
        averaged_direction[0], -averaged_direction[1], atol=1e-1
    ), "Expected to go backwards"


def test_multi_arch_obstacle(visualize=False):
    # TO FIX (!)
    # distance_scaling = 0.5
    distance_scaling = 2.0

    # dynamics = LinearSystem(attractor_position=np.array([0, 0]))
    dynamics = LinearSystem(
        attractor_position=np.array([-4, 4.0]),
        # maximum_velocity=1.0
    )

    container = MultiObstacleContainer()
    container.append(
        create_arch_obstacle(
            wall_width=0.7,
            axes_length=np.array([3, 5.0]),
            pose=Pose(np.array([-1.0, -2.0]), orientation=90 * np.pi / 180.0),
        )
    )

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    container.append(
        create_arch_obstacle(
            wall_width=0.7,
            axes_length=np.array([3.0, 5]),
            pose=Pose(np.array([1.0, 2.0]), orientation=-90 * np.pi / 180.0),
            distance_scaling=distance_scaling,
        )
    )

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-5, 5.0]
        y_lim = [-5, 5.0]
        n_grid = 20

        plot_multi_obstacle_container(
            ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
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
            show_ticks=True,
            # vectorfield_color=vf_color,
        )

    position = np.array([-0.7, 0.0])
    averaged_direction = avoider.evaluate_sequence(position)
    assert averaged_direction[0] < 0, "Expected to continue to the left."
    assert averaged_direction[1] < 0, "Expected to rotate down."


def test_bi_arch_avoidance_nonlinear(visualize=False):
    attractor = np.array([4.0, -3])
    margin_absolut = 0.5

    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=attractor, orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )
    container = MultiObstacleContainer()
    container.append(
        create_arch_obstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.0]),
            pose=Pose(np.array([-1.5, -3.5]), orientation=90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )

    container.append(
        create_arch_obstacle(
            wall_width=0.4,
            axes_length=np.array([4.5, 6.0]),
            pose=Pose(np.array([1.5, 3.0]), orientation=-90 * np.pi / 180.0),
            margin_absolut=margin_absolut,
        )
    )
    # DELETING EVERYTHING (!!!)
    # container = MultiObstacleContainer()

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        default_dynamics=LinearSystem(initial_dynamics.attractor_position),
        create_convergence_dynamics=True,
    )

    if visualize:
        n_grid = 20
        x_lim = [-6.5, 6.5]
        y_lim = [-5.5, 5.5]

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            do_quiver=True,
            show_ticks=False,
        )

        plot_multi_obstacles(
            ax=ax,
            container=container,
            x_lim=x_lim,
            y_lim=y_lim,
        )

    plot_position = False
    if plot_position and visualize:
        # Check all normal directions
        position = np.array([0.8499904639878112, -0.03889134570119339])
        ax.plot(position[0], position[1], "ok")

        colors = ["red", "green", "blue"]
        d_pos = np.array([0.1, 0.1])
        for ii, obs in enumerate(container.get_obstacle_tree(0)._obstacle_list):
            normal = obs.get_normal_direction(position, in_global_frame=True)
            pos_text = obs.center_position + d_pos
            ax.text(pos_text[0], pos_text[1], f"obs={ii}", color=colors[ii])
            ax.arrow(position[0], position[1], normal[0], normal[1], color=colors[ii])

    do_trajectory = False
    if do_trajectory and visualize:
        start_position = np.array([-2.5, 3])
        _ = integrate_with_qolo(
            start_position=start_position,
            velocity_functor=avoider.evaluate_sequence,
            ax=ax,
            it_max=200,
        )

    # Positions close to the boundary
    position = np.array([0.56, -0.04])
    velocity = avoider.evaluate_sequence(position)
    assert np.isclose(velocity[0], 0, atol=1e-1)
    assert not np.any(np.isnan(velocity))


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    # test_2d_blocky_arch(visualize=True)
    # test_2d_blocky_arch_rotated(visualize=True)
    # test_multi_arch_obstacle(visualize=True)
    test_bi_arch_avoidance_nonlinear(visualize=True)
    # test_multi_arch_obstacle(visualize=False)

    print("Tests done.")
