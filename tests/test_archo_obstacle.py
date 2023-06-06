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
from nonlinear_avoidance.vector_rotation import VectorRotationSequence


class ConstantValueWithSequence(Dynamics):
    """Returns constant velocity based on the DynamicalSystem parent-class"""

    def __init__(self, velocity):
        self.constant_velocity = velocity

    def evaluate(self, *args, **kwargs):
        """Random input arguments, but always ouptuts same vector-field"""
        return self.constant_velocity

    def evaluate_dynamics_sequence(self, position: np.ndarray):
        velocity = self.evaluate(position)

        rotation = VectorRotationSequence.create_from_vector_array(
            np.vstack((velocity, velocity)).T
        )
        return rotation


def test_2d_blocky_arch(visualize=False):
    multi_block = BlockArchObstacle(
        wall_width=0.4,
        axes_length=np.array([3.0, 5]),
        pose=Pose(np.array([0.0, 0]), orientation=0.0),
    )
    container = MultiObstacleContainer()
    container.append(multi_block)

    velocity = np.array([-1.0, 0])
    dynamics = ConstantValueWithSequence(velocity)
    multibstacle_avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    linearized_velociy = velocity

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-3, 6.0]
        y_lim = [-5, 5.0]
        n_grid = 30

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
            dynamics=multibstacle_avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
        )

    position = np.array([1.0, -2.0])
    velocity = multibstacle_avoider.evaluate_sequence(position)
    assert abs(velocity[1] / velocity[0]) < 1e-1, "Position almost parallel to wall"

    position = np.array([4, 1.6])
    velocity = multibstacle_avoider.evaluate_sequence(position)
    assert velocity[0] < 0, "Velocity is not going left."
    assert velocity[1] > 0, "Avoiding upwards expected."


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

    position = np.array([2.1, -0.19])
    averaged_direction = multibstacle_avoider.get_tangent_direction(position, velocity)
    assert averaged_direction[0] < 0, "Expected to continue to the left."
    assert averaged_direction[1] > 0, "Expected to rotate down."


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

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        # convergence_dynamics=rotation_projector,
        create_convergence_dynamics=True,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(5, 4))
        # plot_obstacle_dynamics(
        #     obstacle_container=container,
        #     dynamics=avoider.evaluate,
        #     x_lim=x_lim,
        #     y_lim=y_lim,
        #     n_grid=n_grid,
        #     ax=ax,
        #     attractor_position=initial_dynamics.pose.position,
        #     collision_check_functor=collision_checker,
        #     do_quiver=False,
        #     show_ticks=False,
        # )
        plot_multi_obstacles(
            ax=ax,
            container=container,
            x_lim=[-6.5, 6.5],
            y_lim=[-5.5, 5.5],
        )

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

        start_position = np.array([-2.5, 3])
        _ = integrate_with_qolo(
            start_position=start_position,
            velocity_functor=avoider.evaluate,
            ax=ax,
            it_max=200,
        )

    #

    # Integration of Qolo
    position = np.array([0.8499904639878112, -0.03889134570119339])
    velocity = avoider.evaluate(position)
    assert np.isclose(velocity[0], 0, atol=1e-3)
    assert velocity[1] > 0

    # Positions close to the boundary
    position = np.array([0.85, -0.04])
    velocity = avoider.evaluate(position)
    assert np.isclose(velocity[0], 0)
    # assert velocity[1] > 0


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    # test_2d_blocky_arch(visualize=True)
    test_2d_blocky_arch(visualize=True)

    # test_2d_blocky_arch_rotated(visualize=True)
    # test_multi_arch_obstacle(visualize=True)
    # test_bi_arch_avoidance_nonlinear(visualize=False)

    print("Tests done.")
