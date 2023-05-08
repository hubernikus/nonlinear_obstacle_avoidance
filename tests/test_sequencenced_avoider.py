"""
Move Around Corners with Smooth Dynamics
"""
import math
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import LinearSystem

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.dynamics.segmented_dynamics import create_segment_from_points
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    SingularityConvergenceDynamics,
)


def test_sequenced_linear_single_circle(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([4, 0]), maximum_velocity=1.0)

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            pose=Pose.create_trivial(2),
            axes_length=np.array([4.0, 4.0]),
        )
    )
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_lim = [-5, 5]
        y_lim = [-5, 5]
        n_grid = 11

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            do_quiver=True,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim, ax=ax
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

    direction = np.array([-1.0, 0.1])
    velocity1 = avoider.evaluate_sequence(direction * 2)
    velocity2 = avoider.evaluate_sequence(direction * 4)
    assert not np.allclose(velocity1, velocity2), "No influence scaling."
    assert velocity2[0] > velocity1[0], "More effect closer to the obstacle."

    # Evaluate on surface
    direction = direction / np.linalg.norm(direction)
    velocity = avoider.evaluate_sequence(
        direction * obstacle_environment[-1].axes_length[0] * 0.5
    )
    assert np.isclose(np.dot(direction, velocity), 0.0), "Not tangent on surface."

    position = np.array([1.0, 3.0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0
    assert velocity[1] < 0

    position = np.array([-2, -2])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0, "Needs to move to the right."
    assert velocity[1] < 0, "Needs to avoid the obstacle."


def test_sequenced_linear_cuboid(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([4, 0]), maximum_velocity=1.0)

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Cuboid(
            pose=Pose(np.array([0.2, 0.4]), orientation=40 * math.pi / 180.0),
            axes_length=np.array([3.0, 5.0]),
        )
    )
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_lim = [-5, 5]
        y_lim = [-5, 5]
        n_grid = 11

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            # attractor_position=dynamic.attractor_position,
            do_quiver=True,
            # show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim, ax=ax
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

    velocity1 = avoider.evaluate_sequence(np.array([-3, 1]))
    velocity2 = avoider.evaluate_sequence(np.array([-3, -1]))
    assert velocity2[0] > velocity1[0], "Closser to initial below."


def test_multiple_obstacles(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([4, 0]), maximum_velocity=1.0)

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Cuboid(
            pose=Pose(np.array([1, 3]), orientation=-20 * math.pi / 180.0),
            axes_length=np.array([3.0, 2.0]),
        )
    )
    obstacle_environment.append(
        Ellipse(
            pose=Pose(np.array([-2, -3]), orientation=90 * math.pi / 180.0),
            axes_length=np.array([3.0, 2.0]),
        )
    )
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_lim = [-5, 5]
        y_lim = [-5, 5]
        n_grid = 11

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            # attractor_position=dynamic.attractor_position,
            do_quiver=True,
            # show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim, ax=ax
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

    position = np.array([-3, -2])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0
    assert velocity[1] > 0


def test_sequenced_linear_intersecting_circles(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([4, 0]), maximum_velocity=1.0)

    reference_point = np.array([0, 0])
    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            pose=Pose(np.array([0.0, -1.5])),
            axes_length=np.array([4.0, 4.0]),
        )
    )
    obstacle_environment[-1].set_reference_point(reference_point, in_global_frame=True)
    obstacle_environment.append(
        Ellipse(
            pose=Pose(np.array([0.0, 1.5])),
            axes_length=np.array([4.0, 4.0]),
        )
    )
    obstacle_environment[-1].set_reference_point(reference_point, in_global_frame=True)

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_lim = [-5, 5]
        y_lim = [-5, 5]
        n_grid = 11

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            # attractor_position=dynamic.attractor_position,
            do_quiver=True,
            # show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            draw_reference=True,
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

    position = np.array([-2.1, 1.0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] > 0, "Should move away from reference."


def test_sequenced_avoidance_dynamics_single(visualize=False):
    dynamics = create_segment_from_points(
        [[-4.0, -2.5], [0.0, -2.5], [0.0, 2.5], [4.0, 2.5]]
    )

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Cuboid(
            pose=Pose(np.array([0.2, 0.3]), 0 * math.pi / 180.0),
            axes_length=np.array([3.0, 1.0]),
        )
    )
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        # x_lim = [-4, 4]
        # y_lim = [-4, 4]
        x_lim = [-2, 2]
        y_lim = [-2, 2]

        n_grid = 20

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            # attractor_position=dynamic.attractor_position,
            do_quiver=True,
            # show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim, ax=ax
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

    position = np.array([-0.7, -0.4])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] < 0, "Avoiding towards the left"

    position = np.array([1.3, -0.8])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0, "Avoiding towards the right"


def test_sequenced_avoidance_dynamics_multiple(visualize=False):
    dynamics = create_segment_from_points(
        # [[-4.0, -2.5], [0.0, -2.5], [0.0, 2.5], [4.0, 2.5]]
        [[-7.0, -4.0], [0.0, -4.0], [0.0, 4.0], [7.5, 4.0]]
    )

    table_length = axes_length = np.array([1.5, 0.75])
    margin = 0.5

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[-3.4, -2.5], orientation=-0.1 * np.pi),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )
    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[4.0, 3.6], orientation=np.pi / 2),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )

    shared_reference = np.array([0.0, 0.75])
    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[1.2, 0.5], orientation=-0.2 * np.pi),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )
    obstacle_environment[-1].set_reference_point(shared_reference, in_global_frame=True)

    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[-0.4, 1.3], orientation=0.4 * np.pi),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )
    obstacle_environment[-1].set_reference_point(shared_reference, in_global_frame=True)
    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[-0.2, -3.9], orientation=-0.3 * np.pi),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )
    obstacle_environment.append(
        Cuboid(
            pose=Pose(position=[-0.3, 4.0], orientation=-0.9 * np.pi),
            axes_length=table_length,
            margin_absolut=margin,
        )
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=dynamics.attractor_position,
        initial_dynamics=dynamics,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    avoider = SingularityConvergenceDynamics(
        initial_dynamics=dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_lim = [-4, 4]
        y_lim = [-4, 4]
        # x_lim = [-2, 2]
        # y_lim = [-2, 2]

        n_grid = 20

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            # attractor_position=dynamic.attractor_position,
            do_quiver=True,
            # show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            draw_reference=True,
        )

        ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "*k")

        print(f"Time {(end - start)*1000:.2f} ms.")
        print(f"Number of obstacles: {obstacle_environment.n_obstacles}.")

    # position = np.array([-0.7, -0.4])
    # velocity = avoider.evaluate_sequence(position)
    # assert velocity[0] < 0, "Avoiding towards the left"

    # position = np.array([1.3, -0.8])
    # velocity = avoider.evaluate_sequence(position)
    # assert velocity[0] > 0, "Avoiding towards the right"


if (__name__) == "__main__":
    plt.close("all")

    # test_sequenced_linear_single_circle(visualize=True)
    # test_sequenced_linear_cuboid(visualize=True)
    # test_multiple_obstacles(visualize=True)
    # test_sequenced_avoidance_dynamics_single(visualize=True)
    # test_sequenced_avoidance_dynamics_multiple(visualize=True)
    test_sequenced_linear_intersecting_circles(visualize=True)
    # example_timeit()
