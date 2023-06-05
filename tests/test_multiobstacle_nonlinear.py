import numpy as np

import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import LinearSystem

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.dynamics import SimpleCircularDynamics


def test_straight_system(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

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

    position = np.array([3.0, 0])
    # velocity = avoider.evaluate(position)
    velocity = avoider.evaluate_sequence(position)
    assert np.isclose(velocity[1], 0), "Going towards center"
    assert velocity[0] < 0, "Going towards center"

    position = np.array([0.0, 3.0])
    # velocity = avoider.evaluate(position)
    velocity = avoider.evaluate_sequence(position)
    assert velocity[1] < 0, "Going towards center"
    assert velocity[0] > 0, "Going towards center"

    position = np.array([-5.0, 0.0])
    # velocity = avoider.evaluate(position)
    velocity = avoider.evaluate_sequence(position)
    assert np.isclose(velocity[1], 0), "Going towards center"
    assert velocity[0] > 0, "Going towards center"

    position = np.array([-5.0, 1.0])
    # velocity = avoider.evaluate(position)
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0, "Going towards center"
    assert velocity[1] > 0, "Avoiding towards the top"


def test_convergence_direction(visualize=False):
    margin_absolut = 0
    distance_scaling = 5
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.3]),
        ),
        radius=0.1,
    )

    container = MultiObstacleContainer()
    pose_tree = Pose(np.array([-0.2, 0.30]))
    box1 = MultiObstacle(pose_tree)
    pos_bos1 = np.array([0, -0.06])
    box1.set_root(
        Cuboid(
            center_position=pos_bos1,
            axes_length=np.array([0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(box1)
    for obs in container:
        obs.update_pose(obs.pose)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        # x_lim = [-0.6, 0.4]
        # y_lim = [-0.4, 0.6]

        # x_lim = [-0.4, 0.4]
        # y_lim = [-0.1, 0.6]

        x_lim = [0.04, 0.1]
        y_lim = [0.4, 0.5]

        n_resolution = 30
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(ax=ax, obstacle_container=box1, x_lim=x_lim, y_lim=y_lim)

        plot_obstacle_dynamics(
            obstacle_container=box1,
            dynamics=avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position,
        )

    visualize_initial = False
    if visualize_initial:
        # Initial
        fig, ax = plt.subplots(figsize=figsize)
        # plot_obstacles(ax=ax, obstacle_container=box1, x_lim=x_lim, y_lim=y_lim)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            # do_quiver=False,
            # # do_quiver=True,
            # vectorfield_color=vf_color,
            # attractor_position=attractor_position,
        )

    position = np.array([0.065, 0.46])
    velocity2 = avoider.evaluate(position)

    position = np.array([0.06, 0.46])
    velocity1 = avoider.evaluate(position)

    # assert np.allclose(velocity1, velocity2, atol=1e-1), "Expected to be close"


if (__name__) == "__main__":
    figtype = ".pdf"

    test_straight_system(visualize=True)
    # test_convergence_direction(visualize=True)
