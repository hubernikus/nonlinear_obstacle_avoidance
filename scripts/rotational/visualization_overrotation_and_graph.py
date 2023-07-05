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


def visualize_simple_cube(n_resolution=20):
    attractor_position = np.array([0.0, 0.0])
    dynamics = SimpleCircularDynamics(
        pose=Pose(
            attractor_position,
        ),
        radius=0.1,
    )

    # dynamics = LinearSystem(attractor_position, maximum_velocity=1.0)

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

    x_lim = [-0.7, 0.1]
    y_lim = [-0.4, 0.4]

    figsize = (6, 5)

    fig, ax = plt.subplots(figsize=figsize)
    # plot_multi_obstacle_container(
    #     ax=ax, container=container, x_lim=x_lim, y_lim=y_lim
    # )

    # boundary = obstacle_tree[0].get_boundary_with_margin_xy()
    # ax.plot(boundary[0, :], boundary[1, :], "--", color="black")

    plot_multi_obstacle_container(
        container, ax=ax, x_lim=x_lim, y_lim=y_lim, alpha_obstacle=0.4
    )

    print("Doing dynamics.")
    plot_obstacle_dynamics(
        obstacle_container=[],
        dynamics=avoider.evaluate_sequence,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=n_resolution,
        attractor_position=dynamics.attractor_position,
    )


if (__name__) == "__main__":
    plt.close("all")
    figtype = ".pdf"

    visualize_simple_cube()

    print("Done")
