from vartools.states import Pose

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import QuadraticAxisConvergence, LinearSystem

from nonlinear_avoidance.arch_obstacle import MultiObstacleContainer, BlockArchObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider


def visualize_double_arch():
    x_lim = [-5, 5]
    y_lim = [-5, 5]
    n_grid = 10

    margin_absolut = 0.5

    attractor = np.array([4.0, -3])
    initial_dynamics = LinearSystem(
        attractor_position=attractor,
        maximum_velocity=1.0,
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

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        attractor_position=initial_dynamics.attractor_position,
        do_quiver=True,
        show_ticks=False,
    )

    plot_obstacles(
        obstacle_container=obstacle_list,
        ax=ax,
        alpha_obstacle=alpha_obstacle,
    )

    if save_figure:
        fig_name = "circular_repulsion_pi"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    visualize_double_arch()
