""" Test mapping-weights"""
import math

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower

from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def test_convergence_weight_around_flower(visualize=False):
    attractor_position = np.array([0.0, 0.0])

    obstacle = StarshapedFlower(
        center_position=np.array([2.2, 0.0]),
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        # distance_scaling=1,
        distance_scaling=2.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.zeros(2),
        A_matrix=np.array([[-1, -2], [2, -1]]),
        maximum_velocity=1.0,
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=initial_dynamics.pose.position,
        initial_dynamics=initial_dynamics,
        # reference_velocity=lambda x: x - initial_dynamics.center_position,
    )

    if visualize:
        x_lim = [-2.5, 4.0]
        y_lim = [-3.0, 3.0]
        figsize = (10, 8)
        n_resolution = 100

        fig, ax = plt.subplots(figsize=figsize)

        # Create points
        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        convergence_weights = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            convergence_weights[pp] = rotation_projector.evaluate_projected_weight(
                positions[:, pp], obstacle=obstacle
            )
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            convergence_weights.reshape(nx, ny),
            cmap="binary",
            # alpha=kwargs["weights_alpha"],
            # extend="max",
            # vmin=1.0,
            levels=np.linspace(0, 1, 11),
            zorder=-1,
        )
        cbar = fig.colorbar(cs, ticks=np.linspace(0, 1.0, 11))

        plot_obstacles(
            ax=ax,
            obstacle_container=[obstacle],
            alpha_obstacle=0.0,
            draw_reference=True,
            draw_center=False,
        )

    # Check slightly inside
    position = np.array([1.7, 0.8])
    convergence_weight = rotation_projector.evaluate_projected_weight(
        position, obstacle=obstacle
    )
    assert np.isclose(convergence_weight, 1.0)


if (__name__) == "__main__":
    test_convergence_weight_around_flower(visualize=True)
