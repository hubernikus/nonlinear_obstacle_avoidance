"""
Move Around Corners with Smooth Dynamics
"""
import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.dynamics.segmented_dynamics import create_segment_from_points


def main(n_grid=30):
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    # dynamics = create_segment_from_points([[0.0, 0], [4.0, 0], [4.0, 4.0], [8.0, 4.0]])
    dynamics = create_segment_from_points(
        [[-4.0, -2.5], [0.0, -2.5], [0.0, 2.5], [4.0, 2.5]]
    )

    # egments.append(DynamicsSegment(, [3.0, 0]))
    # segments.append(DynamicsSegment([4.0, 1], [4.0, 5]))
    # segments.append(DynamicsSegment([5.0, 6], [8.0, 6.0]))

    fig, ax = plt.subplots(figsize=(6, 5))
    for segment in dynamics.segments:
        ax.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            marker="o",
        )
    # position = np.array([4.0, 0])
    position = np.array([5.2, 7.75])
    velocity = dynamics.evaluate(position)
    position = np.array([5.2, 8.25])
    velocity = dynamics.evaluate(position)

    plot_obstacle_dynamics(
        obstacle_container=[],
        dynamics=dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        do_quiver=True,
        # show_ticks=False,
    )

    dimension = 2
    it_max = 1000
    dt = 0.1
    trajectory = np.zeros((dimension, it_max))
    position_start = np.array([-4, -4.0])
    trajectory[:, 0] = position_start
    for ii in range(1, it_max):
        velocity = dynamics.evaluate(trajectory[:, ii - 1])
        trajectory[:, ii] = trajectory[:, ii - 1] + velocity * dt

    ax.plot(trajectory[0, :], trajectory[1, :], color="red")
    ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color="red")
    # position =


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    main()
