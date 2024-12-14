"""Test / visualization of line following."""

import matplotlib.pyplot as plt
from vartools.dynamical_systems import CircularStable
from nonlinear_avoidance.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics


def _test_circle_following_avoidance(visualize=False):
    global_ds = CircularStable(radius=10, maximum_velocity=1)

    if visualize:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=global_ds.evaluate,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_grid=30,
            ax=ax,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

    RotationContainer()


if (__name__) == "__main__":
    _test_circle_following_avoidance(visualize=True)
    print("Line-follwing complete.")
    input()