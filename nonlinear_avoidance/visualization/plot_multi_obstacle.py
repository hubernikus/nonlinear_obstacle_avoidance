import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider


def plot_multi_obstacles(
    container: MultiObstacleAvoider, x_lim=None, y_lim=None, ax=None, show_ticks=True
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.set_aspect("equal", adjustable="box")
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    for obstacle_tree in container._obstacle_list:
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_tree._obstacle_list,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=not (show_ticks),
        )

    return fig, ax
