"""
Gammafield evaluation visualization functions for different scenarios.
"""
# Author: Lukas Huber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021
import warnings

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.visualization import plot_obstacles


# TODO: make test out of it...
def test_convergence_direction_multihull(
    obstacle_list,
    it_obs,
    n_resolution=30,
    x_lim=None,
    y_lim=None,
    dim=2,
    ax=None,
    assert_check=True,
):
    """Test-like drawing of  a list of obstacles and evaluate the gamma-field
    of the obstacle at 'it_obs'."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    elif x_lim is None:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

    else:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plot_obstacles(
        obstacle_container=obstacle_list,
        ax=ax,
        x_range=x_lim,
        y_range=y_lim,
        noTicks=True,
        showLabel=False,
    )

    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

    positions = np.zeros((dim, n_resolution, n_resolution))
    conv_dir = np.zeros((dim, n_resolution, n_resolution))

    ind_no_collision = np.zeros((n_resolution, n_resolution), dtype=bool)

    attractor_is_inside = (
        obstacle_list[it_obs].get_gamma(
            obstacle_list.pos_attractor, in_global_frame=True
        )
        > 1
    )
    for ix in range(n_resolution):
        for iy in range(n_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            ind_no_collision[ix, iy] = (
                obstacle_list[it_obs].get_gamma(
                    positions[:, ix, iy], in_global_frame=True
                )
                >= 1
            )

            if not ind_no_collision[ix, iy]:
                continue

            conv_dir[:, ix, iy] = obstacle_list.get_convergence_direction(
                position=positions[:, ix, iy], it_obs=it_obs
            )

            # breakpoint()

            if assert_check:
                if attractor_is_inside:
                    assert (
                        np.dot(
                            conv_dir[:, ix, iy],
                            obstacle_list.pos_attractor - positions[:, ix, iy],
                        )
                        >= 0
                    ), "Convergence direction in wrong direction for attractor-piece."

                else:
                    local_attractor = obstacle_list[
                        it_obs
                    ].get_intersection_with_surface(
                        direction=(
                            obstacle_list._end_points[:, it_obs]
                            - obstacle_list[it_obs].center_position
                        ),
                        in_global_frame=True,
                    )

                    assert (
                        np.dot(
                            conv_dir[:, ix, iy], local_attractor - positions[:, ix, iy]
                        )
                        >= 0
                    ), "Convergence direction in wrong direction for local-attractor."

    ind_flatten = ind_no_collision.flatten()

    xx = positions[0, :, :].flatten()[ind_flatten]
    yy = positions[1, :, :].flatten()[ind_flatten]

    vel_x = conv_dir[0, :, :].flatten()[ind_flatten]
    vel_y = conv_dir[1, :, :].flatten()[ind_flatten]

    quiv = ax.quiver(xx, yy, vel_x, vel_y, color="black", zorder=3)

    ax.plot(
        obstacle_list.pos_attractor[0], obstacle_list.pos_attractor[1], "k*", zorder=5
    )
