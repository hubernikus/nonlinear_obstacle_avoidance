import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from vartools.states import Pose
from vartools.colors import hex_to_rgba_float
from vartools.dynamics import QuadraticAxisConvergence, LinearSystem
from vartools.dynamics import WavyRotatedDynamics

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.arch_obstacle import BlockArchObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.visualization.plot_multi_obstacle import plot_multi_obstacles
from nonlinear_avoidance.visualization.plot_qolo import integrate_with_qolo


def test_normals_multi_arch(visualize=False, save_figure=False, n_grid=30):
    margin_absolut = 0.5
    attractor = np.array([4.0, -3])

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

    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=attractor, orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )

    avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        # convergence_dynamics=rotation_projector,
        create_convergence_dynamics=True,
    )

    if visualize:
        figsize = (4, 3.5)

        x_lim = [-6.5, 6.5]
        y_lim = [-5.5, 5.5]

        fig, ax = plt.subplots(figsize=figsize)
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        normals = np.zeros_like(positions)
        for pp in range(positions.shape[1]):
            if container.get_gamma(positions[:, pp]) < 1:
                continue

            normals[:, pp], _ = avoider.compute_averaged_normal_and_gamma(
                positions[:, pp]
            )

        plot_multi_obstacles(ax=ax, container=container)

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="red",
            # scale=quiver_scale,
            # alpha=quiver_alpha,
            # width=0.007,
            zorder=-1,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        fig, ax = plt.subplots(figsize=figsize)
        collision_checker = lambda pos: (not container.is_collision_free(pos))
        plot_obstacle_dynamics(
            obstacle_container=container,
            dynamics=avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.pose.position,
            collision_check_functor=collision_checker,
            do_quiver=True,
            show_ticks=False,
        )
        plot_multi_obstacles(ax=ax, container=container)

        # # Plot contourf of color
        # end_color = hex_to_rgba_float("7ea3caff")
        # colors = np.linspace([1.0, 1.0, 1.0], end_color[:3], 200)
        # my_cmap = ListedColormap(colors)

        # nx = ny = n_speed_resolution
        # x_vals, y_vals = np.meshgrid(
        #     np.linspace(x_lim[0], x_lim[1], nx),
        #     np.linspace(y_lim[0], y_lim[1], ny),
        # )
        # positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        # speed_magnitude = np.zeros(positions.shape[1])

        # for pp in range(positions.shape[1]):
        #     if container.get_gamma(positions[:, pp]) < 1:
        #         speed_magnitude[pp] = 0
        #         continue

        #     modulated_velocity = avoider.evaluate(positions[:, pp])
        #     speed_magnitude[pp] = np.linalg.norm(modulated_velocity)
        #     speed_magnitude[pp] = min(speed_magnitude[pp], 1.0)

        # contourf = ax.contourf(
        #     positions[0, :].reshape(nx, ny),
        #     positions[1, :].reshape(nx, ny),
        #     speed_magnitude.reshape(nx, ny),
        #     levels=np.linspace(0, 1.0001, 21),
        #     # extend="",
        #     zorder=-5,
        #     # cmap="Blues",
        #     cmap=my_cmap,
        #     alpha=1.0,
        # )

    position = np.array([1.58, 3.98])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate(position)
    assert rotated_velocity[0] > 0
    assert rotated_velocity[1] < 0
    assert abs(rotated_velocity[0]) > abs(rotated_velocity[1])
    assert np.linalg.norm(rotated_velocity) > 0.5, "Should still be fast!"

    position = np.array([-5.55, -2.25])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    rotated_velocity = avoider.evaluate(position)
    assert not np.any(np.isnan(rotated_velocity))
    assert np.linalg.norm(rotated_velocity) < 1, "Should be slowed down(!)"

    # Test slow-down
    position = np.array([-4.5, -4.9])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    modulated_velocity = avoider.evaluate(position)
    speed_magnitude = np.linalg.norm(modulated_velocity)
    assert np.isclose(speed_magnitude, 1.0), "No slow down when pointing down."

    # Test normal
    position = np.array([-3.7, 0.481])
    normal, gamma = avoider.compute_averaged_normal_and_gamma(position)
    assert abs(normal[0]) > abs(normal[1]), "Closest wall higher influence."
    assert normal[0] > 0, "Point away from the wall"
    assert normal[1] > 0, "Pointing up"


def visualize_double_arch(save_figure=False, it_max=160):
    # x_ lim = [-7, 7]
    # y_lim = [-6, 6]
    x_lim = [-6.5, 6.5]
    y_lim = [-5.5, 5.5]

    n_speed_resolution = 100
    n_grid = 20

    # Navy-Blue
    vectorfield_color = "#000080"

    figsize = (4, 3.5)
    margin_absolut = 0.5

    attractor = np.array([4.0, -3])
    # initial_dynamics = LinearSystem(
    #     attractor_position=attractor,
    #     maximum_velocity=1.0,
    # )

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

    kwargs_quiver = {
        # "color": "red",
        "scale": 18,
        "alpha": 1,
        "width": 0.007,
    }

    # Plot contourf of color
    end_color = hex_to_rgba_float("719bc5ff")
    end_color = hex_to_rgba_float("7ea3caff")
    colors = np.linspace([1.0, 1.0, 1.0], end_color[:3], 200)
    my_cmap = ListedColormap(colors)

    nx = ny = n_speed_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    speed_magnitude = np.zeros(positions.shape[1])

    collision_checker = lambda pos: (not container.is_collision_free(pos))
    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        vectorfield_color=vectorfield_color,
        obstacle_container=[],
        dynamics=initial_dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        attractor_position=initial_dynamics.pose.position,
        collision_check_functor=None,
        do_quiver=False,
        show_ticks=False,
        kwargs_quiver=kwargs_quiver,
    )
    # plot_multi_obstacles(ax=ax, container=container)

    for pp in range(positions.shape[1]):
        initial_velocity = initial_dynamics.evaluate(positions[:, pp])
        speed_magnitude[pp] = np.linalg.norm(initial_velocity)
        speed_magnitude[pp] = min(speed_magnitude[pp], 1.0)

    contourf = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        speed_magnitude.reshape(nx, ny),
        levels=np.linspace(0, 1.0001, 21),
        # extend="",
        zorder=-5,
        # cmap="Blues",
        cmap=my_cmap,
        alpha=1.0,
    )

    start_position = np.array([-2.5, 3])
    integrate_with_qolo(
        start_position=start_position,
        velocity_functor=initial_dynamics.evaluate,
        ax=ax,
        it_max=it_max,
        show_qolo=False,
    )

    if save_figure:
        fig_name = "two_arch_avoidance_initial"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=figsize)
    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        attractor_position=initial_dynamics.pose.position,
        collision_check_functor=collision_checker,
        do_quiver=True,
        show_ticks=False,
    )
    plot_multi_obstacles(ax=ax, container=container)

    for pp in range(positions.shape[1]):
        if container.get_gamma(positions[:, pp]) < 1:
            speed_magnitude[pp] = 0
            continue

        modulated_velocity = avoider.evaluate(positions[:, pp])
        speed_magnitude[pp] = np.linalg.norm(modulated_velocity)
        speed_magnitude[pp] = min(speed_magnitude[pp], 1.0)

    contourf = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        speed_magnitude.reshape(nx, ny),
        levels=np.linspace(0, 1.0001, 21),
        # extend="",
        zorder=-5,
        # cmap="Blues",
        cmap=my_cmap,
        alpha=1.0,
    )

    integrate_with_qolo(
        start_position=start_position,
        velocity_functor=avoider.evaluate,
        ax=ax,
        it_max=it_max,
    )

    if save_figure:
        fig_name = "two_arch_avoidance"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    # Save the legend
    save_legend = False
    if save_legend:
        # fig, ax = plt.subplots(figsize=(0.3 * figsize[0], figsize[1]))
        # cbar = ax.colorbar(contourf)
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax, ticks=[0, 1])
        cbar.ax.set_xlabel(r"$h(\xi)$")


if (__name__) == "__main__":
    figtype = ".pdf"
    # test_normals_multi_arch(visualize=False)
    visualize_double_arch(save_figure=True)
