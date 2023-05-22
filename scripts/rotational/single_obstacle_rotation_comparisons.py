#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# License: BSD (c) 2021

import copy
import math
import warnings

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from vartools.colors import hex_to_rgba_float
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence
from vartools.dynamical_systems import BifurcationSpiral
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from nonlinear_avoidance.multiboundary_container import (
    MultiBoundaryContainer,
)
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import obstacle_avoidance_rotational
from nonlinear_avoidance.avoidance import RotationalAvoider


from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics

from vartools.visualization import VectorfieldPlotter


def single_ellipse_linear_triple_plot_quiver(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_linear_"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.plottype = "quiver"
    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3

    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def rotated_ellipse_linear_triple_plot_quiver(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_rotated_"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=30.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.plottype = "quiver"
    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3

    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def single_ellipse_nonlinear_triple_plot(n_resolution=100, save_figure=False):
    figure_name = "comparison_nonlinear_vectorfield"

    initial_dynamics = QuadraticAxisConvergence(
        stretching_factor=3,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=np.array([8, 0]),
    )

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def single_ellipse_linear_triple_plot_streampline(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_linear_streamline"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_03")

    # New one
    my_avoider.smooth_continuation_power = 0.0
    my_plotter.create_new_figure()
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_00")

    # New one
    my_avoider.smooth_continuation_power = 1.0
    my_plotter.create_new_figure()
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_10")


def single_ellipse_linear_triple_integration_lines(
    save_figure=False, it_max=300, dt_step=0.02
):
    figure_name = "comparison_linear_integration"

    # initial_dynamics = QuadraticAxisConvergence(
    # stretching_factor=3,
    # maximum_velocity=1.0,
    # dimension=2,
    # attractor_position=np.array([8, 0]),
    # )
    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    my_plotter = VectorfieldPlotter(
        x_lim=x_lim,
        y_lim=y_lim,
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )
    my_plotter.obstacle_alpha = 1
    my_plotter.dt_step = dt_step
    my_plotter.it_max = it_max

    initial_positions = np.vstack(
        (
            np.linspace([x_lim[0], y_lim[0]], [x_lim[1], y_lim[0]], 10),
            np.linspace([x_lim[0], y_lim[0]], [x_lim[0], y_lim[1]], 10),
            np.linspace([x_lim[0], y_lim[1]], [x_lim[1], y_lim[1]], 10),
        )
    ).T

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    # New continuation power value
    my_avoider.smooth_continuation_power = 0.3
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )

    if save_figure:
        my_plotter.save(figure_name + "_03")

    # if True:
    # return

    # New continuation power value
    my_avoider.smooth_continuation_power = 0.0

    my_plotter.create_new_figure()
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )
    if save_figure:
        my_plotter.save(figure_name + "_00")

    # New continuation power value
    my_avoider.smooth_continuation_power = 1.0

    my_plotter.create_new_figure()
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )
    if save_figure:
        my_plotter.save(figure_name + "_10")


def single_ellipse_spiral_triple_plot(save_figure=False, n_resolution=40):
    # TODO: this does not work very well...
    figure_name = "spiral_single_ellipse"

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0, -5]),
        A_matrix=np.array([[-1.0, -3.0], [3.0, -1.0]]),
    )

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=30.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)
    my_plotter = VectorfieldPlotter(
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )
    if save_figure:
        my_plotter.save(figure_name + "_initial")


def visualize_starshape_repulsion(save_figure=False, n_speed_resolution=10):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        StarshapedFlower(
            center_position=np.zeros(2),
            number_of_edges=3,
            radius_magnitude=0.2,
            radius_mean=0.75,
            orientation=10 / 180 * math.pi,
            distance_scaling=1.0,
            # is_boundary=True,
        )
    )

    # Velocity towards attractor
    initial_dynamics = LinearSystem(
        attractor_position=np.array([1.5, 0]),
        maximum_velocity=3.5,
        distance_decrease=1.0,
    )

    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    obstacle_list.set_convergence_directions(
        converging_dynamics=initial_dynamics,
    )
    # ConvergingDynamics=ConstantValue (initial_velocity)
    x_lim = [-1.7, 2.0]
    y_lim = [-2.0, 2.0]
    n_grid_quiver = 10
    alpha_obstacle = 1.0

    # plt.close("all")
    # name_list = ["\pi / 2", "3 \pi / 4", "\pi"]
    # angle_list = [math.pi * 0.5, math.pi * 0.75, math.pi]
    name_list = ["\pi / 2", "\pi"]
    angle_list = [math.pi * 0.5, math.pi]
    fig, axs = plt.subplots(1, len(angle_list), figsize=(5.5, 2.5))

    # end = np.array([8, 54, 116]) / 256
    end = np.array([8, 80, 155]) / 256

    # colors = np.linspace([1.0, 1.0, 1.0], end, 200)
    # my_cmap = ListedColormap(colors)
    end_color = hex_to_rgba_float("7ea3caff")
    colors = np.linspace([1.0, 1.0, 1.0], end_color[:3], 200)
    my_cmap = ListedColormap(colors)
    vectorfield_color = "#000080"

    for ii, (angle_name, angle) in enumerate(zip(name_list, angle_list)):
        ax = axs[ii]
        obstacle_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_list,
            convergence_radius=angle,
        )

        nx = ny = n_grid_quiver
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        velocities = np.zeros_like(positions)

        for pp in range(positions.shape[1]):
            if obstacle_list[-1].get_gamma(positions[:, pp], in_global_frame=True) < 1:
                continue
            velocities[:, pp] = obstacle_avoider.evaluate(positions[:, pp])

        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            color=vectorfield_color,
            scale=34,
            alpha=1.0,
            width=0.011,
            zorder=0,
        )
        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )

        # Evaluate 'repulsion-factor'
        nx = ny = n_speed_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        speed_magnitude = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            initial_velocity = initial_dynamics.evaluate(positions[:, pp])
            rotated_velocity = obstacle_avoider.evaluate(positions[:, pp])

            if not np.linalg.norm(initial_velocity):
                continue

            speed_magnitude[pp] = np.linalg.norm(rotated_velocity) / np.linalg.norm(
                initial_velocity
            )
            speed_magnitude[pp] = min(speed_magnitude[pp], 1.0)

        contourf = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            speed_magnitude.reshape(nx, ny),
            levels=np.linspace(0, 1.0001, 21),
            # extend="",
            zorder=-2,
            # cmap="Blues",
            cmap=my_cmap,
            alpha=1.0,
        )
        ax.set_xlabel(rf"$R^e = {angle_name}$")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.scatter(
            initial_dynamics.attractor_position[0],
            initial_dynamics.attractor_position[1],
            marker="*",
            s=200,
            color=vectorfield_color,
            zorder=5,
        )

        ax.tick_params(
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, ticks=[0, 1])
    cbar.ax.set_xlabel(r"$h(\xi)$")

    if save_figure:
        fig_name = "circular_repulsion_multiplot"
        fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    figtype = ".pdf"
    # plt.close("all")
    plt.ion()

    # single_ellipse_linear_triple_plot_quiver(save_figure=True, n_resolution=15)
    # single_ellipse_linear_triple_integration_lines(save_figure=False)
    # single_ellipse_linear_triple_plot_streampline(save_figure=False, n_resolution=30)
    # single_ellipse_nonlinear_triple_plot(save_figure=True, n_resolution=40)
    # rotated_ellipse_linear_triple_plot_quiver(save_figure=False, n_resolution=30)

    visualize_starshape_repulsion(save_figure=True, n_speed_resolution=100)
    # visualize_starshape_repulsion(save_figure=True, n_speed_resolution=100)
