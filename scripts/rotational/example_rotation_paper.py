#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy
from functools import partial
import math
from typing import Optional, Callable

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence
from vartools.states import ObjectPose
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.visualization import VectorfieldPlotter

# from vartools.dynamical_systems import SinusAttractorSystem
# from vartools.dynamical_systems import BifurcationSpiral

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.multiboundary_container import MultiBoundaryContainer
from nonlinear_avoidance.dynamics import WavyLinearDynamics
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import obstacle_avoidance_rotational
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)
from nonlinear_avoidance.nonlinear_rotation_avoider import NonlinearRotationalAvoider

from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def function_integrator(
    start_point: np.ndarray,
    dynamics: Callable[[np.ndarray], np.ndarray],
    stopping_functor: Optional[Callable[[np.ndarray], bool]] = None,
    stepsize: float = 0.1,
    err_abs: float = 1e-1,
    it_max: int = 100,
) -> np.ndarray:
    points = np.zeros((start_point.shape[0], it_max + 1))
    points[:, 0] = start_point

    for ii in range(it_max):
        velocity = dynamics(points[:, ii])
        points[:, ii + 1] = points[:, ii] + velocity * stepsize

        if stopping_functor is not None and stopping_functor(points[:, ii]):
            print(f"Stopped at it={ii}")
            break

        if err_abs is not None and np.linalg.norm(velocity) < err_abs:
            print(f"Converged at it={ii}")
            break

    return points[:, : ii + 1]


def inverted_star_obstacle_avoidance(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()

    obstacle_list.append(
        StarshapedFlower(
            center_position=np.array([0, 0]),
            radius_magnitude=2,
            number_of_edges=5,
            radius_mean=7,
            orientation=0.0 / 180 * pi,
            tail_effect=False,
            is_boundary=True,
        )
    )
    # initial_dynamics = WavyLinearDynamics(attractor_position=np.array([0, 0]))
    initial_dynamics = WavyLinearDynamics(attractor_position=np.array([6.8, -1]))
    convergence_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position
    )

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )

    if visualize:
        plt.close("all")

        x_lim = [-10, 10]
        y_lim = [-10, 10]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_list,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "wavy_nonlinear_ds_in_star_obstacle"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def visualization_inverted_ellipsoid(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([7, 4]),
            is_boundary=True,
            orientation=30 / 180.0 * math.pi,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.2, 1]))
    convergence_dynamics = initial_dynamics

    # main_avoider = RotationalAvoider()
    # my_avoider = partial(main_avoider.avoid, convergence_radius=math.pi)

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )
    obstacle_avoider.convergence_radius = math.pi

    if visualize:
        plt.close("all")

        x_lim = [-3.4, 3.4]
        y_lim = [-3.4, 3.4]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim=x_lim,
            y_lim=y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_avoider.obstacle_environment,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=obstacle_avoider.initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "linear_dynamics_in_repulsive_ellipse_wall"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def quiver_single_circle_linear_repulsive(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=math.pi,
    )

    if visualize:
        # Arbitrary constant velocity
        tmp_dynamics = LinearSystem(attractor_position=np.array([2.0, 0]))
        tmp_dynamics.distance_decrease = 0.1
        obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
        # ConvergingDynamics=ConstantValue (initial_velocity)
        tmp_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_list,
            convergence_radius=math.pi,
        )
        x_lim = [-2, 3]
        y_lim = [-2.2, 2.2]
        n_grid = 13
        alpha_obstacle = 1.0

        plt.close("all")

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
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

        tmp_avoider.convergence_radius = math.pi * 3.0 / 4.0
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
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
            fig_name = "circular_repulsion_pi_3_4"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

        tmp_avoider.convergence_radius = math.pi * 1.0 / 2.0
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
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
            fig_name = "circular_repulsion_pi_1_2"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def integration_smoothness_around_ellipse(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([3.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        # convergence_radius=math.pi/2,
        tail_rotation=False,
    )

    if visualize:
        plt.close("all")
        x_lim = [-3, 3.8]
        y_lim = [-3.2, 3.3]

        it_max = 100

        n_grid = 10
        points_bottom = np.vstack(
            (np.linspace(x_lim[0], x_lim[1], n_grid), y_lim[0] * np.ones(n_grid))
        )

        points_left = np.vstack(
            (x_lim[0] * np.ones(n_grid), np.linspace(y_lim[1], y_lim[0], n_grid))
        )

        points_top = np.vstack(
            (np.linspace(x_lim[0], x_lim[1], n_grid), y_lim[1] * np.ones(n_grid))
        )

        all_points = np.hstack((points_bottom, points_left, points_top))
        is_colliding = (
            lambda x: main_avoider.obstacle_environment.get_minimum_gamma(x) < 1
        )

        # smoothness = ["00"]
        smoothness = ["00", "03", "10"]
        for ss_string in smoothness:
            # ss =
            ss = float(ss_string) / 10.0
            if not ss:
                ss = 1e-5
            main_avoider.smooth_continuation_power = ss

            fig, ax = plt.subplots(figsize=(3.5, 3))

            for pp in range(all_points.shape[1]):
                points = function_integrator(
                    start_point=all_points[:, pp],
                    dynamics=main_avoider.evaluate,
                    stopping_functor=is_colliding,
                    stepsize=0.03,
                    err_abs=1e-1,
                    it_max=200,
                )

                ax.plot(points[0, :], points[1, :], color="blue")

            point_convergence = np.array([x_lim[0], 0.0])
            points = function_integrator(
                start_point=point_convergence,
                dynamics=main_avoider.evaluate,
                stopping_functor=is_colliding,
                stepsize=0.03,
                err_abs=1e-1,
                it_max=200,
            )
            ax.plot(points[0, :], points[1, :], color="red", linewidth=2)

            plot_obstacles(
                obstacle_container=obstacle_list,
                ax=ax,
                alpha_obstacle=1.0,
                x_lim=x_lim,
                y_lim=y_lim,
            )

            ax.plot(
                initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1],
                "k*",
                linewidth=18.0,
                markersize=18,
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

            if save_figure:
                fig_name = "comparison_linear_integration_" + ss_string
                fig.savefig(
                    "figures/" + fig_name + figtype, bbox_inches="tight", dpi=300
                )


def convergence_direction_comparison_for_circular_dynamics(
    visualize=True, save_figure=False, n_resolution=120
):
    # from vartools.dynamical_systems import CircularStable
    obstacle_environment = RotationContainer()
    center = np.array([2.2, 0.0])
    # obstacle_environment.append(
    #     Ellipse(
    #         center_position=center,
    #         axes_length=np.array([2.0, 1.0]),
    #         orientation=45 * math.pi / 180.0,
    #         # margin_absolut=0.3,
    #     )
    # )
    obstacle_environment.append(
        StarshapedFlower(
            center_position=center,
            radius_magnitude=0.3,
            number_of_edges=4,
            radius_mean=0.75,
            orientation=30 / 180 * pi,
            distance_scaling=1,
            # is_boundary=True,
        )
    )

    # circular_ds = CircularStable(radius=2.5, maximum_velocity=2.0)
    attractor_position = np.array([0.0, 0])
    circular_ds = SimpleCircularDynamics(
        radius=2.0,
        pose=ObjectPose(
            position=attractor_position,
        ),
    )

    # Simple Setup
    convergence_dynamics = LinearSystem(attractor_position=attractor_position)
    obstacle_avoider_globally_straight = RotationalAvoider(
        initial_dynamics=circular_ds,
        obstacle_environment=obstacle_environment,
        convergence_system=convergence_dynamics,
    )

    # Convergence direction [local]
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - circular_ds.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=circular_ds,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
        # Currently not working... -> rotational summing needs to be improved..
        # convergence_radius=math.pi * 3 / 4,
    )

    x_lim = [-3.0, 3.4]
    y_lim = [-3.0, 3.0]
    figsize = (5.0, 4.5)
    vf_color = "blue"
    traj_color = "#DB4914"
    # traj_color = "blue"
    traj_base_color = "#808080"
    it_max = 320
    lw = 4

    start_integration = np.array([-2.005, 0])
    pos_traj_base = function_integrator(
        start_integration, circular_ds.evaluate, it_max=it_max, stepsize=0.05
    )

    fig, ax = plt.subplots(figsize=figsize)
    start_integration = np.array([-2.060, 0])
    pos_traj_global = function_integrator(
        start_integration, obstacle_avoider.evaluate, it_max=it_max, stepsize=0.05
    )
    ax.plot(
        pos_traj_global[0, :], pos_traj_global[1, :], linewidth=lw, color=traj_color
    )
    ax.plot(
        pos_traj_base[0, :],
        pos_traj_base[1, :],
        # "--",
        linewidth=lw,
        color=traj_base_color,
        zorder=-3,
    )

    plot_obstacle_dynamics(
        obstacle_container=obstacle_environment,
        # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
        dynamics=obstacle_avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        # do_quiver=True,
        vectorfield_color=vf_color,
        attractor_position=attractor_position,
    )
    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        noTicks=True,
        # show_ticks=False,
    )

    if save_figure:
        figname = "rotational_avoidance_local_convergence_direction"
        plt.savefig(
            "figures/" + figname + figtype,
            bbox_inches="tight",
        )

    fig, ax = plt.subplots(figsize=figsize)
    # traj_color = "#FF9B00"
    start_integration = np.array([-2.0037, 0])
    pos_traj_global = function_integrator(
        start_integration,
        obstacle_avoider_globally_straight.evaluate,
        it_max=it_max,
        stepsize=0.05,
    )
    ax.plot(
        pos_traj_base[0, :],
        pos_traj_base[1, :],
        # "--",
        linewidth=lw,
        color=traj_base_color,
        zorder=-3,
    )
    plt.plot(
        pos_traj_global[0, :], pos_traj_global[1, :], linewidth=lw, color=traj_color
    )

    plot_obstacle_dynamics(
        obstacle_container=obstacle_environment,
        # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
        dynamics=obstacle_avoider_globally_straight.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        # do_quiver=True,
        vectorfield_color=vf_color,
        attractor_position=attractor_position,
    )
    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        noTicks=True,
        # show_ticks=False,
    )

    if save_figure:
        figname = "rotational_avoidance_global_convergence_direction"
        plt.savefig(
            "figures/" + figname + figtype,
            bbox_inches="tight",
        )


def convergence_direction_comparison_for_linear_spiraling_dynamics(
    visualize=True, save_figure=False, n_resolution=120, fig_basename="roam_linear_ds_"
):
    from scripts.rotational.example_mapping_four_figure import get_main_obstacle
    from scripts.rotational.example_mapping_four_figure import get_initial_dynamics

    dimension = 2
    initial_dynamics = get_initial_dynamics()

    container = MultiObstacleContainer()
    new_tree = MultiObstacle(Pose.create_trivial(dimension))
    new_tree.set_root(get_main_obstacle())
    container.append(new_tree)

    avoider_with_local = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.55 * math.pi,
    )

    avoider_with_global = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        default_dynamics=LinearSystem(initial_dynamics.attractor_position),
        create_convergence_dynamics=True,
        # convergence_radius=0.53 * math.pi,
    )

    x_lim = [-2.5, 4.0]
    y_lim = [-3.0, 3.0]
    figsize = (5.0, 4.5)
    vf_color = "blue"
    traj_color = "#DB4914"
    # traj_color = "blue"
    traj_base_color = "#808080"
    it_max = 400
    lw = 4
    err_conv = 1e-2

    # start_integration = np.array([x_lim[0], -2.7])
    start_integration = np.array([3.1, y_lim[0]])
    pos_traj_base = function_integrator(
        start_integration, initial_dynamics.evaluate, it_max=it_max, stepsize=0.05
    )

    # Local convergence
    fig, ax = plt.subplots(figsize=figsize)
    pos_traj_global = function_integrator(
        start_integration,
        avoider_with_local.evaluate,
        it_max=it_max,
        stepsize=0.05,
        err_abs=err_conv,
    )
    ax.plot(
        pos_traj_global[0, :], pos_traj_global[1, :], linewidth=lw, color=traj_color
    )
    ax.plot(
        pos_traj_base[0, :],
        pos_traj_base[1, :],
        # "--",
        linewidth=lw,
        color=traj_base_color,
        zorder=-3,
    )

    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=avoider_with_local.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        attractor_position=initial_dynamics.attractor_position,
        vectorfield_color=vf_color,
    )

    plot_multi_obstacle_container(
        ax=ax,
        container=container,
        x_lim=x_lim,
        y_lim=y_lim,
        draw_reference=True,
        noTicks=True,
    )

    if save_figure:
        figname = "local_convergence_direction"
        plt.savefig(
            "figures/" + fig_basename + figname + figtype,
            bbox_inches="tight",
        )

    # Global convergence
    fig, ax = plt.subplots(figsize=figsize)
    # traj_color = "#FF9B00"
    pos_traj_global = function_integrator(
        start_integration,
        avoider_with_global.evaluate,
        it_max=it_max,
        stepsize=0.05,
        err_abs=err_conv,
    )
    ax.plot(
        pos_traj_base[0, :],
        pos_traj_base[1, :],
        # "--",
        linewidth=lw,
        color=traj_base_color,
        zorder=-3,
    )
    plt.plot(
        pos_traj_global[0, :], pos_traj_global[1, :], linewidth=lw, color=traj_color
    )

    plot_obstacle_dynamics(
        obstacle_container=container,
        # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
        dynamics=avoider_with_global.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        # do_quiver=True,
        vectorfield_color=vf_color,
        attractor_position=initial_dynamics.attractor_position,
    )

    plot_multi_obstacle_container(
        ax=ax,
        container=container,
        x_lim=x_lim,
        y_lim=y_lim,
        draw_reference=True,
        noTicks=True,
    )

    if save_figure:
        figname = "global_convergence_direction"
        plt.savefig(
            "figures/" + fig_basename + figname + figtype,
            bbox_inches="tight",
        )


if (__name__) == "__main__":
    figtype = ".pdf"
    # figtype = ".png"

    plt.close("all")
    plt.ion()

    # inverted_star_obstacle_avoidance(visualize=True, save_figure=True)
    # visualization_inverted_ellipsoid(visualize=True, save_figure=True)
    # quiver_single_circle_linear_repulsive(visualize=True, save_figure=False)
    # integration_smoothness_around_ellipse(visualize=True, save_figure=True)

    # convergence_direction_comparison_for_circular_dynamics(
    #     visualize=True,
    #     save_figure=True,
    #     n_resolution=100,
    # )

    convergence_direction_comparison_for_linear_spiraling_dynamics(
        visualize=True,
        save_figure=True,
        n_resolution=120,
    )

    print("--- done ---")
