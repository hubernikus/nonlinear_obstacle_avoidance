#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""
__author__ = "lukashuber"
__date__ = "2021-05-16"

import os
import numpy as np


import matplotlib.pyplot as plt

# from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles import Ellipse
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.multihull_convergence import (
    multihull_attraction,
)
from nonlinear_avoidance.multiboundary_container import (
    MultiBoundaryContainer,
)

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles


# from nonlinear_avoidance.gmm_learner.learner.directional import DirectionalGMM
from nonlinear_avoidance.gmm_learner.learner.directional_gmm import DirectionalGMM
from nonlinear_avoidance.gmm_learner.graph import GraphGMM
from nonlinear_avoidance.gmm_learner.learner.visualizer import (
    plot_position_data,
    plot_graph_and_gaussians,
    plot_obstacle_wall_environment,
)
from nonlinear_avoidance.gmm_learner.visualization.gmm_visualization import (
    draw_gaussians,
    draw_obstacle_patches,
)


def test_inverted_simple_ellipse_comparison():
    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-3.38850563, 0.37817845]),
            axes_length=np.array([0.50284162, 0.91692708]),
            orientation=2.409204774887144,
            is_boundary=True,
        )
    )

    position = np.array([-3.421, 0.8947])
    initial_velocity = np.array([0.65816827, 0.75287086])

    main_avoider = RotationalAvoider()
    final_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_environment,
        convergence_velocity=np.array([0.63600316, 0.17352264]),
    )

    assert np.all(final_velocity > 0), "Velocity should point up-left."


def _test_visualization_a_shape(visualize=False, save_figure=False):
    # TODO -> maybe speed up this test by avoiding the learning part...
    dataname = "2D_Ashape"
    x_lim = [-6.0, 1.0]
    # x_lim = [-5.2, -0.1]
    y_lim = [-0.8, 2.5]
    figsize = (6.5, 3.1)

    fig_groupname = "multihull_avoidance" + "_" + dataname

    n_gaussian = 4
    np.random.seed(2)

    dataset_name = os.path.join("dataset", dataname + ".mat")

    # GMM-learner is also a multiboundary-container
    gmm_learner = GraphGMM(file_name=dataset_name, n_gaussian=n_gaussian)
    gmm_learner.create_learned_boundary()
    gmm_learner.update_intersection_graph()
    gmm_learner.set_convergence_directions(NonlinearDynamcis=gmm_learner)

    avoider_functor = lambda x: multihull_attraction(
        x, gmm_learner.predict(x), obstacle_list=gmm_learner
    )

    if visualize:
        # gmm_learner.plot_graph_and_gaussians()
        # n_resolution = 20
        # do_quiver = True

        n_resolution = 120
        do_quiver = False

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            # obstacle_container=gmm_learner,
            obstacle_container=[],
            dynamics=gmm_learner.predict,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_resolution,
            ax=ax,
            # attractor_position=gmm_learner.attractor_position,
            do_quiver=do_quiver,
            show_ticks=True,
        )

        # Downsample (?)
        data_points = gmm_learner.positions
        it_sample = np.arange(0, data_points.shape[0] - 1, 3, dtype=int)
        data_points = data_points[it_sample, :]

        # Downsample?
        ax.plot(
            data_points[:, 0],
            data_points[:, 1],
            ".",
            # color="#DB4391",
            color="black",
            markersize=4.0,
            label="Initial Data",
        )
        obs_colors = [
            "turquoise",
            "darkorange",
            "red",
            "green",
            "purple",
            "tan",
        ]
        draw_obstacle_patches(
            obstacle_list=gmm_learner,
            ax=ax,
            plot_centers=False,
            # axes_scaling=0.9,
            ellipse_alpha=0.5,
            colors=obs_colors,
        )
        # gmm_learner.plot_vectorfield_and_data()

        if save_figure:
            fig_name = "learned"
            plt.savefig(
                os.path.join("figures", fig_groupname + "_" + fig_name + figtype),
                bbox_inches="tight",
            )
            pass

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=gmm_learner,
            # obstacle_container=[],
            dynamics=avoider_functor,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_resolution,
            ax=ax,
            attractor_position=gmm_learner.attractor_position,
            do_quiver=do_quiver,
            show_ticks=True,
        )
        for it_obs, _ in enumerate(gmm_learner):
            local_attractor = gmm_learner._get_local_attractor(it_obs=it_obs)
            if np.allclose(local_attractor, gmm_learner.attractor_position):
                continue

            ax.scatter(
                local_attractor[0],
                local_attractor[1],
                marker="*",
                color="#5A5A5A",
                # color="black",
                s=150,
                zorder=2,
            )

        plot_obstacle_wall_environment(
            gmm_learner=gmm_learner,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            plot_attractor=False,
            plot_annotate_level=False,
            plot_data=False,
            obs_colors=obs_colors,
        )
        if save_figure:
            fig_name = "enclosed"
            plt.savefig(
                os.path.join("figures", fig_groupname + "_" + fig_name + figtype),
                bbox_inches="tight",
            )
            pass

        if False:
            # Plot 'surface point' and next goal
            fig, ax = plt.subplots(figsize=figsize)
            gmm_learner.plot_obstacle_wall_environment(
                ax=ax, plot_local_attractor_construction=True
            )

    # Point within single-boundary obstacle
    position = np.array([-3.421, 0.8947])
    predict_velocity = gmm_learner.predict(position)
    avoided_velocity = avoider_functor(position)
    assert np.all(avoided_velocity > 0), "Velocity should point up-left."

    # Point withing multiple (3) boundary obstacles
    position = np.array([-2.772, 0.508])
    predict_velocity = gmm_learner.predict(position)
    avoided_velocity = avoider_functor(position)
    assert np.all(avoided_velocity > 0)
    assert avoided_velocity[0] < 1, "Expected to point steep upwards."


def visualize_l_shape(visualize=False, save_figure=False):
    dataname = "2D_Lshape"
    x_lim = [-5.5, -0.5]
    y_lim = [-1.4, 2.8]
    figsize = (4, 3)

    fig_groupname = "multihull_avoidance" + "_" + dataname

    n_gaussian = 3
    np.random.seed(2)

    dataset_name = os.path.join("dataset", dataname + ".mat")

    # GMM-learner is also a multiboundary-container
    gmm_learner = GraphGMM(file_name=dataset_name, n_gaussian=n_gaussian)
    gmm_learner.create_learned_boundary()
    gmm_learner.update_intersection_graph()
    gmm_learner.set_convergence_directions(NonlinearDynamcis=gmm_learner)

    if visualize:
        # gmm_learner.plot_graph_and_gaussians()
        n_resolution = 20
        do_quiver = True

        avoider_functor = lambda x: multihull_attraction(
            x, gmm_learner.predict(x), obstacle_list=gmm_learner
        )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            # obstacle_container=gmm_learner,
            obstacle_container=[],
            dynamics=gmm_learner.predict,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_resolution,
            ax=ax,
            # attractor_position=gmm_learner.attractor_position,
            do_quiver=do_quiver,
            show_ticks=True,
        )

        # Downsample (?)
        data_points = gmm_learner.positions
        it_sample = np.arange(0, data_points.shape[0] - 1, 3, dtype=int)
        data_points = data_points[it_sample, :]

        # Downsample?
        ax.plot(
            data_points[:, 0],
            data_points[:, 1],
            ".",
            # color="#DB4391",
            color="black",
            markersize=4.0,
            label="Initial Data",
        )
        obs_colors = [
            "turquoise",
            "darkorange",
            "red",
            "green",
            "purple",
            "tan",
        ]
        draw_obstacle_patches(
            obstacle_list=gmm_learner,
            ax=ax,
            plot_centers=False,
            # axes_scaling=0.9,
            ellipse_alpha=0.5,
            colors=obs_colors,
        )
        # gmm_learner.plot_vectorfield_and_data()

        if save_figure:
            fig_name = "learned_dynamics"
            plt.savefig(
                os.path.join("figures", fig_groupname + "_" + fig_name + figtype),
                bbox_inches="tight",
            )
            pass

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=gmm_learner,
            # obstacle_container=[],
            dynamics=avoider_functor,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_resolution,
            ax=ax,
            attractor_position=gmm_learner.attractor_position,
            do_quiver=do_quiver,
            show_ticks=True,
        )
        for it_obs, _ in enumerate(gmm_learner):
            local_attractor = gmm_learner._get_local_attractor(it_obs=it_obs)
            if np.allclose(local_attractor, gmm_learner.attractor_position):
                continue

            ax.scatter(
                local_attractor[0],
                local_attractor[1],
                marker="*",
                color="#5A5A5A",
                # color="black",
                s=150,
                zorder=2,
            )

        plot_obstacle_wall_environment(
            gmm_learner=gmm_learner,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            plot_attractor=False,
            plot_annotate_level=False,
            plot_data=False,
            obs_colors=obs_colors,
        )


def general_visualzier():
    plt.close("all")
    plt.ion()  # continue program when showing figures
    save_figure = False
    showing_figures = True

    name = None
    # plt.close("all")
    print("Start script .... \n")

    name = "2D_messy-snake"
    n_gaussian = 17

    # name = "2D_incremental_1"
    # n_gaussian = 5

    # dataset_name = "dataset/2D_Sshape.mat"
    # n_gaussian = 5

    # dataset_name = "dataset/2D_Ashape.mat"
    # n_Gaussian = 6
    n_samples = None
    attractor = None

    if False:
        name = "2D_messy-snake"
        n_gaussian = 17

    elif False:
        name = "2D_incremental_1"
        n_gaussian = 5

    elif False:
        dataset_name = "dataset/2D_Sshape.mat"
        n_gaussian = 5
        # n_samples = 300
        attractor = [-4.3, 0]

    elif True:
        name = "2D_Ashape"
        n_gaussian = 6
        # n_samples = 100

    elif False:
        name = "2D_multi-behavior"
        n_gaussian = 11

    elif False:
        dataset_name = "dataset/3D_Cshape_top.mat"

    if name is not None:
        dataset_name = os.path.join("dataset", name + ".mat")

    if True:  # relearn (debugging only)
        np.random.seed(0)
        MainLearner = GraphGMM(file_name=dataset_name, n_gaussian=n_gaussian)
        # MainLearner = DirectionalGMM()
        # MainLearner.load_data_from_mat(file_name=dataset_name)
        # MainLearner.regress(n_gaussian=n_gaussian)

        # gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)

    # MainLearner.plot_position_data()
    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)

    # MainLearner.create_graph()
    MainLearner.create_graph_from_gaussians()
    MainLearner.plot_graph_and_gaussians()

    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "_gaussian_and_data" + figtype),
            bbox_inches="tight",
        )

    MainLearner.create_learned_boundary()
    MainLearner.plot_obstacle_wall_environment()

    # Visualization
    # MainLearner.plot_gaussians_all_directions()

    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)
    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "_gaussianhulls_and_data" + figtype),
            bbox_inches="tight",
        )

    # MainLearner.plot_vectorfield_and_integration()
    # MainLearner.plot_vectorfield_and_data()
    # if save_figure:
    #     plt.savefig(
    #         os.path.join("figures", name + "_vectorfield" + figtype),
    #         bbox_inches="tight",
    #     )

    # MainLearner.plot_time_direction_and_gaussians()
    # MainLearner.plot_vector_field_weights(
    #     n_grid=100, colorlist=gauss_colors, pos_vel_input=True
    # )
    # MainLearner.plot_vector_field_weights(n_grid=100, colorlist=gauss_colors)
    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "weights" + figtype), bbox_inches="tight"
        )

    plt.show()


if (__name__) == "__main__":
    plt.close("all")
    figtype = ".pdf"
    # figtype = ".png"

    # test_inverted_simple_ellipse_comparison()

    _test_visualization_a_shape(visualize=True, save_figure=True)
    # visualize_l_shape(visualize=True, save_figure=False)

    # general_visualzier()

    print("\n\n\n... script finished.")
