"""
Directional [SEDS] Learning
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import os

import matplotlib.pyplot as plt

# import matplotlib.gridspec as gridspec

from vartools.dynamical_systems import LinearSystem

# from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)
from dynamic_obstacle_avoidance.visualization.gamma_field_visualization import (
    gamma_field_multihull,
)

# from nonlinear_avoidance.gmm_learner.learner.directional import DirectionalGMM
from nonlinear_avoidance.gmm_learner.learner.directional_gmm import DirectionalGMM
from nonlinear_avoidance.gmm_learner.graph import GraphGMM

# nonlinear_avoidance.gmm_learner/visualization/
from nonlinear_avoidance.gmm_learner.visualization.convergence_direction import (
    test_convergence_direction_multihull,
)


def main():
    plt.close("all")
    plt.ion()  # continue program when showing figures
    save_figure = True
    showing_figures = True
    name = None

    print("Start script .... \n")

    if True:
        name = "2D_Ashape"
        n_gaussian = 6

    if name is not None:
        dataset_name = os.path.join("dataset", name + ".mat")

    if True:  # relearn (debugging only)
        import numpy as np

        np.random.seed(4)
        MainLearner = GraphGMM(file_name=dataset_name, n_gaussian=n_gaussian)
        # MainLearner = DirectionalGMM()
        # MainLearner.load_data_from_mat(file_name=dataset_name)
        # MainLearner.regress(n_gaussian=n_gaussian)
        # gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)

    # MainLearner.plot_position_data()
    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)
    MainLearner.create_learned_boundary()

    # Is now included in learned boundary.
    # MainLearner.create_graph_from_gaussians()

    if False:
        MainLearner.plot_obstacle_wall_environment()
        # plt.savefig(os.path.join("figures", name+"_convergence_direction" + ".png"), bbox_inches="tight")

    # pos_attractor = MainLearner.pos_attractor

    MainLearner.set_convergence_directions(NonlinearDynamcis=MainLearner)

    x_lim, y_lim = MainLearner.get_xy_lim_plot()

    plot_gamma_value = False
    if plot_gamma_value:
        n_subplots = 6
        n_cols = 3
        n_rows = int(n_subplots / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

        for it_obs in range(n_subplots):
            it_x = it_obs % n_rows
            it_y = int(it_obs / n_rows)
            ax = axs[it_x, it_y]

            gamma_field_multihull(
                MainLearner, it_obs, n_resolution=100, x_lim=x_lim, y_lim=y_lim, ax=ax
            )

        plt.subplots_adjust(wspace=0.001, hspace=0.001)
        if save_figure:
            plt.savefig(
                os.path.join("figures", "gamma_value_subplots" + ".png"),
                bbox_inches="tight",
            )

    plot_local_attractor = True
    if plot_local_attractor:
        n_subplots = 6
        n_cols = 3
        n_rows = int(n_subplots / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

        for it_obs in range(n_subplots):
            # for it_obs in [1]:
            it_x = it_obs % n_rows
            it_y = int(it_obs / n_rows)
            ax = axs[it_x, it_y]

            # fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            test_convergence_direction_multihull(
                MainLearner,
                it_obs,
                n_resolution=30,
                x_lim=x_lim,
                y_lim=y_lim,
                ax=ax,
                assert_check=False,
            )

        plt.subplots_adjust(wspace=0.001, hspace=0.001)
        # if save_figure:
        if True:
            plt.savefig(
                os.path.join("figures", "test_convergence_direction" + ".png"),
                bbox_inches="tight",
            )

    plot_vectorfield = True
    if plot_vectorfield:
        n_resolution = 30

        # def initial_ds(position):
        # return evaluate_linear_dynamical_system(position, center_position=pos_attractor)

        initial_dynamics = LinearSystem(
            attractor_position=MainLearner.attractor_position
        )

        rotation_container = RotationContainer(MainLearner._obstacle_list)

        rotation_avoider = RotationalAvoider(
            obstacle_environment=rotation_container,
            # convergence_radius=math.pi / 2.0,
            initial_dynamics=initial_dynamics,
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        plot_obstacle_dynamics(
            obstacle_container=rotation_container,
            dynamics=rotation_avoider.avoid,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_resolution,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=False,
            show_ticks=True,
        )

        plot_obstacles(
            obstacle_container=rotation_container,
            ax=ax,
            alpha_obstacle=1.0,
            x_lim=x_lim,
            y_lim=y_lim,
        )
        # Simulation_vectorFields(
        #     x_lim,
        #     y_lim,
        #     n_resolution,
        #     # obs=obstacle_list,
        #     # point_grid=3,
        #     obs=MainLearner,
        #     saveFigure=True,
        #     figName=name + "_converging_linear_base",
        #     noTicks=True,
        #     showLabel=False,
        #     draw_vectorField=True,
        #     dynamical_system=InitialSystem.evaluate,
        #     obs_avoidance_func=rotation_avoider.evaluate,
        #     automatic_reference_point=False,
        #     pos_attractor=InitialSystem.attractor_position,
        #     fig_and_ax_handle=(fig, ax),
        #     # Quiver or Streamplot
        #     show_streamplot=False,
        #     # show_streamplot=False,
        # )

        # if True:
        # MainLearner.set_convergence_directions(NonlinearDynamcis=MainLearner)
        MainLearner.reset_relative_references()

    plt.show()


if (__name__) == "__main__":
    main()
    print("\n\n\n... script finished.")
