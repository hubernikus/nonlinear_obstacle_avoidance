#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""
__author__ = "lukashuber"
__date__ = "2021-05-16"

import os

import matplotlib.pyplot as plt

# from nonlinear_avoidance.gmm_learner.learner.directional import DirectionalGMM
from nonlinear_avoidance.gmm_learner.learner.directional_gmm import DirectionalGMM
from nonlinear_avoidance.gmm_learner.graph import GraphGMM


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()  # continue program when showing figures
    save_figure = True
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
        import numpy as np

        np.random.seed(0)
        MainLearner = GraphGMM()
        # MainLearner = DirectionalGMM()
        MainLearner.load_data_from_mat(file_name=dataset_name)
        MainLearner.regress(n_gaussian=n_gaussian)

        gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)

    MainLearner.plot_position_data()
    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "data" + ".png"), bbox_inches="tight"
        )

    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)
    MainLearner.create_graph()
    MainLearner.plot_graph_and_gaussians()
    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "graph_gaussian" + ".png"),
            bbox_inches="tight",
        )

    MainLearner.create_learned_boundary()
    MainLearner.plot_obstacle_wall_environment()
    if save_figure:
        plt.savefig(
            os.path.join("figures", name + "wall" + ".png"), bbox_inches="tight"
        )

    plt.show()

print("\n\n\n... script finished.")
