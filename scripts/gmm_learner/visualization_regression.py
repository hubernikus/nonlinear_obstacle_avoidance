9  #!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ = "lukashuber"
__date__ = "2021-05-16"

import matplotlib.pyplot as plt

# from nonlinear_avoidance.gmm_learner.learner.directional import DirectionalGMM
from nonlinear_avoidance.gmm_learner.learner.gpr_directional import DirectionalGPR


def plot_simple_regression():
    plt.close("all")
    plt.ion()

    n_samples = None

    # dataset_name = "dataset/2D_messy-snake.mat"
    # n_gaussian = 17

    # dataset_name = "dataset/2D_incremental_1.mat"
    # n_gaussian = 5

    # dataset_name = "dataset/2D_Sshape.mat"
    # n_gaussian = 5
    # n_samples = 300

    dataset_name = "dataset/2D_Ashape.mat"
    n_Gaussian = 6
    n_samples = 100

    MainLearner = DirectionalGPR()
    MainLearner.load_data_from_mat(file_name=dataset_name, n_samples=n_samples)
    MainLearner.fit()

    # gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)
    # MainLearner.plot_time_direction_and_gaussians()
    MainLearner.plot_vectorfield_and_integration(n_grid=100)
    # MainLearner.plot_vectorfield_and_data(n_grid=100)
    plt.show()


if (__name__) == "__main__":
    plot_simple_regression()


print("\n\n\n... script finished.")
