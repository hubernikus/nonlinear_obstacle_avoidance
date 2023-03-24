#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ = "lukashuber"
__date__ = "2021-05-16"


import sys
import os
import warnings
import time

from functools import lru_cache

from math import pi
import numpy as np

import scipy.io  # import *.mat files -- MATLAB files

# Machine learning datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared

# Custom libraries
from nonlinear_avoidance.gmm_learner.math_tools import mag_linear_maximum

from nonlinear_avoidance.gmm_learner.learner.base import Learner
from nonlinear_avoidance.gmm_learner.learner.visualizer import LearnerVisualizer

from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array


class DirectionalGPR(LearnerVisualizer, Learner):
    """Virtual class to learn from demonstration / implementation."""

    def __init__(self, directory_name="", file_name=""):
        self.directory_name = directory_name
        self.file_name = file_name

        # where should this ideally be defined?
        self.dim = None

        self.pos = None
        self.vel = None

        self.pos_attractor = None

        # Noramlized etc. regression data
        self.X = None
        self.y = None

        super().__init__()

        self.n_features = 2

    def load_data_from_mat(self, file_name=None, dims_input=None, n_samples=None):
        """Load data from file mat-file & evaluate specific parameters"""
        if file_name is not None:
            self.file_name = file_name

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name)
        )

        self.dim_space = self.dim = 2
        self.dim_output = 1

        if dims_input is None:
            self.dims_input = [0, 1]

        # Only take the first fold.
        ii = 0
        self.pos = self.dataset["data"][0, ii][:2, :].T
        self.vel = self.dataset["data"][0, ii][2:4, :].T

        t = np.linspace(0, 1, self.dataset["data"][0, ii].shape[1])

        pos_attractor = np.zeros((self.dim))

        for it_set in range(1, self.dataset["data"].shape[1]):
            self.pos = np.vstack((self.pos, self.dataset["data"][0, it_set][:2, :].T))
            self.vel = np.vstack((self.vel, self.dataset["data"][0, it_set][2:4, :].T))

            pos_attractor = (
                pos_attractor
                + self.dataset["data"][0, it_set][:2, -1].T
                / self.dataset["data"].shape[1]
            )

            print("pos_attractor", self.dataset["data"][0, it_set][:2, -1].T)

            # TODO include velocity - rectify
            t = np.hstack(
                (t, np.linspace(0, 1, self.dataset["data"][0, it_set].shape[1]))
            )

        self.pos_attractor = pos_attractor

        print("pos attractor", self.pos_attractor)
        if self.pos_attractor is not None:
            # self.null_ds = lambda x: evaluate_linear_dynamical_system(
            #     x, center_position=self.pos_attractor
            # )
            self.null_ds = LinearSystem(attractor_position=self.pos_attractor)

        self.X = self.pos

        # self.X = np.hstack((self.pos, direction.T, np.tile(t, (1, 1)).T))
        n_input = self.X.shape[0]
        print(f"Number of samples imported: {n_input}.")

        weightDir = 4

        # 2D system
        linear_dynamics = LinearSystem(attractor_position=np.zeros(2))
        if n_samples is not None:
            # Large number of samples should be chosen to avoid
            ind = np.random.choice(n_input, size=n_samples, replace=False)
            self.X = self.X[ind, :]

            self.n_samples = n_input

            directions = get_angle_space_of_array(
                directions=self.vel.T[:, ind],
                positions=self.pos.T[:, ind],
                func_vel_default=linear_dynamics.evaluate,
            )
        else:
            directions = get_angle_space_of_array(
                directions=self.vel.T,
                positions=self.pos.T,
                func_vel_default=linear_dynamics.evaluate,
            )

            self.n_samples = n_input

            # Regressor properties
            self._gp_kernel = None
            self._gpr = None

        self.y = np.squeeze(directions)

    def fit(self, kernel_parameters=None, kernel_type=RBF, kernel_noise=1e-1):
        if kernel_parameters is None:
            # TODO: properly...
            length_scale = np.ones(self.n_features)

        self._gp_kernel = RBF(length_scale) + WhiteKernel(kernel_noise)
        self._gpr = GaussianProcessRegressor(kernel=self._gp_kernel)

        stime = time.time()
        self._gpr.fit(self.X, self.y)

        print("Time for GPR fitting: %.3f" % (time.time() - stime))

    def _predict(self, xx):
        """Output compatible with higher-dimensional regression."""
        # return self._gpr.predict(xx)
        return np.reshape(self._gpr.predict(xx), (-1, 1))
