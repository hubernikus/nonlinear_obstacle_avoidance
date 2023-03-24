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

import matplotlib.pyplot as plt

# Machine learning datasets
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline  # [Try to use this one for predction]

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# Custom libraries
from nonlinear_avoidance.gmm_learner.math_tools import mag_linear_maximum

from nonlinear_avoidance.gmm_learner.learner.base import Learner
from nonlinear_avoidance.gmm_learner.learner.visualizer import LearnerVisualizer

from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array


class DirectionalSVR(LearnerVisualizer, Learner):
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

        self._svr = None

    def load_data_from_mat(
        self, file_name=None, dims_input=None, n_samples=None, attractor=None
    ):
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

        for it_set in range(1, self.dataset["data"].shape[1]):
            self.pos = np.vstack((self.pos, self.dataset["data"][0, it_set][:2, :].T))
            self.vel = np.vstack((self.vel, self.dataset["data"][0, it_set][2:4, :].T))

            # TODO include velocity - rectify
            t = np.hstack(
                (t, np.linspace(0, 1, self.dataset["data"][0, it_set].shape[1]))
            )

        if attractor is None:
            pos_attractor = np.zeros((self.dim))

            for it_set in range(1, self.dataset["data"].shape[1]):
                pos_attractor = (
                    pos_attractor
                    + self.dataset["data"][0, it_set][:2, -1].T
                    / self.dataset["data"].shape[1]
                )
                print("pos_attractor", self.dataset["data"][0, it_set][:2, -1].T)

            self.pos_attractor = pos_attractor
            self.null_ds = lambda x: evaluate_linear_dynamical_system(
                x, center_position=self.pos_attractor
            )

        elif attractor is False:
            # Does not have attractor
            self.pos_attractor = False
            self.null_ds = attracting_circle
        else:
            self.pos_attractor = np.array(attractor)

            self.null_ds = lambda x: evaluate_linear_dynamical_system(
                x, center_position=self.pos_attractor
            )

        self.X = self.pos

        # self.X = np.hstack((self.pos, direction.T, np.tile(t, (1, 1)).T))
        n_input = self.X.shape[0]
        print(f"Number of samples imported: {n_input}.")

        weightDir = 4

        if n_samples is not None:
            # Large number of samples should be chosen to avoid
            ind = np.random.choice(n_input, size=n_samples, replace=False)
            self.X = self.X[ind, :]

            self.n_samples = n_input

            directions = get_angle_space_of_array(
                directions=self.vel.T[:, ind],
                positions=self.pos.T[:, ind],
                func_vel_default=self.null_ds,
            )
        else:
            directions = get_angle_space_of_array(
                directions=self.vel.T,
                positions=self.pos.T,
                func_vel_default=self.null_ds,
            )

            self.n_samples = n_input
        self.y = np.squeeze(directions)

    def grid_search(self):
        self._svr = GridSearchCV(
            SVR(kernel="rbf", gamma=0.1),
            param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
        )

        stime = time.time()
        self._svr.fit(self.X, self.y)
        print("Time for SVR fitting: %.3f" % (time.time() - stime))
        print("Best search params:", self._svr.best_params_)

    def fit(self, kernel_type="rbf", C=1.0, gamma=1, grid_search=False):
        if not grid_search:
            # self._svr = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)
            self._svr = SVR(kernel="rbf", C=10, gamma=1, epsilon=0.1)
            stime = time.time()
            self._svr.fit(self.X, self.y)
            print("Time for SVR fitting: %.3f" % (time.time() - stime))
            print("N_support vectors: %.i" % self._svr.support_vectors_.shape[0])

        else:
            self.grid_search()

    def _predict(self, xx):
        """Output compatible with higher-dimensional regression."""
        # return self._gpr.predict(xx)
        return np.reshape(self._svr.predict(xx), (-1, 1))

    def plot_support_vectors_and_data(self, n_grid=100):
        self.plot_vectorfield_and_integration(n_grid=n_grid)

        plt.plot(
            self._svr.support_vectors_[:, 0],
            self._svr.support_vectors_[:, 1],
            "o",
            color="red",
        )
