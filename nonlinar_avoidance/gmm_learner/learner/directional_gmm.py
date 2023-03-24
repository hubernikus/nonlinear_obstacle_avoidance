"""
Directional [SEDS] Learning
"""
__author__ = "lukashuber"
__date__ = "2021-05-16"

import sys
import os
import copy
import warnings
from math import pi

from functools import lru_cache

import numpy as np

import scipy.io  # import *.mat files -- MATLAB files

# Machine learning datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import mixture

# Custom libraries
# from nonlinear_avoidance.gmm_learner.visualization.gmm_visualization import draw_gaussians
from nonlinear_avoidance.gmm_learner.math_tools import (
    rk4,
    rk4_pos_vel,
    mag_linear_maximum,
)
from nonlinear_avoidance.gmm_learner.direction_space import (
    velocity_reduction,
    velocity_reconstruction,
    velocity_reduction,
    get_mixing_weights,
    get_mean_yx,
)
from nonlinear_avoidance.gmm_learner.learner.base import Learner
from nonlinear_avoidance.gmm_learner.learner.visualizer import LearnerVisualizer

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamical_systems import LinearSystem

# from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system


class DirectionalGMM(Learner, LearnerVisualizer):
    """Virtual class to learn from demonstration / implementation.
    The Gaussian mixture regression on the slides found in:
    http://calinon.ch/misc/EE613/EE613-slides-9.pdf
    """

    def __init__(
        self,
        directory_name="",
        file_name="",
        dim_space: int = 2,
        beta_power: float = 5e-2,
    ):
        self.directory_name = directory_name
        self.file_name = file_name

        self.dim_space = self.dim = dim_space
        self.dim_gmm = None

        self.beta_power = beta_power

        # TODO: remove dataset from 'attributes'
        self.dataset = None
        self.pos = None
        self.vel = None

        # Noramlized etc. regression data
        self.X = None

        self.scale_x = None
        self.mean_x = None

        # A cap should only be used for 'input' since it's not a bijective-function
        # Cap is applied after scaling & mean
        self.crop_vel = 1.0

        self.has_path_completion = False

        self.convergence_attractor = False

        super().__init__()

    @property
    def n_gaussians(self):
        # Depreciated -> use 'n_components' instead
        return self.dpgmm.covariances_.shape[0]

    @property
    def n_components(self):
        return self.dpgmm.covariances_.shape[0]

    @property
    def attractor_position(self):
        # Consistent with general dynamical systems
        return self.pos_attractor

    def transform_initial_to_normalized(self, values, dims_ind):
        """Inverse-normalization and return the modified value."""
        n_samples = values.shape[0]

        if self.mean_x is not None:
            values = values - np.tile(self.mean_x[dims_ind], (n_samples, 1))

        if self.scale_x is not None:
            values = values / np.tile(self.scale_x[dims_ind], (n_samples, 1))

        if self.crop_vel is not None:
            velocities = values[self.dim_space : self.dim_space * 2]
            mag_vel = np.linalg.norm(velocities, axis=1)
            ind_large = mag_vel > self.crop_vel

            if np.sum(ind_large):
                values = np.copy(values)
                velocities[ind_large, :] = (
                    velocities[ind_large, :]
                    / np.tile(mag_vel[ind_large], (values.shape[1], 1)).T
                )

                values[self.dim_space : self.dim_space * 2, :] = velocities
        return values

    def transform_normalized_to_initial(self, values, dims_ind):
        """Inverse-normalization and return the modified values."""
        n_samples = values.shape[0]

        if self.scale_x is not None:
            values = values * np.tile(self.scale_x[dims_ind], (n_samples, 1))

        if self.mean_x is not None:
            values = values + np.tile(self.mean_x[dims_ind], (n_samples, 1))

        return values

    def normalize_velocity(self, X, ind_vel=None, crop_vel=1):
        """Normalize the velocity by the mean & cap it at 1.0."""
        if ind_vel is None:
            ind_vel = np.arange(self.dim_space, self.dim_space * 2)

        X = copy.deepcopy(X)
        velocity = X[:, ind_vel]

        vel_mag = np.linalg.norm(velocity, axis=1)

        mean_vel = np.mean(vel_mag)
        if self.crop_vel is not None:
            vel_mag = np.maximum(mean_vel * np.ones(vel_mag.shape) * crop_vel, vel_mag)

        velocity = velocity / np.tile(vel_mag, (ind_vel.shape[0], 1)).T

        # Stretching happening here
        # if self.scale_x is None:
        # self.scale_x = np.ones(X.shape[1])

        # self.scale_x[ind_vel] = mean_vel
        self.scale_vel = mean_vel

        X[:, ind_vel] = velocity

        # Story mean_vel
        return X

    def load_data_from_mat(self, file_name=None, feat_in=None, attractor=None):
        """Load data from file mat-file & evaluate specific parameters"""
        if file_name is not None:
            self.file_name = file_name

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name)
        )

        if feat_in is None:
            self.feat_in = [0, 1]

        ii = 0  # Only take the first fold.
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

        direction = get_angle_space_of_array(
            directions=self.vel.T,
            positions=self.pos.T,
            func_vel_default=LinearSystem(dimension=self.dim_space).evaluate,
        )

        self.X = np.hstack((self.pos, self.vel, direction.T))

        self.X = self.normalize_velocity(self.X)

        self.num_samples = self.X.shape[0]
        self.dim_gmm = self.X.shape[1]

        weightDir = 4

        if attractor is None:
            pos_attractor = np.zeros((self.dim))

            for it_set in range(0, self.dataset["data"].shape[1]):
                pos_attractor = (
                    pos_attractor
                    + self.dataset["data"][0, it_set][:2, -1].T
                    / self.dataset["data"].shape[1]
                )
                print("pos_attractor", self.dataset["data"][0, it_set][:2, -1].T)
            print(pos_attractor)
            self.pos_attractor = pos_attractor
            # self.null_ds = lambda x: evaluate_linear_dynamical_system(
            # x, center_position=self.pos_attractor)
            self.null_ds = LinearSystem(attractor_position=self.pos_attractor)

        elif attractor is False:
            # Does not have attractor
            self.pos_attractor = False
            self.null_ds = attracting_circle
        else:
            self.pos_attractor = np.array(attractor)

            # self.null_ds = lambda x: evaluate_linear_dynamical_system(
            # x, center_position=self.pos_attractor)
            self.null_ds = LinearSystem(attractor_position=self.pos_attractor)

        if self.has_path_completion:
            # TODO
            raise NotImplementedError()

        # Normalize dataset
        normalize_dataset = False
        if normalize_dataset:
            self.meanX = np.mean(self.X, axis=0)

            self.meanX = np.zeros(4)
            # X = X - np.tile(meanX , (X.shape[0],1))
            self.varX = np.var(self.X, axis=0)

            # All distances should have same variance
            self.varX[: self.dim] = np.mean(self.varX[: self.dim])

            # All directions should have same variance
            self.varX[self.dim : 2 * self.dim - 1] = np.mean(
                self.varX[self.dim : 2 * self.dim - 1]
            )

            # Stronger weight on directions!
            self.varX[self.dim : 2 * self.dim - 1] = (
                self.varX[self.dim : 2 * self.dim - 1] * 1 / weightDir
            )

            self.X = self.X / np.tile(self.varX, (self.X.shape[0], 1))

        else:
            self.meanX = None
            self.varX = None

            self.pos_attractor = pos_attractor

    def regress(self, *args, **kwargs):
        # TODO: remove
        return self.fit(*args, **kwargs)

    def fit(self, n_gaussian=5, tt_ratio=0.75):
        """Regress based on the data given."""
        a_label = np.zeros(self.num_samples)
        all_index = np.arange(self.num_samples)

        train_index, test_index = train_test_split(all_index, test_size=(1 - tt_ratio))

        X_train = self.X[train_index, :]
        X_test = self.X[test_index, :]

        y_train = a_label[train_index]
        y_test = a_label[test_index]

        cov_type = "full"

        self.dpgmm = mixture.BayesianGaussianMixture(
            n_components=n_gaussian, covariance_type="full"
        )

        # sample dataset
        reference_dataset = 0
        n_start = 0
        for it_set in range(reference_dataset):
            n_start += self.dataset["data"][0, it_set].shape[1]

        index_sample = [
            int(
                np.round(
                    n_start
                    + self.dataset["data"][0, reference_dataset].shape[1]
                    / n_gaussian
                    * ii
                )
            )
            for ii in range(n_gaussian)
        ]

        self.dpgmm.means_init = self.X[index_sample, :]
        self.dpgmm.means_init = X_train[np.random.choice(np.arange(n_gaussian)), :]

        self.dpgmm.fit(X_train[:, :])

    def regress_gmm(self, *args, **kwargs):
        # TODO: Depreciated; remove..
        return self._predict(*args, **kwargs)

    def _predict(
        self,
        X,
        input_output_normalization=True,
        feat_in=None,
        feat_out=None,
        convergence_attractor: None = False,
        p_beta=2,
        beta_min=0.5,
        beta_r=0.3,
    ):
        """Evaluate the regress field at all the points X"""
        dim = self.dim_gmm
        n_samples = X.shape[0]
        dim_in = X.shape[1]

        if feat_in is None:
            feat_in = np.arange(dim_in)

        if feat_out is None:
            # Default only the 'direction' at the end; additional -1 for indexing at end
            feat_out = self.dim_gmm - 1 - np.arange(self.dim_space - 1)
        dim_out = np.array(feat_out).shape[0]

        if input_output_normalization:
            X = self.transform_initial_to_normalized(X, dims_ind=feat_in)

        # Gausian Mixture Model Properties
        beta = self.get_mixing_weights(X, feat_in=feat_in, feat_out=feat_out)
        mu_yx = self.get_mean_yx(X, feat_in=feat_in, feat_out=feat_out)

        # Covariance computation only
        # covariance_output = self.get_covariance_out(feat_in=feat_in, feat_out=feat_out)
        # estimate_normals = self.get_gaussian_probability(
        # X=X, mean=mu_yx, covariance_matrices=covariance_output)
        # breakpoint()
        # estimate_normals = mu_yx
        if convergence_attractor is not None:
            self.convergence_attractor = convergence_attractor

        if self.convergence_attractor:
            if self.pos_attractor is None:
                raise ValueError("Convergence to attractor without attractor...")

            if dim_in == self.dim_space:
                attractor = self.pos_attractor
            else:
                # Attractor + zero-velocity
                attractor = np.hstack((self.pos_attractor, np.zeros(self.dim_space)))

            dist_attr = np.linalg.norm(X - np.tile(attractor, (n_samples, 1)), axis=1)

            beta = np.vstack((beta, np.zeros(n_samples)))

            # Zero values
            ind_zero = dist_attr == 0
            beta[:, ind_zero] = 0
            beta[-1, ind_zero] = 1

            # Nonzeros values
            ind_nonzero = dist_attr != 0
            beta[-1, ind_nonzero] = (dist_attr[ind_nonzero] / beta_r) ** (
                -p_beta
            ) + beta_min
            beta[:, ind_nonzero] = beta[:, ind_nonzero] / np.tile(
                np.linalg.norm(beta[:, ind_nonzero], axis=0), (self.n_gaussians + 1, 1)
            )

            # Add zero velocity
            mu_yx = np.dstack((mu_yx, np.zeros((dim_out, n_samples, 1))))

        beta_norm = np.sum(beta)
        beta = beta / beta_norm * (beta_norm**self.beta_power)

        regression_value = np.sum(np.tile(beta.T, (dim_out, 1, 1)) * mu_yx, axis=2).T

        if input_output_normalization:
            regression_value = self.transform_normalized_to_initial(
                regression_value, dims_ind=feat_out
            )

        return regression_value

    def get_mixing_weights(
        self,
        X,
        feat_in,
        feat_out,
        input_needs_normalization=False,
        normalize_probability=False,
        weight_factor=4.0,
    ):
        """Get input positions X of the form [dimension, number of samples]."""
        # TODO: try to learn the 'weight_factor' [optimization problem?]
        if input_needs_normalization:
            X = self.transform_initial_to_normalized(X, feat_in)

        n_samples = X.shape[0]
        dim_in = feat_in.shape[0]

        prob_gaussian = self.get_gaussian_probability(X, feat_in=feat_in)
        sum_probGaussian = np.sum(prob_gaussian, axis=0)

        alpha_times_prob = (
            np.tile(self.dpgmm.weights_, (n_samples, 1)).T * prob_gaussian
        )

        if normalize_probability:
            beta = alpha_times_prob / np.tile(
                np.sum(alpha_times_prob, axis=0), (self.n_gaussians, 1)
            )
        else:
            beta = alpha_times_prob
            max_weight = np.max(self.dpgmm.weights_)
            beta = beta / max_weight * weight_factor**dim_in

            sum_beta = np.sum(beta, axis=0)
            ind_large = sum_beta > 1
            beta[:, ind_large] = beta[:, ind_large] / np.tile(
                sum_beta[ind_large], (self.n_gaussians, 1)
            )
        return beta

    def get_gaussian_probability(
        self, X, feat_in=None, covariance_matrices=None, mean=None
    ):
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        prob_gauss (beta): array of shape (n_samples)
            The weights (similar to prior) which is gaussian is assigned.
        """
        if covariance_matrices is None:
            covariance_matrices = self.dpgmm.covariances_[:, feat_in, :][:, :, feat_in]
        if mean is None:
            mean = self.dpgmm.means_[:, feat_in]

        n_samples = X.shape[0]
        dim_in = X.shape[1]
        # dim_in = mean.shape[1]

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((self.n_gaussians, n_samples))

        for gg in range(self.n_gaussians):
            # Create function of this
            covariance = covariance_matrices[gg, :, :]
            try:
                fac = 1 / (
                    (2 * pi) ** (dim_in * 0.5) * (np.linalg.det(covariance)) ** (0.5)
                )
            except:
                breakpoint()
            dX = X - np.tile(mean[gg, :], (n_samples, 1))

            val_pow_fac = np.sum(
                np.tile(np.linalg.pinv(covariance), (n_samples, 1, 1))
                * np.swapaxes(np.tile(dX, (dim_in, 1, 1)), 0, 1),
                axis=2,
            )

            val_pow = np.exp(-np.sum(dX * val_pow_fac, axis=1))
            prob_gauss[gg, :] = fac * val_pow

        return prob_gauss

    def get_covariance_out(self, feat_in, feat_out, stretch_input_values=False):
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        mu_yx: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        dim_out = np.array(feat_out).shape[0]
        covariance_out = np.zeros((dim_out, dim_out, self.n_gaussians))

        for gg in range(self.n_gaussians):
            covariance = self.dpgmm.covariances_[gg, :, :]
            covariance_out[:, :, gg] = (
                covariance[feat_out, :][:, feat_out]
                - covariance[feat_out, :][:, feat_in]
                @ np.linalg.pinv(covariance[feat_in, :][:, feat_in])
                @ covariance[feat_in, :][:, feat_out]
            )
        return covariance_out

    def get_mean_yx(self, X, feat_in, feat_out, stretch_input_values=False):
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        mu_yx: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        dim_out = np.array(feat_out).shape[0]
        n_samples = X.shape[0]

        mu_yx = np.zeros((dim_out, n_samples, self.n_gaussians))
        mu_yx_hat = np.zeros((dim_out, n_samples, self.n_gaussians))

        for gg in range(self.n_gaussians):
            mu_yx[:, :, gg] = np.tile(self.dpgmm.means_[gg, feat_out], (n_samples, 1)).T
            matrix_mult = self.dpgmm.covariances_[gg][feat_out, :][:, feat_in].dot(
                np.linalg.pinv(self.dpgmm.covariances_[gg][feat_in, :][:, feat_in])
            )

            mu_yx[:, :, gg] += matrix_mult.dot(
                (X - np.tile(self.dpgmm.means_[gg, feat_in], (n_samples, 1))).T
            )

            ### START REMOVE ###
            for nn in range(n_samples):  # TODO #speed - batch process!!
                mu_yx_hat[:, nn, gg] = self.dpgmm.means_[
                    gg, feat_out
                ] + self.dpgmm.covariances_[gg][feat_out, :][
                    :, feat_in
                ] @ np.linalg.pinv(
                    self.dpgmm.covariances_[gg][feat_in, :][:, feat_in]
                ) @ (
                    X[nn, :] - self.dpgmm.means_[gg, feat_in]
                )

        if np.sum(mu_yx - mu_yx_hat) > 1e-6:
            breakpoint()
        else:
            # TODO: remove when warning never shows up anymore
            warnings.warn("Remove looped multiplication, since is the same...")
        return mu_yx

    def integrate_trajectory(
        self,
        num_steps=200,
        delta_t=0.05,
        nTraj=3,
        starting_points=None,
        convergence_err=0.01,
        velocity_based=False,
    ):
        """Return integrated trajectory with runge-kutta-4 based on the learned system.
        Default starting points are chosen at the starting points of the learned data"""

        if starting_points is None:
            x_traj = np.zeros((self.dim, num_steps, nTraj))
            for ii in range(nTraj):
                x_traj[:, 0, ii] = self.dataset["data"][0, ii][:2, 0]
        else:
            nTraj = starting_points.shape[1]
            x_traj = np.zeros((self.dim, num_steps, nTraj))

        # Do the first step without velocity
        print("Doint the integration.")
        for ii in range(nTraj):
            # x_traj[:, 1, ii]= rk4(delta_t, x_traj[:, 0, ii], self.predict)
            # xd = x_traj[:, 1, ii] - x_traj[:, 0, ii]

            xd = self.predict2(x_traj[:, 0, ii])
            x_traj[:, 1, ii] = x_traj[:, 0, ii] + xd * delta_t

            for nn in range(2, num_steps):
                # x_traj[:, nn, ii], xd = rk4(
                # delta_t, x_traj[:, nn-1, ii],
                # xd,
                # lambda x: self.predict(np.hstack((x, xd)))
                # )

                # x_traj[:, nn, ii], xd = rk4_pos_vel(
                #     dt=delta_t, pos0=x_traj[:, nn-1, ii],
                #     vel0=xd,
                #     ds=self.predict
                #     )

                if velocity_based:
                    # if True:
                    xd = self.predict(np.hstack((x_traj[:, nn - 1, ii], xd)))

                else:
                    # if True:
                    xd = self.predict(x_traj[:, nn - 1, ii])

                # print('xd', xd)
                x_traj[:, nn, ii] = x_traj[:, nn - 1, ii] + xd * delta_t

                if np.linalg.norm(x_traj[:, nn, ii]) < convergence_err:
                    print(f"Converged after {nn} iterations.")
                    break

        print("This took me a while...")
        return x_traj
