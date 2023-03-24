#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ = "lukashuber"
__date__ = "2021-05-16"

import sys
import os
import warnings

# from functools import lru_cache

from math import pi
import numpy as np

import scipy.io  # import *.mat files -- MATLAB files

# Machine learning datasets
# from sklearn.mixture import BaseMixture
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import mixture

from nonlinear_avoidance.gmm_learner.math_tools import rk4, mag_linear_maximum
from nonlinear_avoidance.gmm_learner.learner.base import Learner
from nonlinear_avoidance.gmm_learner.learner.visualizer import LearnerVisualizer

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system


"""
General Notes for understanding:

fit / fit_predict are the main calls for this

the optimization loop (~ 100 iterations)
does the
1. _e_step (estimation step) uses _estimate_log_prob_resp to
 Calculates: Mean of the logarithms of the probabilities norm (of each sample X)
 log_responsibility (Logarithm of the posterior probabilities)
 
1.1 [] = _estimate_log_prob_resp
Small calculations for log_responsibilities

1.1.1 [weighted_log_prob] = _estimate_weighted_log_prob() 
return self._estimate_log_prob(X) + self._estimate_log_weights()

1.1.1.1 _estimate_log_prob [Gaussian]
gaussian prob + 'log_lambda' - (feature / precision)

1.1.1.1.1 _estimate_log_gaussian_prob
Various calculation to Estimate the log Gaussian probability
1.1.1.1.1.1 _compute_log_det_cholesky (logarithm type)

1.1.1.2 _estimate_log_weights
weights as type dirichlet_process (or other)

2. _m_step (maximaztion step)
2.1 _estimate_gaussian_parameters
Cacluate covariance paramteres
2.2 _estimate_weights
various params
2.3 _estimate_means
various params
2.4 _estimate_precisions
various params

"""


class DirectionalGaussian(BayesianGaussianMixture):
    """
    Important things
    """

    def __init__(self):
        pass

    def import_data(self, file_name):
        pass

    def fit(self, X, y=None):
        # PART OF _base class
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.
        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                ######################################################## The main steps
                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                ########################################################
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X):
        """E step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """M step.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            digamma_sum = digamma(
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (
                digamma_a
                - digamma_sum
                + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
            )
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            return digamma(self.weight_concentration_) - digamma(
                np.sum(self.weight_concentration_)
            )

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5
                * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)
