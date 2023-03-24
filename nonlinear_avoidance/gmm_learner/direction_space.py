#!/usr/bin/python3
"""
Tools to handle the direction space.
"""

__author__ = "lukashuber"
__date__ = "2021-05-16"

import sys
import warnings
import random

import numpy as np
from math import pi
import scipy.io  # import *.mat files -- MATLAB files

# Machine learning datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import mixture

# Quadratic programming
# from cvxopt.solvers import coneqp

from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse


def guassian_function(pos, gmm_mu, dir_mean, width=1, dx_pow=1, c_dx=1):
    dim = pos.shape[0]
    n_samples = pos.shape[1]

    dir_ref = pos - np.tile(gmm_mu, (n_samples, 1)).T

    # TODO make width and pow based on position
    delta_x = np.linalg.norm(dir_ref, axis=0)

    dir_ref = pos - np.tile(gmm_mu, (n_samples, 1)).T

    if width <= 0:
        warnings.warn("Bad width.")
        width = 1

    # deltaKappa_dir, orthonormal_matrix = get_angle_space(pos, dir_ref, -pos)
    deltaKappa_dir = deltaKappa_dir / np.linalg.norm(deltaKappa_dir, axis=0)
    vel_init = -pos
    deltaKappa_dir_new = velocity_reduction(dir_ref, vel_init)

    breakpoint()

    # Function velocity in coordinate system
    dx = c_dx
    dy = -((delta_x / width) ** dx_pow)

    deltaKappa_mag = np.arctan(dy / dx)

    # return in kappa-space
    return deltaKappa_mag * deltaKappa_dir


def orthonormal_matrix(v):
    """Calculate matrix which is orthogonal to the input vector v."""

    warnings.warn("Outdated use 'direction-space' functions instead.")
    if True:
        raise ValueError()

    v = np.array(v)
    dim = v.shape[0]

    N_samples = v.shape[1]

    V_orth = np.zeros((dim, dim, N_samples))

    v_norm = np.linalg.norm(v, axis=0)
    ind_nonzero = np.nonzero(v_norm)[0]

    V_orth[:, 0, ind_nonzero] = v[:, ind_nonzero] / np.tile(v_norm[ind_nonzero], (2, 1))

    for i in range(1, dim):
        for j in range(0, dim - i):
            V_orth[j, i, ind_nonzero] = (
                V_orth[j, 0, ind_nonzero] * V_orth[dim - i, 0, ind_nonzero]
            )
        V_orth[dim - i, i, ind_nonzero] = (-1) * np.sum(
            V_orth[: (dim - i), 0, ind_nonzero] ** 2, axis=0
        )

        normV = np.linalg.norm(V_orth[:, i, ind_nonzero], axis=0)
        # if normV: # nonzero valuee
        V_orth[:, i, ind_nonzero] /= normV
        # else:
        # warnings.warn('ORTHONORMAL MATRIX HAS NOT FULL RANK.')

    # breakpoint()
    return V_orth, v_norm


def velocity_reduction(x, xd, pos_attractor=[], xd_init=[]):
    warnings.warn("Outdated use 'direction-space' functions instead.")
    if True:
        raise ValueError()
    # TODO -- add velocity Twist (range should be larger than +-pi!!!)
    # x: dim x N
    # xd: dim x N

    x = np.array(x)
    xd = np.array(xd)

    dim = x.shape[0]
    n_samples = x.shape[1]

    if np.array(pos_attractor).shape[0]:  # nonzero
        x = x - np.tile(pos_attractor, (n_samples, 1)).T

    # TODO ds_rf
    if not np.array(xd_init).shape[0]:  # zero-value
        xd_init = -x

    # Create orthogonal matrix
    basisOrth_xdInit, xd_init_mag = orthonormal_matrix(xd_init)

    xd_mag = np.linalg.norm(xd, axis=0)
    it_nonzero = np.nonzero(xd_mag)[0]
    norm_xd = xd
    norm_xd[:, it_nonzero] = norm_xd[:, it_nonzero] / np.tile(
        xd_mag[it_nonzero], (2, 1)
    )

    # Rf = Ff.T # ?! True???
    basisOrth_xdInit_inv = basisOrth_xdInit.swapaxes(0, 1)  # equivalent to transpose

    k_ds = np.zeros((dim - 1, n_samples))

    # norm_xd_hat = Rf @ norm_xd # TODO - check matrix multiplication
    norm_xd_hat = np.sum(
        basisOrth_xdInit_inv * np.tile(norm_xd, (dim, 1, 1)), axis=1
    )  # manual matrix multiplication

    kappa_xd = norm_xd_hat[1:, :]
    kappa_xd_mag = np.linalg.norm(kappa_xd, axis=0)  # Normalize
    ind_nonzero = np.nonzero(kappa_xd_mag)[0]

    kappa_xd[:, ind_nonzero] = kappa_xd[:, ind_nonzero] / np.tile(
        kappa_xd_mag[ind_nonzero], ((dim - 1), 1)
    )

    kappa_xd = np.tile(np.arccos(norm_xd_hat[0, :]), ((dim - 1), 1)) * kappa_xd
    # breakpoint()
    return kappa_xd, basisOrth_xdInit


def velocity_reconstruction(x, kappa, ds_ref="TODO"):
    warnings.warn("Outdated use 'direction-space' functions instead.")
    if True:
        raise ValueError()

    x = np.array(x)
    dim = x.shape[0]
    n_samples = x.shape[1]

    xd_init = -x
    # Create orthogonal matrix
    basisOrth_xdInit, xd_init_mag = orthonormal_matrix(xd_init)

    kappa_mag = np.linalg.norm(kappa, axis=1)
    index_nonzero = np.nonzero(kappa_mag)

    norm_xd = np.hstack(
        (
            np.array([np.cos(kappa_mag)]).T,
            np.tile(np.sin(kappa_mag) / kappa_mag, (kappa.shape[1], 1)).T * kappa,
        )
    )

    norm_xd = np.sum(
        basisOrth_xdInit * np.tile(norm_xd.T, (dim, 1, 1)), axis=1
    )  # manual matrix multiplication
    # breakpoint()
    return norm_xd


def get_mean_yx(X, gmm, dims_input):
    n_gaussian = gmm.covariances_.shape[0]
    dim = gmm.covariances_[0].shape[0]
    dim_in = np.array(dims_input).shape[0]

    n_samples = X.shape[0]
    dims_output = [gg for gg in range(dim) if gg not in dims_input]

    mu_yx = np.zeros((dim - dim_in, n_samples, n_gaussian))
    mu_yx_test = np.zeros((dim - dim_in, n_samples, n_gaussian))

    for gg in range(n_gaussian):
        for nn in range(n_samples):  # TODO #speed - batch process!!

            # mu_yx[:, :, gg] = gmm.means_[gg,dims_output] + gmm.covariances_[gg][dims_output,:][:,dims_input] @ np.linalg.pinv(gmm.covariances_[gg][dims_input,:][:,dims_input]) @ (X - np.tile(gmm.means_[gg,dims_input], (n_samples,1) ) )
            mu_yx[:, nn, gg] = gmm.means_[gg, dims_output] + gmm.covariances_[gg][
                dims_output, :
            ][:, dims_input] @ np.linalg.pinv(
                gmm.covariances_[gg][dims_input, :][:, dims_input]
            ) @ (
                X[nn, :] - gmm.means_[gg, dims_input]
            )
        # dX = X[:,:] - np.tile(gmm.means_[gg,dims_input],(n_samples,1) )
        # muXX_times_dX = np.sum(np.tile(np.linalg.pinv(gmm.covariances_[gg][dims_input,:][:,dims_input]),(n_samples,1,1)).swapaxes(0,1) * np.tile(dX, (dim_in,1,1)),axis=0)

        # mu_yx_test[:, :, gg] = (np.tile(gmm.means_[gg,dims_output], (n_samples,1)) + np.sum(np.tile(gmm.covariances_[gg][dims_output,:][:,dims_input],(n_samples,1,1) ).swapaxes(0,1) * np.tile(muXX_times_dX, (dim-dim_in,1,1)), axis=0 ) ).T

    # if np.sum(mu_yx != mu_yx_test):
    # print('Wanring: this matrix mult does not look good !')
    # else:
    # print('Wel done braaa!')

    return mu_yx


def get_mixing_weights(X, gmm, dims_input):
    dim = X.shape[1]
    n_samples = X.shape[0]
    n_gaussian = gmm.covariances_.shape[0]

    prob_gaussian = get_gaussianProbability(X, gmm, dims_input)

    sum_probGaussian = np.sum(prob_gaussian, axis=0)

    alpha_times_prob = np.tile(gmm.weights_, (n_samples, 1)).T * prob_gaussian

    beta = alpha_times_prob / np.tile(np.sum(alpha_times_prob, axis=0), (n_gaussian, 1))

    return beta


def regress_gmm(
    X,
    gmm,
    dims_input,
    mu=[],
    var=[],
    convergence_attractor=True,
    attractor=[],
    p_beta=2,
    beta_min=0.5,
    beta_r=0.3,
):
    warnings.warn("This function is now included in the hypahypa class.")
    dim = gmm.covariances_[0].shape[1]
    dim_in = np.array(dims_input).shape[0]
    n_samples = X.shape[0]
    n_gaussian = gmm.covariances_.shape[0]

    dims_output = [gg for gg in range(dim) if gg not in dims_input]

    if np.array(mu).shape[0]:
        X = X - np.tile(mu[dims_input], (n_samples, 1))

    if np.array(var).shape[0]:
        X = X / np.tile(var[dims_input], (n_samples, 1))

    beta = get_mixing_weights(X, gmm, dims_input)
    mu_yx = get_mean_yx(X, gmm, dims_input)

    if convergence_attractor:
        if np.array(np.array(attractor).shape[0]):  # zero attractor
            dist_attr = np.linalg.norm(X - np.tile(attractor, (n_samples, 1)), axis=1)
        else:
            dist_attr = np.linalg.norm(X, axis=1)

        beta = np.vstack((beta, np.zeros(n_samples)))

        # Zero values
        beta[:, dist_attr == 0] = 0
        beta[-1, dist_attr == 0] = 1

        # Nonzeros values
        beta[-1, dist_attr != 0] = (dist_attr[dist_attr != 0] / beta_r) ** (
            -p_beta
        ) + beta_min
        beta[:, dist_attr != 0] = beta[:, dist_attr != 0] / np.tile(
            np.linalg.norm(beta[:, dist_attr != 0], axis=0), (n_gaussian + 1, 1)
        )

        mu_yx = np.dstack((mu_yx, np.zeros((dim - dim_in, n_samples, 1))))

    regression_value = np.sum(np.tile(beta.T, (dim - dim_in, 1, 1)) * mu_yx, axis=2).T

    if np.array(var).shape[0]:
        regression_value = regression_value * np.tile(var[dims_output], (n_samples, 1))
    if np.array(mu).shape[0]:
        regression_value = regression_value + np.tile(mu[dims_output], (n_samples, 1))

    return regression_value


def get_gaussianProbability(X, dpgmm, dims_input=[]):
    dim = X.shape[1]
    n_samples = X.shape[0]
    n_gaussian = dpgmm.covariances_.shape[0]

    if not np.array(dims_input).shape[0]:
        dims_input = np.arange(dim)

    # Calculate weight (GAUSSIAN ML)
    prob_gauss = np.zeros((n_gaussian, n_samples))

    for gg in range(n_gaussian):
        # Create function of this
        cov_matrix = dpgmm.covariances_[gg, :, :][dims_input, :][:, dims_input]
        fac = 1 / ((2 * pi) ** (dim * 0.5) * (np.linalg.det(cov_matrix)) ** (0.5))

        dX = X - np.tile(dpgmm.means_[gg, dims_input], (n_samples, 1))

        pow = np.sum(
            np.tile(np.linalg.pinv(cov_matrix), (n_samples, 1, 1))
            * np.swapaxes(np.tile(dX, (dim, 1, 1)), 0, 1),
            axis=2,
        )
        pow = np.exp(-np.sum(dX * pow, axis=1))

        prob_gauss[gg, :] = fac * pow
        # gg[0,:] = fac*pow
    return prob_gauss
