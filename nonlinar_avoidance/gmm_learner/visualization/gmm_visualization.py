#!/usr/bin/python3
"""
Tools to simplify visualization
"""
# Author: Lukas Huber
# Created: 2021-05-16

import sys
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import numpy.linalg as LA

import matplotlib as mpl

# import matplotlib.pyplot as plt


def draw_obstacle_patches(
    obstacle_list,
    ax,
    plot_dims: npt.ArrayLike = (0, 1),
    colors: Optional[list[str]] = None,
    plot_centers: bool = True,
    ellipse_alpha: float = 0.5,
    axes_scaling: float = 1.0,
):
    for ii, obs in enumerate(obstacle_list):
        path_axes = obs.axes_length * 2 * axes_scaling
        ell = mpl.patches.Ellipse(
            obs.center_position,
            path_axes[0],
            path_axes[1],
            obs.orientation * 180 / np.pi,
            color=colors[ii],
            zorder=-2,
        )

        ell.set_clip_box(ax.bbox)
        ell.set_alpha(ellipse_alpha)
        ax.add_artist(ell)

        if plot_centers:
            ax.plot(
                gmm.means_[n, plot_dims[0]],
                gmm.means_[n, plot_dims[1]],
                "k.",
                markersize=12,
                linewidth=30,
            )
        # ax.plot(gmm.means_[n, 0], gmm.means_[n, 1], 'k+', s=12)


def draw_gaussians(
    gmm,
    ax,
    plot_dims: npt.ArrayLike = (0, 1),
    colors: Optional[list[str]] = None,
    plot_centers: bool = True,
    ellipse_alpha: float = 0.5,
):
    if colors is None:
        colors = [
            "navy",
            "turquoise",
            "darkorange",
            "blue",
            "green",
            "purple",
            "black",
            "violet",
            "tan",
            "red",
        ]

    if isinstance(colors, np.ndarray):
        color_list = []
        for ii in range(colors.shape[1]):
            color_list.append(colors[:, ii])
        colors = color_list

    n_gaussian = gmm.n_components

    for n in range(n_gaussian):
        color = colors[np.mod(n, len(colors))]
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][plot_dims, :][:, plot_dims]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[plot_dims, :][:, plot_dims]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][plot_dims])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

        ell = mpl.patches.Ellipse(
            gmm.means_[n, plot_dims],
            v[0],
            v[1],
            180 + angle,
            color=color,
            zorder=-2,
        )

        ell.set_clip_box(ax.bbox)
        ell.set_alpha(ellipse_alpha)
        ax.add_artist(ell)

        if plot_centers:
            ax.plot(
                gmm.means_[n, plot_dims[0]],
                gmm.means_[n, plot_dims[1]],
                "k.",
                markersize=12,
                linewidth=30,
            )
        # ax.plot(gmm.means_[n, 0], gmm.means_[n, 1], 'k+', s=12)
