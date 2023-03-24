#!/usr/bin/python3
"""
Mathematical tools to simplify the implementation
"""
__author__ = "lukashuber"
__date__ = "2021-05-16"

import sys
import warnings
import random

import numpy as np


def mag_linear_maximum(x, k=1, maxMag=1.0):
    magnitude = np.linalg.norm(x, axis=0) * k
    max_indeces = magnitude > maxMag
    if np.sum(max_indeces.shape[0]):  # nonzero
        magnitude[max_indeces] = maxMag * np.ones(np.sum(max_indeces))

    return magnitude


def rk4_pos_vel(dt, pos0, vel0, ds, k_f=1):
    # TODO: do better..
    # k1
    pos = pos0
    pos_vel = np.hstack((pos, vel0))
    xd = ds(pos_vel) * k_f
    k1 = dt * xd

    # k2
    pos = pos0 + 0.5 * k1
    # vel = 0.5*vel0 + 0.5*xd
    pos_vel = np.hstack((pos, vel0))
    xd = ds(pos_vel) * k_f
    k2 = dt * xd

    # k3
    pos = pos0 + 0.5 * k2
    # vel = 0.5*vel0 + 0.5*xd
    pos_vel = np.hstack((pos, vel0))
    xd = ds(pos_vel) * k_f
    k3 = dt * xd

    # k4
    pos = pos0 + k3
    # vel = 0.5*vel0 + 0.5*xd
    pos_vel = np.hstack((pos, vel0))
    xd = ds(pos_vel) * k_f
    k4 = dt * xd

    # x final
    xd = 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x = pos0 + xd  # + O(dt^5)

    return x, xd


def rk4(dt, x, ds, x0=[0, 0], k_f=1):
    """Returns the final position approached in a runge-kutta style."""
    x0 = np.array((x0))
    # k1
    xd = ds(x) * k_f
    k1 = dt * xd

    # k2
    xd = ds(x + 0.5 * k1) * k_f
    k2 = dt * xd

    # k3
    xd = ds(x + 0.5 * k2) * k_f
    k3 = dt * xd

    # k4
    xd = ds(x + k3) * k_f
    k4 = dt * xd

    # x final
    xd = 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x = x + xd  # + O(dt^5)

    return x
