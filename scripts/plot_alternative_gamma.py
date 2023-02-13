import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt


def gamma_function(rel_dist, d_max=7.0, power_factor=1.0 / 1.0):
    if (dist_norm := LA.norm(rel_dist)) >= d_max:
        # Large value
        return 1.0e2

    gamma = (d_max / (d_max - dist_norm)) ** power_factor
    print("gamma", gamma)
    return gamma


def visualize_network(n_resolution=100):
    x1_lim = [-10, 10]
    x2_lim = [-10, 10]

    nx = n_resolution
    ny = n_resolution
    xx, yy = np.meshgrid(
        np.linspace(x1_lim[0], x1_lim[1], nx), np.linspace(x2_lim[0], x2_lim[1], ny)
    )

    input_values = np.vstack((xx.reshape(1, -1), yy.reshape(1, -1)))
    output = np.zeros(input_values.shape[1])
    for ii in range(input_values.shape[1]):
        output[ii] = gamma_function(input_values[:, ii])

    fig, ax = plt.subplots(constrained_layout=True)
    cs = ax.contourf(
        xx.reshape(nx, ny),
        yy.reshape(nx, ny),
        output.reshape(nx, ny),
        levels=np.linspace(0, 20, 21),
        extend="max",
        # cmap=plt.cm.bone,
        # origin=origin,
    )
    cbar = fig.colorbar(cs)


if (__name__) == "__main__":
    plt.close("all")

    plt.ion()
    visualize_network()
