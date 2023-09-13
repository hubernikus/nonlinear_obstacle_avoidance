#!/USSR/bin/python3.10
""" Test / visualization of line following. """
# Author: Lukas Huber
# Created: 2022-11-25
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2025


import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.states import Pose

from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics


def test_simple_circular(visualize=False):
    dynamics = SimpleCircularDynamics(pose=Pose.create_trivial(2), radius=2.0)

    if visualize:
        x_lim = [-4, 4]
        y_lim = [-4, 4]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            plottype="quiver",
            n_resolution=40,
        )

    # Pointing outwards close to obstacle
    position = np.array([0.1, 0.0])
    velocity = dynamics.evaluate(position)

    norm_vel = velocity / np.linalg.norm(velocity)
    norm_pos = position / np.linalg.norm(position)
    assert np.dot(norm_vel, norm_pos) > 0.5


if (__name__) == "__main__":
    import matplotlib.pyplot as plt  # For debugging only (!)
    from vartools.dynamical_systems import plot_dynamical_system

    figtype = ".pdf"

    test_simple_circular(visualize=True)

    print("Tests done")
