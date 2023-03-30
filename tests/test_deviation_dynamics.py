#!/USSR/bin/python3.10
""" Test directional orientation system. """
# Author: Lukas Huber
# Created: 2022-11-27
# Github: hubernikus
# License: M (c) 2022

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import CircularStable
from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence

from nonlinear_avoidance.datatypes import Vector
from nonlinear_avoidance.dynamics.deviation_dynamics import (
    ObstacleRotatedDynamics,
)


def _test_local_rotation(visualize=False):
    obstacle_list = ObstacleContainer()  #
    obstacle_list.append(
        Ellipse(
            center_position=np.array([5, 4]),
            axes_length=np.array([4, 6]),
            orientation=90 * math.pi / 180.0,
        )
    )

    circular_ds = CircularStable(radius=10, maximum_velocity=1)

    rotated_ds = ObstacleRotatedDynamics(
        obstacle_container=obstacle_list,
        initial_dynamics=circular_ds,
    )

    if visualize:
        plt.close("all")

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=rotated_ds.evaluate,
            # dynamics=circular_ds.evaluate,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_grid=30,
            ax=ax,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=0.6,
        )

    # TODO: add assert (!)
    position = np.array([10, -10])
    velocity = rotated_ds.evaluate(position)

    position = np.array([5, 5])
    velocity = rotated_ds.evaluate(position)

    position = np.array([8.5, 7.4])
    velocity = rotated_ds.evaluate(position)

    position = np.array([-9.91, 10.86])
    velocity = rotated_ds.evaluate(position)
    # breakpoint()

    position = np.array([1, 0])
    velocity = rotated_ds.evaluate(position)
    #
    # Position
    position = np.array([0, 0])
    velocity = rotated_ds.evaluate(position)


if (__name__) == "__main__":
    figtype = ".png"

    # _test_local_rotation(visualize=False)
    _test_local_rotation(visualize=True)
    print("Tests done")
