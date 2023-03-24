#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ = "lukashuber"
__date__ = "2021-05-16"

from abc import ABC, abstractmethod
import numpy as np

from nonlinear_avoidance.gmm_learner.math_tools import rk4

from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array


class Learner(ABC):
    """Virtual class to learn from demonstration / implementation."""

    def __init__(self, null_ds=None):
        """Initialize the virtual base class"""
        if null_ds is None:
            self.null_ds = LinearSystem(dimension=2)
        else:
            self.null_ds = null_ds

    # @abstractmethod
    # def load_data(self):
    # """ Load the data from a specific regressor. """
    # pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the chosen learner onto the daat."""
        pass

    def evaluate(self, position):
        return self.predict(xx=position)

    @abstractmethod
    def _predict(self, xx):
        """Predicts learned DS but returns value in direction space.

        Parameters
        ----------
        xx: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        direction_space_array: single data-point (n_samples, n_output_features)
        """
        pass

    def predict(self, xx, **kwargs):
        """Predict based on learn model and xx being an input matrix
        with single datapoints.

        Parameters
        ----------
        xx: single data point of shape (n_input_features)
        Single n_features-dimensional output data-point.
        **kwargs: Key-word-arguments for the prediction of the regression

        Returns
        -------
        velocity: single data-point (n_samples, n_output_features)
        """
        # print('X', xx)
        # Ensure only velocity
        # dir_angle_space  = self._predict(np.array([xx[:2]]), **kwargs)
        # print('Semi angle space:', dir_angle_space)

        dir_angle_space = self._predict(np.array([xx]), **kwargs)
        # print('Full angle space:', dir_angle_space)

        dir_angle_space = dir_angle_space[0, : self.dim - 1]

        null_direction = self.null_ds.evaluate(xx[: self.dim_space])
        velocity = get_angle_space_inverse(
            dir_angle_space=dir_angle_space, null_direction=null_direction
        )

        return velocity

    def predict_array(self, xx, **kwargs):
        """Predict based on learn model and xx being an input matrix
        with multiple datapoints.

        Parameters
        ----------
        xx: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.
        **kwargs: Key-word-arguments for the prediction of the regression

        Returns
        -------
        velocities: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        directions_angle_space = self._predict(xx.T, **kwargs)

        if len(directions_angle_space.shape) == 1:
            directions_angle_space = directions_angle_space.reshape(-1, 1)

        # directions_angle_space = np.zeros(directions_angle_space.shape)
        velocities = get_angle_space_inverse_of_array(
            vecs_angle_space=directions_angle_space.T,
            positions=xx,
            # func_vel_default=self.null_ds,
            default_system=self.null_ds,
        )

        return velocities

    def integrate_trajectory(
        self,
        n_steps=400,
        delta_t=0.02,
        convergence_margin=0.01,
        n_trajectories=3,
        starting_points=None,
    ):
        """
        Integrates trajectories based on the learn dynamical system and the starting_points
        (or takes starting points from the data-recording).

        Parameters
        ----------
        n_steps: Maximum number of steps of (int)
        delta_t: Time step of simulation (float).
        convergence_margin: Margin at which trajectory is stopped(float)
        n_trajectories: Number of trajectories (int)
        starting_points: List of starting points

        Returns
        -------
        trajectory_list: List of the simulated trajectories
        """
        if starting_points is None:
            trajectory_list = np.zeros((self.dim, n_steps, n_trajectories))
            for ii in range(n_trajectories):
                trajectory_list[:, 0, ii] = self.dataset["data"][0, ii][:2, 0]
        else:
            n_trajectories = starting_points.shape[1]
            trajectory_list = np.zeros((self.dim, n_steps, n_trajectories))

        for ii in range(n_trajectories):
            for nn in range(1, n_steps):
                trajectory_list[:, nn, ii] = rk4(
                    delta_t, trajectory_list[:, nn - 1, ii], self.predict
                )

                if np.linalg.norm(trajectory_list[:, nn, ii]) < convergence_margin:
                    print(f"Converged after {nn} iterations.")
                    break

        return trajectory_list
