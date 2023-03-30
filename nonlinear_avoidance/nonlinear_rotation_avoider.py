"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2021-09-01

import warnings
import copy
import math
from functools import partial
from typing import Protocol, Optional

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from vartools.math import get_intersection_with_circle
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import (
    get_directional_weighted_sum_from_unit_directions,
)
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import UnitDirection

# from vartools.directional_space DirectionBase
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import AxesFollowingDynamics
from vartools.dynamical_systems import ConstantValue

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.avoidance import BaseAvoider
from nonlinear_avoidance.avoidance import (
    RotationalAvoider,
)
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def get_convergence_weight(gamma: float) -> float:
    if math.isclose(gamma, 0):
        return 1.0
    return min(1 / gamma, 1)


class ObstacleConvergenceDynamics(Protocol):
    def evaluate_convergence_around_obstacle(
        self, position: npt.ArrayLike, obstacle: Obstacle
    ) -> np.ndarray:
        ...

    def get_base_convergence(self, position: npt.ArrayLike) -> np.ndarray:
        ...


class LinearConvergenceDynamics:
    def __init__(
        self, convergence_dynamics: DynamicalSystem, initial_dynamics: DynamicalSystem
    ) -> None:
        self.initial_dynamics = initial_dynamics
        self.convergence_dynamics = convergence_dynamics

    def evaluate_convergence_around_obstacle(
        self, position: npt.ArrayLike, obstacle: Obstacle
    ) -> np.ndarray:
        return self.initial_dynamics.evaluate(
            # TODO: could also be reference point...
            obstacle.center_position
        )

    def get_base_convergence(self, position: npt.ArrayLike) -> np.ndarray:
        return self.convergence_dynamics.evaluate(position)


class NonlinearRotationalAvoider(BaseAvoider):
    """
    NonlinearRotationalAvoider -> Rotational Obstacle Avoidance by additionally considering initial dynamics
    """

    # TODO:
    #   - don't use UnitDirection (as I assume it has a large overhead)

    def __init__(
        self,
        initial_dynamics: DynamicalSystem,
        obstacle_environment: RotationContainer,
        obstacle_convergence: ObstacleConvergenceDynamics,
        **kwargs,
    ) -> None:
        """Initial dynamics, convergence direction and obstacle list are used."""
        self._rotation_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_environment,
            # convergence_system=convergence_system,
            cut_off_gamma=10,
            **kwargs,
        )

        self.obstacle_convergence = obstacle_convergence

    @property
    def dimension(self):
        return self._rotation_avoider.initial_dynamics.dimension

    @property
    def cut_off_gamma(self):
        return self._rotation_avoider.cut_off_gamma

    @property
    def n_obstacles(self):
        return len(self._rotation_avoider.obstacle_environment)

    @property
    def obstacle_environment(self):
        return self._rotation_avoider.obstacle_environment

    def evaluate_initial_dynamics(self, position: np.ndarray) -> np.ndarray:
        return self._rotation_avoider.initial_dynamics.evaluate(position)

    # def evaluate_convergence_dynamics(self, position: np.ndarray) -> np.ndarray:
    #     return self._rotation_avoider.convergence_dynamics.evaluate(position)

    def avoid(self, position, initial_velocity):
        convergence_velocity = self.evaluate_convergence_dynamics(position)
        return self._rotation_avoider.avoid(
            position=position,
            initial_velocity=initial_velocity,
            convergence_velocity=convergence_velocity,
        )

    def evaluate(self, position, **kwargs):
        initial_velocity = self.evaluate_initial_dynamics(position)
        local_convergence_velocity = self.evaluate_weighted_dynamics(
            position, initial_velocity
        )

        return self._rotation_avoider.avoid(
            position=position,
            initial_velocity=initial_velocity,
            convergence_velocity=local_convergence_velocity,
            **kwargs,
        )

    def _compute_gamma_weights(self, position: np.ndarray):
        pass

    def evaluate_weighted_dynamics(
        self, position: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        """Returns the weighted-convergence velocity for all obstacles.

        Arguments
        ---------
        position: Vector of the array of the position of evaluation
        initial_velocity: The initial dynamics are used as a 'baseline' for the convergence
        """
        # convergence_velocity = self.evaluate_convergence_dynamics(position)
        # convergence_velocity = self.obstacle_convergence.get_base_convergence(position)
        if not (initial_norm := LA.norm(initial_velocity)):
            return initial_velocity
        initial_velocity = initial_velocity / initial_norm

        # TODO: this gamma/weight calculation could be shared...
        gamma_array = np.zeros((self.n_obstacles))
        for ii in range(self.n_obstacles):
            gamma_array[ii] = self._rotation_avoider.obstacle_environment[ii].get_gamma(
                position, in_global_frame=True
            )

        gamma_min = 1
        # Store weights -> mostly for visualization
        self.weights = np.zeros(self.n_obstacles)

        ind_obs = gamma_array <= gamma_min
        if sum_close := np.sum(ind_obs):
            # Dangerously close..
            weights = np.ones(sum_close) * 1.0 / sum_close
            weight_sum = 1

        else:
            ind_obs = gamma_array < self._rotation_avoider.cut_off_gamma

            if not np.sum(ind_obs):
                return initial_velocity

            weights = 1.0 / (gamma_array[ind_obs] - gamma_min) - 1 / (
                self.cut_off_gamma - gamma_min
            )
            if (weight_sum := np.sum(weights)) > 0:
                # Normalize weight, but leave possibility to be smaller than one (!)
                weights = weights / weight_sum

            # Influence of each obstacle -> but better mapping to [0, 1]
            ww_weights = (
                1 / gamma_array[ind_obs] - 1 / self._rotation_avoider.cut_off_gamma
            )
            ww_weights = ww_weights / (1 - 1 / self._rotation_avoider.cut_off_gamma)
            weights = weights * np.minimum(1, ww_weights)

        self.weights[ind_obs] = weights

        # Remaining convergence is the linear system, if it is far..
        # initial_norm = LA.norm(initial_velocity)
        # if weight_sum < 1:
        #     local_velocities = np.zeros((self.dimension, np.sum(ind_obs) + 1))

        #     weights = np.append(weights, 1 - weight_sum)

        #     if not initial_norm:
        #         return initial_velocity

        #     local_velocities[:, -1] = initial_velocity / initial_norm

        # else:

        # Initial velocity will be the 'base velocity'
        local_velocities = np.zeros((self.dimension, np.sum(ind_obs)))

        # Evaluating center directions for the relevant obstacles
        for ii, it_obs in enumerate(np.arange(self.n_obstacles)[ind_obs]):
            # local_velocities[:, ii] = self.evaluate_initial_dynamics(
            #     # TODO: could also be reference point...
            #     self._rotation_avoider.obstacle_environment[ii].center_position
            # )

            local_velocities[
                :, ii
            ] = self.obstacle_convergence.evaluate_convergence_around_obstacle(
                position, obstacle=self._rotation_avoider.obstacle_environment[ii]
            )

            if not LA.norm(local_velocities[:, ii]):
                # What should be done here (?)
                # <-> smoothly reduce the weight as we approach the center(?)
                raise NotImplementedError()

        # Weighted sum -> should have the same result as 'the graph summing'
        # (but current implementation of weighted_sum is more stable)
        averaged_direction = get_directional_weighted_sum(
            null_direction=initial_velocity,
            weights=weights,
            directions=local_velocities,
        )
        # breakpoint()

        return initial_norm * averaged_direction
