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

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.dynamics.sequenced_dynamics import evaluate_dynamics_sequence
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)
from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence

from nonlinear_avoidance.datatypes import Vector


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


class ConvergenceDynamicsWithoutSingularity:
    def __init__(
        self, convergence_dynamics: DynamicalSystem, initial_dynamics: DynamicalSystem
    ) -> None:
        self.initial_dynamics = initial_dynamics
        self.convergence_dynamics = convergence_dynamics

    def evaluate_convergence_around_obstacle(
        self, position: npt.ArrayLike, obstacle: Obstacle
    ) -> np.ndarray:
        velocity = self.initial_dynamics.evaluate(obstacle.global_reference_point)
        if np.any(np.isnan(velocity)):
            breakpoint()
        return velocity

    def evaluate_rotation_position_to_transform(
        self, position: np.ndarray, obstacle: Obstacle
    ) -> VectorRotationXd:
        """Returns zero-transform as there is no singularity."""
        velocity = self.convergence_dynamics.evaluate(position)
        return VectorRotationXd.from_directions(velocity, velocity)

    def evaluate_rotation_start_to_end(
        self, start: np.ndarray, end: np.ndarray, center: np.ndarray
    ) -> VectorRotationXd:
        """Returns zero-transform as there is no singularity."""
        velocity = self.initial_dynamics.evaluate(start)
        return VectorRotationXd.from_directions(velocity, velocity)

    def evaluate_projected_weight(
        self, position: np.ndarray, obstacle: Obstacle
    ) -> float:
        gamma = obstacle.get_gamma(position, in_global_frame=True)
        if gamma >= 1:
            weight = 1.0 / gamma
        else:
            weight = 1
        return weight

    def evaluate_convergence_sequence_around_obstacle(
        self,
        position: Vector,
        obstacle: Obstacle,
        initial_sequence: VectorRotationSequence,
    ) -> VectorRotationSequence:
        self.obstacle = obstacle
        obstacle_sequence = evaluate_dynamics_sequence(
            obstacle.get_reference_point(in_global_frame=True),
            dynamics=self.initial_dynamics,
        )
        return obstacle_sequence

    def get_base_convergence(self, position: npt.ArrayLike) -> np.ndarray:
        return self.convergence_dynamics.evaluate(position)


class LinearConvergenceDynamics(ConvergenceDynamicsWithoutSingularity):
    pass


class SingularityConvergenceDynamics(BaseAvoider):
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

        if obstacle_convergence is None:
            # Assumption of stable center
            self.obstacle_convergence = ProjectedRotationDynamics(
                attractor_position=initial_dynamics.attractor_position,
                initial_dynamics=initial_dynamics,
            )

        else:
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

    def evaluate_initial_dynamics_sequence(
        self, position: np.ndarray
    ) -> Optional[VectorRotationSequence]:
        return evaluate_dynamics_sequence(
            position, self._rotation_avoider.initial_dynamics
        )

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
        if not (initial_norm := LA.norm(initial_velocity)):
            return initial_velocity

        initial_velocity = initial_velocity / initial_norm

        convergence_velocity = self.evaluate_weighted_dynamics(
            position, initial_velocity
        )

        rotated_velocity = self._rotation_avoider.avoid(
            position=position,
            initial_velocity=initial_velocity,
            convergence_velocity=convergence_velocity * initial_norm,
            **kwargs,
        )

        return rotated_velocity * initial_norm

    def evaluate_sequence(self, position: np.ndarray) -> np.ndarray:
        if hasattr(self._rotation_avoider.initial_dynamics, "evaluate_magnitude"):
            magnitude = self._rotation_avoider.initial_dynamics.evaluate_magnitude(
                position
            )
        else:
            initial_velocity = self._rotation_avoider.initial_dynamics.evaluate(
                position
            )
            magnitude = np.linalg.norm(initial_velocity)

        if not magnitude:
            return np.zeros_like(position)

        initial_sequence = self.evaluate_initial_dynamics_sequence(position)
        if initial_sequence is None:
            return np.zeros(self.dimension)

        convergence_sequence = self.evaluate_weighted_dynamics_sequence(
            position, initial_sequence
        )

        rotated_sequence = self._rotation_avoider.avoid_sequence(
            position=position,
            initial_sequence=initial_sequence,
            convergence_sequence=convergence_sequence,
            convergence_default=False,
        )

        return rotated_sequence.get_end_vector() * magnitude

    def get_base_convergence(self, position: np.ndarray) -> np.ndarray:
        # TODO: test this...
        raise NotImplementedError()

    def evaluate_convergence_around_obstacle(self, position, obstacle):
        raise NotImplementedError()

    def compute_gamma_weights(
        self, position: np.ndarray, gamma_min: float = 1.0, weight_power: float = 2.0
    ) -> np.ndarray:
        gamma_array = np.zeros((self.n_obstacles))
        for ii in range(self.n_obstacles):
            gamma_array[ii] = self._rotation_avoider.obstacle_environment[ii].get_gamma(
                position, in_global_frame=True
            )

        # Store weights -> mostly for visualization
        self.weights = np.zeros(self.n_obstacles)

        ind_obs = gamma_array <= gamma_min
        if sum_close := np.sum(ind_obs):
            # Dangerously close..
            weights = np.ones(sum_close) * 1.0 / sum_close
            weight_sum = 1.0

        else:
            ind_obs = gamma_array < self._rotation_avoider.cut_off_gamma

            if not np.sum(ind_obs):
                return np.zeros(0, dtype=bool)

            weights = 1.0 / (gamma_array[ind_obs] - gamma_min) - 1 / (
                self.cut_off_gamma - gamma_min
            )
            weights = weights**weight_power

            # weights = weights**weight_power
            if (weight_sum := np.sum(weights)) > 0:
                # Normalize weight, but leave possibility to be smaller than one (!)
                weights = weights / weight_sum

            # Influence of each obstacle -> but better mapping to [0, 1]
            # ww_weights = (
            #     1.0 / gamma_array[ind_obs] - 1.0 / self._rotation_avoider.cut_off_gamma
            # )
            # ww_weights = ww_weights / (1.0 - 1.0 / self._rotation_avoider.cut_off_gamma)
            # breakpoint()
            # weights = weights * np.minimum(1, ww_weights)

        self.weights[ind_obs] = weights
        return ind_obs

    def evaluate_weighted_dynamics_sequence(
        self, position: np.ndarray, initial_sequence: VectorRotationSequence
    ) -> VectorRotationSequence:
        # ind_obs = self.compute_gamma_weights(position)
        # if not len(ind_obs):
        #     return initial_sequence

        # Assumption of shared root_id
        root_id = -10
        initial_id = -1
        direction_tree = VectorRotationTree.from_sequence(
            root_id=root_id, node_id=initial_id, sequence=initial_sequence
        )

        node_list = []
        node_weights = []

        # for ii, it_obs in enumerate(np.arange(self.n_obstacles)[ind_obs]):
        for ii, it_obs in enumerate(np.arange(self.n_obstacles)):
            projected_weight = self.obstacle_convergence.evaluate_projected_weight(
                position, self._rotation_avoider.obstacle_environment[ii]
            )
            if projected_weight <= 0:
                continue

            obstacle_convergence_sequence = (
                self.obstacle_convergence.evaluate_convergence_sequence_around_obstacle(
                    position,
                    obstacle=self._rotation_avoider.obstacle_environment[ii],
                    initial_sequence=initial_sequence,
                )
            )
            direction_tree.add_sequence(
                sequence=obstacle_convergence_sequence,
                node_id=it_obs,
                parent_id=root_id,
            )
            node_list.append(it_obs)
            node_weights.append(projected_weight)
            # node_weights.append(self.weights[it_obs] * projected_weight)

            # print("obs conv", obstacle_convergence_sequence.get_end_vector())

        # print("Got all obs-convergence")
        # node_list = np.append(np.arange(self.n_obstacles)[ind_obs], initial_id)
        # node_weights = np.append(self.weights[ind_obs], 1 - sum(self.weights[ind_obs]))

        if (weight_sum := np.sum(node_weights)) >= 1:
            node_weights = np.array(node_weights) / sum(node_weights)
        else:
            node_list.append(initial_id)
            node_weights.append(1 - sum(node_weights))

        rotation_sequence = direction_tree.reduce_weighted_to_sequence(
            node_list=node_list,
            weights=node_weights,
        )

        return rotation_sequence

    def evaluate_weighted_dynamics(
        self, position: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        """Returns the weighted-convergence velocity for all obstacles.

        Arguments
        ---------
        position: Vector of the array of the position of evaluation
        initial_velocity: The initial dynamics are used as a 'baseline' for the convergence
        """
        # TODO: this gamma/weight calculation could be shared...
        ind_obs = self.compute_gamma_weights(position)
        if not len(ind_obs):
            return initial_velocity

        # Initial velocity will be the 'base velocity'
        # TODO: storing of 'local_velocities' not needed anymore as its in the rotation tree
        local_velocities = np.zeros((self.dimension, np.sum(ind_obs)))
        direction_tree = VectorRotationTree()
        direction_tree.set_root(root_idx=-1, direction=initial_velocity)

        # Evaluating center directions for the relevant obstacles
        for ii, it_obs in enumerate(np.arange(self.n_obstacles)[ind_obs]):
            local_velocities[
                :, ii
            ] = self.obstacle_convergence.evaluate_convergence_around_obstacle(
                position, obstacle=self._rotation_avoider.obstacle_environment[ii]
            )

            direction_tree.add_node(
                direction=local_velocities[:, ii], node_id=it_obs, parent_id=-1
            )

            if not LA.norm(local_velocities[:, ii]):
                # What should be done here (?)
                # <-> smoothly reduce the weight as we approach the center(?)
                raise NotImplementedError()

        # Weighted sum -> should have the same result as 'the graph summing'
        # (but current implementation of weighted_sum is more stable)
        # averaged_direction = get_directional_weighted_sum(
        #     null_direction=initial_velocity,
        #     weights=weights,
        #     directions=local_velocities,
        # )

        rotation_sequence = direction_tree.reduce_weighted_to_sequence(
            node_list=np.arange(self.n_obstacles)[ind_obs],
            weights=self.weights[ind_obs],
        )

        return rotation_sequence.get_end_vector()

        # TODO: finish this with history-of-rotation
        # print("tree_average_dir", tree_average_dir)
        # print("averaged_direction", averaged_direction)
        # breakpoint()

        # return initial_norm * averaged_direction


class NonlinearRotationalAvoider(SingularityConvergenceDynamics):
    pass
