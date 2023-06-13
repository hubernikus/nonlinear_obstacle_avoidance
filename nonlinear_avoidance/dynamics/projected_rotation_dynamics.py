"""
Class to deviate a DS based on an underlying obstacle.
"""
import sys
import math
import copy
import os
from typing import Optional

# from enum import Enum

import numpy as np
from numpy import linalg as LA
import warnings

from vartools.dynamical_systems import DynamicalSystem
from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import get_directional_weighted_sum

# from vartools.linalg import get_orthogonal_basis
# from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.avoidance import (
    obstacle_avoidance_rotational,
)
from nonlinear_avoidance.dynamics.sequenced_dynamics import evaluate_dynamics_sequence

from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.vector_rotation import VectorRotationSequence
from nonlinear_avoidance.vector_rotation import VectorRotationTree

from nonlinear_avoidance.datatypes import Vector


class ProjectedRotationDynamics:
    """
    A dynamical system which locally modulates

    Properties
    ----------
    obstacle: The obstacle around which shape the DS is deformed
    attractor_position: Position of the attractor
    center_direction: The direction of the DS at the center of the obstacle

    (Optional)
    min_gamma (> 1): The position at which the DS has 'maximum' rotation
    max_gamma (> min_gamma): The gamma-distance at which the influence stops.
    """

    # TODO: include Lyapunov function which checks avoidance.
    # TODO: should this really be a class or rather refactored (?)
    # this might effect the 'single-saddle point' on the surface'

    def __init__(
        self,
        attractor_position: np.ndarray,
        reference_velocity: Optional[np.ndarray] = None,  # Probably remove this..
        initial_dynamics: Optional[np.ndarray] = None,
        obstacle: Optional[Obstacle] = None,
        min_gamma: float = 1,
        max_gamma: float = 10,
    ) -> None:
        self.dimension = attractor_position.shape[0]

        self.obstacle = obstacle
        self.attractor_position = attractor_position

        # self.maximum_velocity = LA.norm(reference_velocity)
        # if not self.maximum_velocity:
        #     raise ValueError("Zero velocity was obtained.")

        # reference_velocity = reference_velocity / self.maximum_velocity

        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        # Modify if needed
        self.attractor_influence = 3
        self.dotprod_projection_power = 2

        if initial_dynamics is None:
            self.initial_dynamics = LinearSystem(attractor_position=attractor_position)
        else:
            self.initial_dynamics = initial_dynamics

        # self.base = get_orthogonal_basis()
        # self.deviation = get_angle_space(reference_velocity, null_matrix=self.base)

    def get_projected_gamma(self, position: Vector) -> float:
        # Get gamma
        gamma = self.obstacle.get_gamma(position, in_global_frame=True)
        if gamma >= self.max_gamma:
            return self.max_gamma

        elif gamma >= self.min_gamma:
            # Weight is additionally based on dot-product
            attractor_dir = self.attractor_position - position
            if dist_attractor := LA.norm(attractor_dir):
                attractor_dir = attractor_dir / dist_attractor
                dot_product = np.dot(attractor_dir, self.rotation.base0)
                gamma = gamma ** (2 / (dot_product + 1))

                # if dist_obs := LA.norm(self.obstacle.center_position):
                if dist_obs := LA.norm(self.obstacle.global_reference_point):
                    dist_stretching = LA.norm(position) / LA.norm(
                        # self.obstacle.center_position
                        self.obstacle.global_reference_point
                    )
                    gamma = gamma**dist_stretching
                else:
                    gamma = self.max_gamma

            else:
                gamma = self.max_gamma

    def _get_deflation_weight(self, gamma: float) -> float:
        # TODO: this needs to be improved to ensure that the projected position
        # is outside the obstacle
        # return 1.0 / gamma
        return 1.0

    def _get_position_after_deflating_obstacle(
        self,
        position: Vector,
        in_obstacle_frame: bool = True,
        deflation_weight: float = 1.0,
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk.
        (in the obstacle-frame.)

        Due to the orientation the evaluation should be done in the obstacle frame (!)
        """
        radius = self.obstacle.get_local_radius(
            position, in_global_frame=not (in_obstacle_frame)
        )

        if in_obstacle_frame:
            relative_position = position
        else:
            relative_position = position - self.obstacle.global_reference_point

        pos_norm = LA.norm(relative_position)

        if pos_norm < radius:
            # TODO: we could keep partial position (?)
            if in_obstacle_frame:
                return np.zeros_like(position)
            else:
                return np.copy(self.obstacle.global_reference_point)

        deflated_position = (
            (pos_norm - radius * deflation_weight) / pos_norm
        ) * relative_position

        if in_obstacle_frame:
            return deflated_position
        else:
            return deflated_position + self.obstacle.global_reference_point

    def _get_position_after_inflating_obstacle(
        self,
        position: Vector,
        in_obstacle_frame: bool = True,
        deflation_weight: float = 1.0,
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk.

        Due to the orientation the evaluation should be done in the obstacle frame (!)
        """
        radius = self.obstacle.get_local_radius(
            position, in_global_frame=not (in_obstacle_frame)
        )

        if in_obstacle_frame:
            # Make sure it is float
            relative_position = np.copy(position).astype(float)
        else:
            relative_position = position - self.obstacle.global_reference_point

        if not (pos_norm := LA.norm(relative_position)):
            # Needs a tiny value
            relative_position[0] = 1e-6
            pos_norm = relative_position[0]

        inflated_position = (
            (pos_norm + radius * deflation_weight) / pos_norm
        ) * relative_position

        if in_obstacle_frame:
            return inflated_position
        else:
            return inflated_position + self.obstacle.global_reference_point

    def _get_folded_position_opposite_kernel_point(
        self,
        position: Vector,
        attractor_position: Vector,
        in_obstacle_frame: bool = True,
    ) -> Vector:
        """Returns the relative position folded with respect to the dynamics center.
        obstacle_center position is 'stable'
        """
        if in_obstacle_frame:
            # If it's in the obstacle-frame => needs to be inverted...
            vec_attractor_to_obstacle = (-1) * attractor_position
        else:
            vec_attractor_to_obstacle = (
                self.obstacle.global_reference_point - attractor_position
            )

        # 'Unfold' the circular plane into an infinite -y/+y-plane
        if not (dist_attr_obs := LA.norm(vec_attractor_to_obstacle)):
            warnings.warn("Implement for position at center.")
            return position
            # raise NotImplementedError("Implement for position at center.")

        dir_attractor_to_obstacle = vec_attractor_to_obstacle / dist_attr_obs

        # Get values in the attractor frame of reference
        vec_attractor_to_position = position - attractor_position

        basis = get_orthogonal_basis(dir_attractor_to_obstacle)
        transformed_position = basis.T @ vec_attractor_to_position

        # Stretch x-values along x-axis in order to have a x-weight at the attractor
        if dist_attr_pos := LA.norm(vec_attractor_to_position):
            transformed_position[0] = dist_attr_obs * math.log(
                dist_attr_pos / dist_attr_obs
            )

        else:
            transformed_position[0] = (-1) * sys.float_info.max

        dir_attractor_to_position = vec_attractor_to_position / dist_attr_pos
        dot_prod = np.dot(dir_attractor_to_position, dir_attractor_to_obstacle)

        if dot_prod <= -1.0:
            # Put it very far away in a random direction
            transformed_position[1] = sys.float_info.max

        elif dot_prod < 1.0:
            if trafo_norm := LA.norm(transformed_position[1:]):
                # Numerical error can lead to zero division, even though it should be excluded
                dotprod_factor = 2 / (1 + dot_prod) - 1
                dotprod_factor = dotprod_factor ** (1.0 / self.dotprod_projection_power)

                transformed_position[1:] = (
                    transformed_position[1:]
                    / LA.norm(transformed_position[1:])
                    * dotprod_factor
                )
        expanded_position = basis @ transformed_position
        # TODO: should the scaling be applied to all dimensions?
        # How would this change the effect for obstacles / far or close
        expanded_position[0] = expanded_position[0] * dist_attr_obs

        if not in_obstacle_frame:
            expanded_position = expanded_position + self.obstacle.global_reference_point
        return expanded_position

    def _get_unfolded_position_opposite_kernel_point(
        self,
        transformed_position: Vector,
        attractor_position: Vector,
        in_obstacle_frame: bool = True,
    ) -> Vector:
        """Returns UNfolded rleative position folded with respect to the dynamic center.
        This function is used for debugging to check coherence of the method.
        Input and output are in the obstacle frame of reference."""
        if in_obstacle_frame:
            vec_attractor_to_obstacle = (-1) * attractor_position
        else:
            vec_attractor_to_obstacle = (
                self.obstacle.global_reference_point - attractor_position
            )

        if not (dist_attr_obs := LA.norm(vec_attractor_to_obstacle)):
            # No unfolding possible - TODO: make the 'switch' smoother
            return transformed_position

        dir_attractor_to_obstacle = vec_attractor_to_obstacle / dist_attr_obs

        # Dot product is sufficient, as we only need first element.
        # Rotation is performed with VectorRotationXd
        # vec_attractor_to_position = transformed_position - attractor_position
        if in_obstacle_frame:
            vec_obstacle_to_position = transformed_position
        else:
            vec_obstacle_to_position = (
                transformed_position - self.obstacle.global_reference_point
            )

        # The normalization with with attractor distance scales the 'radial' direction
        vec_obstacle_to_position[0] = vec_obstacle_to_position[0] / dist_attr_obs

        if not (pos_norm := LA.norm(vec_obstacle_to_position)):
            # At the center of the obstacle -> attractor dynamcis
            return transformed_position

        dir_obstacle_to_position = vec_obstacle_to_position / pos_norm

        # radius = np.dot(dir_attractor_to_obstacle, dir_attractor_to_position)
        radius = np.dot(dir_attractor_to_obstacle, dir_obstacle_to_position) * pos_norm

        # Ensure that the square root stays positive close to limits
        # dotprod_factor = pos_norm * math.sqrt(max(1 - radius**2, 0))
        dotprod_factor = math.sqrt(max(pos_norm**2 - radius**2, 0))
        dot_prod = dotprod_factor**self.dotprod_projection_power
        dot_prod = 2.0 / (dot_prod + 1) - 1

        if dot_prod < 1:
            dir_perp = dir_obstacle_to_position - dir_attractor_to_obstacle * dot_prod

            rotation_ = VectorRotationXd.from_directions(
                vec_init=dir_attractor_to_obstacle,
                vec_rot=dir_perp / LA.norm(dir_perp),
            )
            rotation_.rotation_angle = math.acos(dot_prod)

            # Initially a unit vector
            uniform_position = rotation_.rotate(dir_attractor_to_obstacle)
        else:
            uniform_position = dir_attractor_to_obstacle

        relative_position = (
            uniform_position * math.exp(radius / dist_attr_obs) * dist_attr_obs
        )

        # Move out-of from attractor-frame
        relative_position = relative_position + attractor_position

        # # Simplified transform (without rotation), since everything was in obstacle frame
        # if not in_obstacle_frame:
        #     relative_position = relative_position + self.obstacle.pose.position
        # No transform to / from obstacle frame necessary, as attractor_positiion
        # should contain it

        return relative_position

    def get_projected_position_and_rotation(
        self, position: Vector
    ) -> tuple[Vector, VectorRotationXd]:
        raise NotImplementedError()

    def get_projected_position(self, position: Vector) -> Vector:
        """Projected point in 'linearized' environment

        Assumption of the point being outside of the obstacle."""

        # Do the evaluation in local frame
        relative_position = self.obstacle.pose.transform_position_to_relative(position)
        relative_attractor = self.obstacle.pose.transform_position_to_relative(
            self.attractor_position
        )

        if self.obstacle.get_gamma(relative_attractor, in_obstacle_frame=True) < 1:
            return position
            # breakpoint()  # This should be treated specially!(!)

        gamma = self.obstacle.get_gamma(relative_position, in_obstacle_frame=True)

        MIN_GAMMA = 1
        if gamma <= MIN_GAMMA:
            # Position in obstacle -> projection does not have an effect
            return position

        weight = self._get_deflation_weight(gamma)

        # Shrunk position
        deflated_position = self._get_position_after_deflating_obstacle(
            relative_position, in_obstacle_frame=True, deflation_weight=weight
        )
        deflated_attractor = self._get_position_after_deflating_obstacle(
            relative_attractor, in_obstacle_frame=True, deflation_weight=weight
        )

        folded_position = self._get_folded_position_opposite_kernel_point(
            deflated_position, deflated_attractor, in_obstacle_frame=True
        )

        if np.linalg.norm(folded_position) > 1e10:
            # Return directly to avoid numerical errors
            return folded_position

        inflated_position = self._get_position_after_inflating_obstacle(
            folded_position, in_obstacle_frame=True, deflation_weight=weight
        )

        projected_position = self.obstacle.pose.transform_position_from_relative(
            inflated_position
        )

        return projected_position

    def _get_lyapunov_gradient(self, position: Vector) -> Vector:
        """Returns the Gradient of the Lyapunov function.
        For now, we assume a quadratic Lyapunov function."""
        # Weight is additionally based on dot-product
        attractor_dir = self.attractor_position - position
        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(position)

        attractor_dir = attractor_dir / dist_attractor
        return attractor_dir

    def _get_projected_lyapunov_gradient(self, position: Vector) -> Vector:
        """Returns projected lyapunov gradient function.

        It is assumed that z-axis is the base gradient."""
        attractor_dir = self.attractor_position - self.obstacle.global_reference_point

        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(attractor_dir)

        return attractor_dir / dist_attractor

    def _get_vector_rotation_of_modulation(
        self, position: Vector, velocity: Vector
    ) -> VectorRotationXd:
        """Returns the rotation of the modulation close to an obstacle."""
        if not (vel_norm := LA.norm(velocity)):
            return VectorRotationXd(np.eye(self.dimension, 2), rotation_angle=0.0)

        modulated_velocity = obstacle_avoidance_rotational(
            position,
            velocity,
            obstacle_list=[self.obstacle],
            convergence_velocity=velocity,
        )
        if not (mod_vel_norm := LA.norm(modulated_velocity)):
            return VectorRotationXd(np.eye(self.dimension, 2), rotation_angle=0.0)

        return VectorRotationXd.from_directions(
            velocity / vel_norm, modulated_velocity / mod_vel_norm
        )

    def get_base_convergence(self, position: Vector) -> Vector:
        # This should be either +/- attractor-position
        dist_attr = self.attractor_position - position
        if dist_norm := LA.norm(dist_attr):
            return dist_attr / LA.norm(dist_attr)
        else:
            return dist_attr

    def evaluate_rotation_start_to_end(
        self, start: np.ndarray, end: np.ndarray, center: np.ndarray
    ) -> Optional[VectorRotationXd]:
        """Returns VectorRotationXd needed to go from position to the obstacle-reference.
        The base vectors are pointing towards the attractor for compatibalitiy with straight-stable dynamics.
        """
        dir_attr_to_pos = center - start
        if not (dir_norm := np.linalg.norm(dir_attr_to_pos)):
            # We're at the attractor -> zero-velocity
            return None

        dir_attr_to_pos = dir_attr_to_pos / dir_norm

        dir_attr_to_obs = center - end
        if not (obs_norm := LA.norm(dir_attr_to_obs)):
            raise NotImplementedError("Obstacle is at attractor.")
        dir_attr_to_obs = dir_attr_to_obs / obs_norm

        if np.dot(dir_attr_to_pos, dir_attr_to_obs) <= -1:
            return None

        rotation_pos_to_transform = VectorRotationXd.from_directions(
            dir_attr_to_pos, dir_attr_to_obs
        )
        return rotation_pos_to_transform

    def evaluate_rotation_position_to_transform(
        self, position: np.ndarray, obstacle: Obstacle
    ) -> Optional[VectorRotationXd]:
        """Returns VectorRotationXd needed to go from position to the obstacle-reference.
        The base vectors are pointing towards the attractor for compatibalitiy with straight-stable dynamics.
        """
        dir_attr_to_pos = self.attractor_position - position
        if not (dir_norm := LA.norm(dir_attr_to_pos)):
            # We're at the attractor -> zero-velocity
            return None
        dir_attr_to_pos = dir_attr_to_pos / dir_norm

        dir_attr_to_obs = self.attractor_position - obstacle.global_reference_point
        if not (obs_norm := LA.norm(dir_attr_to_obs)):
            raise NotImplementedError("Obstacle is at attractor.")
        dir_attr_to_obs = dir_attr_to_obs / obs_norm

        if np.dot(dir_attr_to_pos, dir_attr_to_obs) <= -1:
            return None

        rotation_pos_to_transform = VectorRotationXd.from_directions(
            dir_attr_to_pos, dir_attr_to_obs
        )
        return rotation_pos_to_transform

    def compute_obstacle_convergence_sequence(
        self,
        position: np.ndarray,
        root_obs: Obstacle,
        initial_sequence: VectorRotationSequence,
    ) -> VectorRotationSequence:
        """Returns the sequence at the obstacle position and the relative rotation towards it."""

        weight = self.evaluate_projected_weight(position, root_obs)
        # Start this weight reduction later
        # weight = max(1, weight * 2)

        if weight <= 0:
            relative_rotation = None
            return initial_sequence

        # Trafo is expected to be not None, as we have: weight > 0
        trafo_pos_to_attr = self.evaluate_rotation_position_to_transform(position)

        convergence_sequence = evaluate_dynamics_sequence(
            root_obs.get_reference_point(in_global_frame=True),
            self.initial_dynamics,
        )
        if convergence_sequence is None:
            raise NotImplementedError("Obstacle at center.")

        # Get an average between convergence / initial direction
        root_id = 0
        init_id = 1
        self.conv_tree = VectorRotationTree.from_sequence(
            root_id=root_id,
            node_id=init_id,
            sequence=initial_sequence,
        )
        obs_id = 2
        self.conv_tree.add_node(
            orientation=trafo_pos_to_attr,
            node_id=obs_id,
            parent_id=root_id,
        )
        conv_id = 3
        self.conv_tree.add_sequence(
            sequence=convergence_sequence, node_id=conv_id, parent_id=obs_id
        )

        weighted_sequene = self.conv_tree.reduce_weighted_to_sequence(
            node_list=[conv_id, init_id], weights=[weight, (1 - weight)]
        )
        return weighted_sequene

    def evaluate_rotation_shrunkposition_to_transform(
        self, position: np.ndarray, obstacle: Obstacle
    ) -> Optional[VectorRotationXd]:
        """Returns VectorRotationXd needed to go from position to the obstacle-reference.
        The base vectors are pointing towards the attractor for compatibalitiy with straight-stable dynamics.
        """
        shrinking_weight = 1.0
        shrunk_position = self._get_position_after_deflating_obstacle(
            position, in_obstacle_frame=False, weight=shrinking_weight
        )
        shrunk_attractor = self._get_position_after_deflating_obstacle(
            self.attractor_position, in_obstacle_frame=False, weight=shrinking_weight
        )

        dir_attr_to_pos = shrunk_attractor - shrunk_position
        if not (dir_norm := LA.norm(dir_attr_to_pos)):
            # We're at the attractor -> zero-velocity
            return None
        dir_attr_to_pos = dir_attr_to_pos / dir_norm

        dir_attr_to_obs = shrunk_attractor - obstacle.global_reference_point
        if not (obs_norm := LA.norm(dir_attr_to_obs)):
            raise NotImplementedError("Obstacle is at attractor.")
        dir_attr_to_obs = dir_attr_to_obs / obs_norm

        if np.dot(dir_attr_to_pos, dir_attr_to_obs) <= -1:
            return None

        rotation_pos_to_transform = VectorRotationXd.from_directions(
            dir_attr_to_pos, dir_attr_to_obs
        )
        return rotation_pos_to_transform

    def evaluate_projected_weight(
        self, position: np.ndarray, obstacle: Obstacle, weight_power: float = 1.0 / 2.0
    ) -> float:
        # Obstacle velocity will not change when being transformed, as it's the static point
        self.obstacle = obstacle  # Just to be sure that its the same...

        projected_position = self.get_projected_position(position)
        proj_gamma = obstacle.get_gamma(projected_position, in_global_frame=True)
        gamma = obstacle.get_gamma(position, in_global_frame=True)

        if proj_gamma <= 1.0 or gamma <= 1.0:
            return 1.0

        weight = (1.0 / ((proj_gamma - 1) * (gamma - 1) + 1)) ** weight_power
        # if weight < 1e-2:
        #     breakpoint()
        return min(weight, 1)

    def evaluate_convergence_sequence_around_obstacle(
        self,
        position: Vector,
        obstacle: Obstacle,
        initial_sequence: VectorRotationSequence,
    ) -> VectorRotationSequence:
        self.obstacle = obstacle

        # Evaluate weighted position
        rotation_pos_to_transform = self.evaluate_rotation_position_to_transform(
            position, obstacle
        )
        if rotation_pos_to_transform is None:
            # Opposite -> zero weight
            return initial_sequence

        obstacle_sequence = evaluate_dynamics_sequence(
            obstacle.get_reference_point(in_global_frame=True),
            dynamics=self.initial_dynamics,
        )
        obstacle_sequence.push_root_from_base_and_angle(
            rotation_pos_to_transform.base, rotation_pos_to_transform.rotation_angle
        )
        return obstacle_sequence

    def evaluate_convergence_around_obstacle(
        self,
        position: Vector,
        obstacle: Obstacle,
    ) -> Vector:
        """Returns the 'averaged' direction.l"""
        # Store obstacle -> TODO: this should be done more cleanly
        self.obstacle = obstacle

        initial_velocity = self.initial_dynamics.evaluate(position)
        obstacle_velocity = self.initial_dynamics.evaluate(
            obstacle.global_reference_point
        )

        # base_convergence_direction = self.get_base_convergence(position)
        # The transform for the obstacle-velocity is zero
        rotation_pos_to_transform = self.evaluate_rotation_position_to_transform(
            position, obstacle
        )
        if rotation_pos_to_transform is None:
            # At the attractor or opposite side -> no linearization
            return initial_velocity

        initial_velocity_transformed = rotation_pos_to_transform.rotate(
            initial_velocity
        )
        weight = self.evaluate_projected_weight(position, obstacle)

        # print("initial_velocity_transformed1", initial_velocity_transformed)

        # TODO: use VectorRotationXd for this...
        averaged_direction_transformed = get_directional_weighted_sum(
            null_direction=initial_velocity_transformed,
            directions=np.vstack((obstacle_velocity, initial_velocity_transformed)).T,
            weights=[weight, 1 - weight],
            normalize=True,
        )

        # The 'back-rotation' only needs to be applied, when it's not linearized,
        # we hence have to weight it
        averaged_direction = rotation_pos_to_transform.rotate(
            averaged_direction_transformed, rot_factor=(-1) * (1 - weight)
        )

        # print("weight", weight)
        # print("initial_velocity_transformed2", initial_velocity_transformed)
        # print("averaged_direction_transformed", averaged_direction_transformed)
        # print("averaged_direction", averaged_direction)
        # breakpoint()

        averaged_direction = averaged_direction * LA.norm(initial_velocity)
        return averaged_direction
