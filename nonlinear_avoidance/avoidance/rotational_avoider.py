"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2021-09-01

import warnings
import copy
import math
from typing import Optional

# from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_with_circle, IntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import get_angle_from_vector
from vartools.directional_space import get_vector_from_angle
from vartools.directional_space import UnitDirection
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity
from dynamic_obstacle_avoidance.avoidance import BaseAvoider
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence

# Type definition
Vector = np.ndarray


class RotationalAvoider(BaseAvoider):
    """
    RotationalAvoider -> Obstacle Avoidance based on local avoider.

    Attributes
    ----------
    """

    # TODO:
    #   - don't use UnitDirection (as it has a large overhead)
    #   - put back into function for simplified changing of avoidance
    #   - remove redundant functions.

    def __init__(
        self,
        initial_dynamics: DynamicalSystem = None,
        obstacle_environment=None,
        convergence_system: Optional[DynamicalSystem] = None,
        cut_off_gamma: float = 1e6,
        tail_rotation: bool = False,
        convergence_radius: float = math.pi / 2.0,
    ) -> None:
        """Initial dynamics, convergence direction and obstacle list are used."""
        super().__init__(
            initial_dynamics=initial_dynamics, obstacle_environment=obstacle_environment
        )

        if convergence_system is None:
            self.convergence_system = self.initial_dynamics
        else:
            self.convergence_system = convergence_system

        self.cut_off_gamma = cut_off_gamma

        # Zero continuation power -> not smoothing at the end
        # The larger the smoother (a good value is 0.3) )
        self.smooth_continuation_power = 0.3

        self.tail_rotation = tail_rotation
        self.convergence_radius = convergence_radius

    @property
    def dimension(self) -> int:
        return self.initial_dynamics.dimension

    @property
    def n_obstacles(self) -> int:
        return self.obstacle_environment.n_obstacles

    @property
    def convergence_dynamics(self):
        # Compatibility
        return self.convergence_system

    def evaluate(self, position):
        return self.avoid(position)

    @staticmethod
    def compute_obstacle_gamma(
        position: Vector, obstacle_list: ObstacleContainer
    ) -> np.ndarray:
        gamma_array = np.zeros((len(obstacle_list)))
        for ii in range(len(obstacle_list)):
            gamma_array[ii] = obstacle_list[ii].get_gamma(
                position, in_global_frame=True
            )

            if gamma_array[ii] < 1 and not obstacle_list[ii].is_boundary:
                # Since boundaries are mutually subtracted,
                # raise NotImplementedError()
                warnings.warn("The evaluation is in the boundary.")
                # TODO: the repulsion could / should be increased with increasing
                # penetration of the obstacle$
                gamma_array[ii] = 1

        return gamma_array

    @staticmethod
    def compute_normal_tensor(
        position: np.ndarray,
        obstacle_list: ObstacleContainer,
        ind_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        dimension = position.shape[0]

        if ind_obs is None:
            ind_obs = np.ones(len(obstacle_list), dtype=bool)

        normal_orthogonal_matrix = np.zeros((dimension, dimension, np.sum(ind_obs)))

        normal_dirs = np.zeros((dimension, len(obstacle_list)))
        for it, it_obs in enumerate(np.arange(len(obstacle_list))[ind_obs]):
            # gamma_proportional[it] = obstacle_list[it_obs].get_gamma(
            #     position,
            #     in_global_frame=True,
            # )

            normal_dirs[:, it] = obstacle_list[it_obs].get_normal_direction(
                position, in_global_frame=True
            )
            normal_orthogonal_matrix[:, :, it] = get_orthogonal_basis(
                normal_dirs[:, it]
            )
        return normal_orthogonal_matrix

    def avoid_sequence(
        self,
        position: np.ndarray,
        initial_sequence: VectorRotationSequence,
        convergence_sequence: VectorRotationSequence,
        convergence_default: bool = True,
    ) -> VectorRotationSequence:
        """The evaluation is under the assumption that only close obstacles are in the list.

        IMPORTANT: The initial_sequence is expected to be in the relative-frame,
        i.e., with respect to the obstacles velocity."""
        if not self.n_obstacles:
            return initial_sequence

        gamma_array = self.compute_obstacle_gamma(position, self.obstacle_environment)
        importance_weights = compute_weights(gamma_array)
        inv_gamma_weight = get_weight_from_inv_of_gamma(gamma_array)
        normal_orthogonal_matrix = self.compute_normal_tensor(
            position, self.obstacle_environment
        )

        # Create tree from initial dynamics and convergence dynamics
        root_id = -10
        initial_id = -2
        conv_id = -1
        rotated_tree = VectorRotationTree.from_sequence(
            sequence=initial_sequence, root_id=root_id, node_id=initial_id
        )
        rotated_tree.add_sequence(
            sequence=convergence_sequence, parent_id=root_id, node_id=conv_id
        )
        node_list = []
        node_weights = []
        for ii, obs in enumerate(self.obstacle_environment):
            # It is with respect to the close-obstacles
            # -- it_obs ONLY to use in obstacle_list (whole)
            # => the null matrix should be based on the normal
            # direction (not reference, right?!)
            reference_vector = obs.get_reference_direction(
                position, in_global_frame=True
            )

            # Null matrix (zero-vector) and reference direction should be pointing
            # towards the wall - the initial reference direction is pointing towards
            # the reference
            if obs.is_boundary:
                reference_vector = (-1) * reference_vector
                base = normal_orthogonal_matrix[:, :, ii]
            else:
                base = (-1) * normal_orthogonal_matrix[:, :, ii]

            # Note that the inv_gamma_weight was prepared for the multiboundary
            # environment through the reference point displacement (see 'loca_reference_point')

            cont_weight = self.get_smooth_continuation_weight(
                gamma=gamma_array[ii],
                vector_convergence=convergence_sequence.get_end_vector(),
                vector_reference=reference_vector,
                base=base,
            )

            if cont_weight <= 0:
                continue

            vector_convergence_tangent = self.get_pseudo_tangent(
                vector_convergence=convergence_sequence.get_end_vector(),
                vector_reference=reference_vector,
                base=base,
            )

            if self.convergence_radius > math.pi * 0.5:
                # Add intermediate convergence
                intermediate_vector = self.get_intermediate_convergence_direction(
                    vector_convergence=convergence_sequence.get_end_vector(),
                    vector_reference=reference_vector,
                    base=base,
                )

                parent_id = (2, ii)
                rotated_tree.add_node(
                    node_id=parent_id,
                    parent_id=conv_id,
                    direction=intermediate_vector,
                )
            else:
                parent_id = conv_id

            rotated_tree.add_node(
                node_id=ii,
                parent_id=parent_id,
                direction=vector_convergence_tangent,
            )

            node_weights.append(cont_weight * importance_weights[ii])
            node_list.append(ii)

        if convergence_default:
            # In the surrounding of the obstacle fall back to the convergence,
            # rather than the initial dynamics
            node_list.append(conv_id)
            node_weights.append(sum(importance_weights) - sum(node_weights))

        # Add initial weight
        node_list.append(initial_id)
        node_weights.append(1 - sum(node_weights))

        averaged_sequence = rotated_tree.reduce_weighted_to_sequence(
            node_list=node_list, weights=node_weights
        )

        return averaged_sequence

    def avoid(
        self,
        position: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None,
        obstacle_list: Optional[list] = None,
        convergence_velocity: Optional[np.ndarray] = None,
        sticky_surface: bool = True,
        convergence_radius: Optional[float] = None,
    ) -> np.ndarray:
        """Obstacle avoidance based on 'local' rotation and the directional weighted mean.

        Parameters
        ----------
        position : array of the position at which the modulation is performed
            float-array of (dimension,)
        initial_velocity : Initial velocity which is modulated
        obstacle_list :
        gamma_distance : factor for the evaluation of the proportional gamma

        Return
        ------
        modulated_velocity : array-like of shape (n_dimensions,)
        """
        if initial_velocity is None:
            initial_velocity = self.initial_dynamics.evaluate(position)

        if convergence_radius is None:
            convergence_radius = self.convergence_radius
        else:
            self.convergence_radius = convergence_radius

        if obstacle_list is None:
            if self.obstacle_environment is None:
                raise ValueError("Need environment information.")
            obstacle_list = self.obstacle_environment
        else:
            self.obstacle_environment = obstacle_list

        n_obstacles = len(obstacle_list)
        if not n_obstacles:  # No obstacles in the environment
            return initial_velocity

        norm_initial = LA.norm(initial_velocity)

        if hasattr(obstacle_list, "update_relative_reference_point"):
            # TODO: directly return gamma_array
            obstacle_list.update_relative_reference_point(position=position)

        dimension = position.shape[0]

        gamma_array = self.compute_obstacle_gamma(position, obstacle_list)
        ind_obs = np.logical_and(gamma_array < self.cut_off_gamma, gamma_array >= 1)

        if not any(ind_obs):
            return initial_velocity

        n_obs_close = np.sum(ind_obs)

        gamma_array = gamma_array[ind_obs]  # Only keep relevant ones
        # gamma_proportional = np.zeros((n_obs_close))

        normal_orthogonal_matrix = self.compute_normal_tensor(
            position, obstacle_list, ind_obs
        )

        weights = compute_weights(gamma_array)
        inv_gamma_weight = get_weight_from_inv_of_gamma(gamma_array)

        relative_velocity = get_relative_obstacle_velocity(
            position=position,
            obstacle_list=obstacle_list,
            ind_obstacles=ind_obs,
            gamma_list=gamma_array,
            E_orth=normal_orthogonal_matrix,
            weights=weights,
        )

        initial_velocity = initial_velocity - relative_velocity
        if not LA.norm(initial_velocity):
            return initial_velocity + relative_velocity

        # rotated_directions = [None] * n_obs_close
        rotated_directions = np.zeros((dimension, n_obs_close))
        for it, it_obs in enumerate(np.arange(n_obstacles)[ind_obs]):
            # It is with respect to the close-obstacles
            # -- it_obs ONLY to use in obstacle_list (whole)
            # => the null matrix should be based on the normal
            # direction (not reference, right?!)
            reference_dir = obstacle_list[it_obs].get_reference_direction(
                position, in_global_frame=True
            )

            # Null matrix (zero-vector) and reference direction should be pointing
            # towards the wall - the initial reference direction is pointing towards
            # the reference
            if obstacle_list[it_obs].is_boundary:
                reference_dir = (-1) * reference_dir
                # null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)
                null_matrix = normal_orthogonal_matrix[:, :, it]

            else:
                # reference_dir = (-1) * reference_dir
                null_matrix = (-1) * normal_orthogonal_matrix[:, :, it]
                # null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)

            if np.dot(reference_dir, null_matrix[:, 0]) < 0:
                # TODO: this check should not be necessary with proper obstacle definition
                null_matrix = null_matrix * (-1)

            # Convergence direcctions can be local for certain obstacles
            # / convergence environments
            if convergence_velocity is None:
                if self.convergence_dynamics is None:
                    convergence_velocity = obstacle_list.get_convergence_direction(
                        position=position, it_obs=it_obs
                    )
                else:
                    convergence_velocity = self.convergence_dynamics.evaluate(position)

            # if not (conv_vel_norm := LA.norm(convergence_velocity)):
            if not LA.norm(convergence_velocity):
                # Zero value
                # base = DirectionBase(matrix=null_matrix)
                base = null_matrix

                rotated_directions[:, it] = initial_velocity
                continue

            if hasattr(obstacle_list, "convergence_radiuses") and len(
                obstacle_list.convergence_radiuses
            ):
                # Custom convergence radius per obstacle
                convergence_radius = obstacle_list.convergence_radiuses[it_obs]

            # Note that the inv_gamma_weight was prepared for the multiboundary
            # environment through the reference point displacement (see 'loca_reference_point')
            rotated_directions[:, it] = self.directional_convergence_summing(
                convergence_vector=convergence_velocity,
                reference_vector=reference_dir,
                # weight=inv_gamma_weight[it],
                gamma=gamma_array[it],
                nonlinear_velocity=initial_velocity,
                base=null_matrix,
                convergence_radius=convergence_radius,
            )

        base = get_orthogonal_basis(initial_velocity)

        rotated_velocity = get_directional_weighted_sum(
            null_direction=base[:, 0],
            weights=weights,
            directions=rotated_directions,
        )

        if sticky_surface:
            rotated_velocity = self._limit_magnitude(
                modulated_velocity=rotated_velocity,
                initial_magintude=LA.norm(initial_velocity),
                gammas=gamma_array,
                normals=normal_orthogonal_matrix[:, 0, :],
                weights=weights,
            )

            rotated_velocity = rotated_velocity + relative_velocity

        else:
            # Get averaged normal
            averaged_normal = np.sum(
                # normal_dirs * np.tile(weights, (dimension, 1)), axis=1
                normal_orthogonal_matrix[:, 0, :] * np.tile(weights, (dimension, 1)),
                axis=1,
            )

            rotated_velocity = self.compute_safe_magnitude(
                rotated_velocity=rotated_velocity,
                initial_norm=np.linalg.norm(initial_velocity),
                averaged_normal=averaged_normal,
                gamma=np.min(gamma_array),
            )

            rotated_velocity = rotated_velocity + relative_velocity

            if velocity_norm := LA.norm(rotated_velocity):
                rotated_velocity = rotated_velocity / velocity_norm * norm_initial

        # TODO: check maximal magnitude (in dynamic environments); i.e. see paper
        return rotated_velocity

    @staticmethod
    def _limit_magnitude(
        modulated_velocity, initial_magintude, gammas, normals, weights
    ):
        """Returns scaled velocity such that zero on the surface of an obstacle."""
        # magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)
        min_dot_prod = 0
        it_min = None
        for ii in range(len(gammas)):
            # Under the assumption of normalized input vectors
            dot_prod = np.dot(modulated_velocity, normals[:, ii])

            if dot_prod < min_dot_prod:
                min_dot_prod = dot_prod
                it_min = ii

        if it_min is None:
            return modulated_velocity * initial_magintude

        modulated_velocity = (
            modulated_velocity
            * initial_magintude
            * (1 + (1 / gammas[it_min]) * min_dot_prod)
        )

        return modulated_velocity

    @staticmethod
    def compute_safe_scaling(
        velocity: Vector,
        averaged_normal: Vector,
        gamma: float,
        dot_scaling: float = 0.8,
        power_root: float = 3,
    ) -> float:
        # TODO: very similar to 'compute_safe_magnitude' -> can they be merged (?!)
        if not (rotated_norm := np.linalg.norm(velocity)):
            return 1.0

        dot_product = np.dot(velocity, averaged_normal)
        normal_norm = np.linalg.norm(averaged_normal)

        if dot_product > 0 or np.isclose(normal_norm, 0):
            return 1.0

        elif gamma <= 1.0:
            if dot_product < (-1e-3):
                return 0.0
            return 1.0

        # At this stage, the dot product is negative
        power_factor = (1.0 / (gamma - 1) * normal_norm) ** (1.0 / power_root)
        return ((1.0 + dot_scaling * dot_product)) ** power_factor

    @staticmethod
    def compute_safe_magnitude(
        rotated_velocity: Vector,
        initial_norm: float,
        averaged_normal: Vector,
        gamma: float,
        dot_scaling: float = 0.8,
        power_root: float = 3,
    ) -> Vector:
        if not (rotated_norm := np.linalg.norm(rotated_velocity)):
            return rotated_velocity

        dot_product = np.dot(rotated_velocity, averaged_normal)
        normal_norm = np.linalg.norm(averaged_normal)

        # if dot_product < 0:
        #     print()
        #     print("dot_product", dot_product)
        #     print("gamma", gamma)

        if dot_product > 0 or normal_norm == 0:
            scaling = 1.0

        elif gamma <= 1:
            if dot_product < (-1e-3):
                return np.zeros_like(rotated_velocity)
            scaling = 1.0

        else:
            # Remember that at this stage, the dot product is negative
            power_factor = (1.0 / (gamma - 1) * normal_norm) ** (1.0 / power_root)
            scaling = ((1.0 + dot_scaling * dot_product)) ** power_factor

        if scaling > 1.0 or np.isnan(scaling):
            # TODO: remove debug check
            raise ValueError()
            # breakpoint()

        return rotated_velocity / rotated_norm * initial_norm * scaling

    @staticmethod
    def _get_directional_deviation_weight(
        weight: float,
        weight_deviation: float,
        power_factor: float = 3.0,
    ) -> float:
        """This 'smooth'-weighting needs to be done, in order to have a smooth vector-field
        which can be approximated by the nonlinear DS."""
        if weight_deviation <= 0:
            return 0
        elif weight_deviation >= 1 or weight >= 1:
            return 1
        else:
            return weight ** (1.0 / (weight_deviation * power_factor))

    @staticmethod
    def _get_nonlinear_inverted_weight(
        inverted_conv_rotated_norm: float,
        inverted_nonlinear_norm: float,
        inv_convergence_radius: float,
        weight: float,
    ) -> float:
        """Returns modified weight which ensure continuous transformation of direction."""
        # Potentially set to 0 when approaching radius, since this would allow continuity.

        if inverted_nonlinear_norm <= inv_convergence_radius:
            return 0

        if inverted_conv_rotated_norm <= inv_convergence_radius:
            return weight

        delta_nonl = inverted_nonlinear_norm - inv_convergence_radius
        delta_conv = inverted_conv_rotated_norm - inv_convergence_radius
        # weight_nonl = weight * delta_nonl/(delta_nonl + delta_conv)
        return weight * delta_nonl / (delta_nonl + delta_conv)

    @staticmethod
    def _get_projection_of_inverted_convergence_direction(
        inv_conv_rotated: UnitDirection,
        inv_nonlinear: UnitDirection,
        inv_convergence_radius: UnitDirection,
    ) -> UnitDirection:
        """Returns projected converenge direction based in the (normal)-direction space.

        The method only projects when crossing is actually needed.
        It checks the connection points on the direction-circle from dir_nonlinear to
        dir_convergence, and does following:
        - [dir_nolinear in convergence_radius] => [weight=0] => return dir_nonlinear
        - [No Intersection] => [weight=0] => no rotation => return dir_nonlinear
        - [dir_convergence in convergence_radius] => [weight=1] => return the intersection point
        - [two intersection points] => return relative secant

        Note: This method has the danger that the modulation only ever happens if the rotation is
        going in the correct direction, i.e., it is in correct part of the circle.
        It is hence important to activate the rotation early enough.

        Parameters
        ----------
        dir_conv_rotated
        # delta_dir_conv: Difference between dir_conv_rotated & delta_dir_conv (in normal space
        inv_convergence_radius: the inverted convergence radius

        Returns
        -------
        Convergence value rotated.
        """
        # TODO: remove
        warnings.warn("This function is outdated.")

        if inv_nonlinear.norm() <= inv_convergence_radius:
            # Already converging
            return inv_conv_rotated

        if inv_conv_rotated.norm() <= inv_convergence_radius:
            point = get_intersection_with_circle(
                start_position=inv_conv_rotated.as_angle(),
                direction=(inv_conv_rotated - inv_nonlinear).as_angle(),
                radius=inv_convergence_radius,
                only_positive=True,
            )
            # sectant_dist = LA.norm(point - inv_conv_rotated.as_angle())
            # Since $w_cp = 1$ it we directly return the value
            return UnitDirection(inv_conv_rotated.base).from_angle(point)

        # Both points are returned since we need to measure the 'sequente'
        points = get_intersection_with_circle(
            start_position=inv_conv_rotated.as_angle(),
            direction=(inv_conv_rotated - inv_nonlinear).as_angle(),
            radius=inv_convergence_radius,
            only_positive=False,
        )

        if points is None:
            # No intersection => we are outside
            return inv_conv_rotated

        if False:
            # => Don't do this check anymore since its done at the follow up weight

            # Check if the two intersection points are inbetween the two directions
            # Any of the points can be chosen for the u check, since either both
            # or none points are between / outside
            dir_vector_conv = (inv_conv_rotated - inv_nonlinear).as_angle()
            dir_vector_point = points[:, 0] - inv_nonlinear.as_angle()

            if np.dot(dir_vector_conv, dir_vector_point) < 0 or np.dot(
                dir_vector_conv, dir_vector_point
            ) > np.dot(dir_vector_conv, dir_vector_conv):
                # Intersections are behind or in front of both points with respect to the
                # two angles
                return inv_conv_rotated

        w_cp = LA.norm(points[:, 0] - points[:, 1]) / LA.norm(
            inv_conv_rotated.as_angle() - points[:, 0]
        )

        if w_cp > 1:
            # TODO: remove theck
            breakpoint()

        angle_inv_conv_proj = (
            # (1 - w_cp) * points[:, 0] + (w_cp) * points[:, 1]
            (1 - w_cp) * points[:, 0]
            + w_cp * inv_conv_rotated.as_angle()
        )
        inv_conv_proj = UnitDirection(inv_conv_rotated.base).from_angle(
            angle_inv_conv_proj
        )

        if LA.norm(inv_conv_proj.as_angle()) > math.pi:
            # -> DEBUG
            raise NotImplementedError(
                f"Unexpected value of {LA.norm(inv_conv_proj.as_angle())}"
            )

        return inv_conv_proj

    @staticmethod
    def _get_projected_nonlinear_velocity(
        dir_conv_rotated: UnitDirection,
        dir_nonlinear: UnitDirection,
        weight: float,
        convergence_radius: float = math.pi / 2,
    ) -> UnitDirection:
        """Invert the directional-circle-space and project the nonlinear velocity to approach
        the linear counterpart.

        Parameters
        ----------
        dir_conv_rotated: rotated convergence direction which guides the nonlinear
        dir_nonlinear: nonlinear direction which is pulled towards the convergence direction.
        weight: initial weight - which is taken into the calculation
        convergence_radius: radius of circle to which direction is pulled
            towards (e.g. pi/2 = tangent)

        Returns
        -------
        nonlinar_conv: Projected nonlinear velocity to be aligned with convergence velocity
        """
        # if True:
        #     raise Exception("Is this still used?")
        # TODO: remove
        # Invert matrix to get smooth summing.
        inv_nonlinear = dir_nonlinear.invert_normal()

        # Only project when 'outside the radius'
        inv_convergence_radius = math.pi - convergence_radius
        if inv_nonlinear.norm() <= inv_convergence_radius:
            return dir_nonlinear

        inv_conv_rotated = dir_conv_rotated.invert_normal()
        weight_nonl = RotationalAvoider._get_nonlinear_inverted_weight(
            inv_conv_rotated.norm(),
            inv_nonlinear.norm(),
            inv_convergence_radius,
            weight=weight,
        )

        if not weight_nonl:  # Zero value
            return inv_nonlinear.invert_normal()

        # TODO: integrate this function here
        inv_conv_proj = (
            RotationalAvoider._get_projection_of_inverted_convergence_direction(
                inv_conv_rotated=inv_conv_rotated,
                inv_nonlinear=inv_nonlinear,
                inv_convergence_radius=inv_convergence_radius,
            )
        )

        inv_nonlinear_conv = (
            weight_nonl * inv_conv_proj + (1 - weight_nonl) * inv_nonlinear
        )

        return inv_nonlinear_conv.invert_normal()

    @staticmethod
    def _get_projected_velocity(
        dir_convergence_tangent: UnitDirection,
        dir_initial_velocity: UnitDirection,
        weight: float,
        convergence_radius: float = math.pi / 2,
    ) -> UnitDirection:
        """Invert the directional-circle-space and project the nonlinear velocity to approach
        the linear counterpart.
        """
        if convergence_radius != dir_convergence_tangent.norm():
            dir_convergence_tangent = copy.deepcopy(
                convergence_radius
                / dir_convergence_tangent.norm()
                * dir_convergence_tangent
            )

        # Check if the velocity is already going in the correct direction
        angle_tangent = dir_convergence_tangent.as_angle()
        delta_angle = dir_initial_velocity.as_angle() - angle_tangent

        delta_norm = LA.norm(delta_angle)
        if not delta_norm:
            # Tangent and initial are identical vector, hence no interpolation needed
            return dir_initial_velocity

        dot_prod_direction = np.dot(angle_tangent, delta_angle) / (
            LA.norm(angle_tangent) * delta_norm
        )

        # Reduce the tail effect -> already going in the correct direction
        if dot_prod_direction > 0:
            weight = weight * (1 - dot_prod_direction)

        dir_convergence = (
            weight * dir_convergence_tangent + (1 - weight) * dir_initial_velocity
        )

        if weight < 0 or weight > 1:
            breakpoint()
            raise ValueError("Unexpected weight value... -> DEBUG")

        return dir_convergence

    @staticmethod
    def get_projected_tangent_from_vectors(
        initial_vector: Vector,
        normal: Vector,
        reference: Vector,
        convergence_radius: float = math.pi / 2,
    ) -> Vector:
        """Returns the (pseudo)-tangent for the avoidance."""

        # breakpoint()
        if np.dot(initial_vector, (-1) * normal) > np.cos(convergence_radius):
            base = get_orthogonal_basis((-1) * normal)
            angle_ref = get_angle_from_vector(reference, base=base)
            tmp_radius = convergence_radius

        else:
            # Switch the circle-basis
            # Note: This switch is continuous as it happens when the initial_vector
            # is on the surface of the circle-boundary
            base = get_orthogonal_basis(normal)
            angle_ref = get_angle_from_vector((-1) * reference, base=base)
            tmp_radius = math.pi - convergence_radius

        angle_init = get_angle_from_vector(initial_vector, base=base)

        if not LA.norm(angle_init - angle_ref):
            return initial_vector

        surface_angle = get_intersection_with_circle(
            start_position=angle_ref,
            direction=(angle_init - angle_ref),
            radius=tmp_radius,
            intersection_type=IntersectionType.FAR,
        )

        if surface_angle is None:
            raise ValueError("No tangent found.")

        # breakpoint()

        return get_vector_from_angle(surface_angle, base=base)

    @staticmethod
    def get_tangent_convergence_direction_from_unit_direction(
        dir_convergence: UnitDirection,
        dir_reference: UnitDirection,
        convergence_radius: float = math.pi / 2,
    ) -> UnitDirection:
        """Projects the reference direction onto the surface

        Similar behavior as 'get_projected_tangent_from_vectors' but angle/unit direction input.
        This function may be removed in the future (!).
        """

        if not (dir_convergence - dir_reference).norm():
            # What if they are aligned -> for now return default vector
            base_angle = np.zeros(dir_convergence.as_angle().shape)
            base_angle[0] = convergence_radius
            return UnitDirection(dir_reference.base).from_angle(base_angle)

        surface_angle = get_intersection_with_circle(
            start_position=dir_reference.as_angle(),
            direction=(dir_convergence - dir_reference).as_angle(),
            radius=convergence_radius,
            only_positive=True,
        )

        if surface_angle is None:
            raise ValueError(
                "No intersection with surface found with"
                + f"radius={convergence_radius}."
            )

        return UnitDirection(dir_reference.base).from_angle(surface_angle)

    @staticmethod
    def get_tangent_convergence_direction(
        dir_convergence: np.ndarray,
        dir_reference: np.ndarray,
        convergence_radius: float = np.pi * 0.5,
    ) -> np.ndarray:
        """The input is expected in direction space with respect to the negative normal."""

        if not np.linalg.norm(dir_convergence - dir_reference):
            # What if they are aligned -> for now return default vector
            base_angle = np.zeros_like(dir_convergence)
            base_angle[0] = convergence_radius
            return base_angle

        surface_angle = get_intersection_with_circle(
            start_position=dir_reference,
            direction=(dir_convergence - dir_reference),
            radius=convergence_radius,
            # only_positive=True,
            intersection_type=IntersectionType.FAR,
        )

        if surface_angle is None:
            raise ValueError(
                "No intersection with surface found with"
                + f"radius={convergence_radius}."
            )

        return surface_angle

    @staticmethod
    def _get_rotated_convergence_direction(
        weight: float,
        convergence_radius: float,
        convergence_vector: np.ndarray,
        reference_vector: np.ndarray,
        base: np.ndarray,
    ) -> UnitDirection:
        """Rotates the convergence vector according to given input and basis"""

        dir_reference = UnitDirection(base).from_vector(reference_vector)
        dir_convergence = UnitDirection(base).from_vector(convergence_vector)

        if dir_convergence.norm() >= convergence_radius:
            # Initial velocity 'dir_convergence' already pointing away from obstacle
            return dir_convergence

        # Find intersection a with radius of pi/2 inside the tangent radius,
        # i.e. vectorfield towards obstacle [no-tail-effect]
        # Do the math in the angle space
        delta_dir_conv = dir_convergence - dir_reference

        norm_dir_conv = delta_dir_conv.norm()
        if not norm_dir_conv:  # Zero value
            return None

        angle_tangent = get_intersection_with_circle(
            start_position=dir_reference.as_angle(),
            direction=delta_dir_conv.as_angle(),
            radius=convergence_radius,
        )

        dir_tangent = UnitDirection(base).from_angle(angle_tangent)

        norm_tangent_dist = (dir_tangent - dir_reference).norm()

        # Weight to ensure that:
        weight_deviation = norm_dir_conv / norm_tangent_dist
        w_conv = RotationalAvoider._get_directional_deviation_weight(
            weight, weight_deviation=weight_deviation
        )

        # Weight which ensures continuity at far end
        return w_conv * dir_tangent + (1 - w_conv) * dir_convergence

    @staticmethod
    def get_rotation_weight(
        normal_vector: np.ndarray,
        reference_vector: np.ndarray,
        convergence_vector: np.ndarray,
        gamma_value: float,
        convergence_radius: float = math.pi * 0.5,
        smooth_continuation_power: float = 1.0,
        radius_base: float = math.pi * 0.5,
    ) -> float:
        """Returns smoothing weight, used in nonlinear obstacle avoidance.
        smooth_continuation_power:
        """
        # TODO: the angle calculation could be done one level up (?)
        # Max effect when on the surface
        if gamma_value <= 1.0:
            return 1.0

        if np.dot(convergence_vector, normal_vector) > np.cos(convergence_radius):
            # Switch the circle-basis
            # Note: This switch is continuous as it happens when the initial_vector
            # is on the surface of the circle-boundary
            base = get_orthogonal_basis(normal_vector)
            angle_ref = get_angle_from_vector((-1) * reference_vector, base=base)
            angle_conv = get_angle_from_vector(convergence_vector, base=base)

        else:
            base = get_orthogonal_basis((-1) * normal_vector)
            angle_ref = get_angle_from_vector(reference_vector, base=base)
            angle_conv = get_angle_from_vector(convergence_vector, base=base)

        delta_angle = float(np.linalg.norm(angle_ref - angle_conv))
        # print("delta_angle", delta_angle)
        ref_radius = min(radius_base, convergence_radius) - LA.norm(angle_ref)

        if ref_radius <= delta_angle:
            return 1.0 / gamma_value

        # rotation_power in [0, 1]
        rotation_power = (ref_radius / delta_angle) ** smooth_continuation_power
        if rotation_power < 1.0:
            # Should never happen -> remove after debug
            breakpoint()
            # rotation_power = 1.0

        # print("weight", (1.0 / gamma_value) ** (rotation_power))
        # print("rotation_power", rotation_power)
        return (1.0 / gamma_value) ** (rotation_power)

    def get_smooth_continuation_weight(
        self,
        gamma: float,
        vector_reference: np.ndarray,
        vector_convergence: np.ndarray,
        base: np.ndarray,
    ) -> float:
        """
        dir_reference and dir_convergence are expected to be in the the direction-space, with respect
        to the negative of the normal
        """
        dir_reference = UnitDirection(base).from_vector(vector_reference).as_angle()
        dir_convergence = UnitDirection(base).from_vector(vector_convergence).as_angle()

        weight = 1.0 / gamma

        if weight <= 0.0:
            return 0.0
        if weight >= 1.0:
            return 1.0

        if np.linalg.norm(dir_convergence) >= self.convergence_radius:
            return weight
        # MAYBE: incorporate this in the tangent length(?)
        continuation_weight = np.linalg.norm(dir_convergence - dir_reference)
        radius_factor = self.convergence_radius - np.linalg.norm(dir_reference)
        radius_factor = min(radius_factor, np.pi * 0.5)
        continuation_weight = continuation_weight / radius_factor

        if continuation_weight <= 0:
            return 0.0

        continuation_weight = min(continuation_weight, 1.0)
        continuation_weight = continuation_weight**self.smooth_continuation_power
        weight = weight ** (1.0 / continuation_weight)

        return weight

    def get_pseudo_tangent(
        self,
        vector_convergence: np.ndarray,
        vector_reference: np.ndarray,
        base: np.ndarray,
    ) -> np.ndarray:
        """Input and output are expected in the directional-space with respect to
        the negative normal."""
        dir_convergence = UnitDirection(base).from_vector(vector_convergence)
        if (
            np.linalg.norm(dir_convergence.as_angle()) >= self.convergence_radius
            and not self.tail_rotation
        ):
            return vector_convergence

        dir_reference = UnitDirection(base).from_vector(vector_reference)

        dir_convergence_tangent = RotationalAvoider.get_tangent_convergence_direction(
            dir_convergence=dir_convergence.as_angle(),
            dir_reference=dir_reference.as_angle(),
            convergence_radius=self.convergence_radius,
        )
        # breakpoint()
        return UnitDirection(base).from_angle(dir_convergence_tangent).as_vector()

    def get_intermediate_convergence_direction(
        self,
        vector_convergence: np.ndarray,
        vector_reference: np.ndarray,
        base: np.ndarray,
    ) -> np.ndarray:
        unitdir_convergence = UnitDirection(base).from_vector(vector_convergence)
        unitdir_reference = UnitDirection(base).from_vector(vector_reference)

        dir_intermediate_convergence = RotationalAvoider.get_tangent_convergence_direction(
            dir_convergence=unitdir_convergence.as_angle(),
            dir_reference=unitdir_reference.as_angle(),
            # base=base,
            convergence_radius=math.pi * 0.5,
        )
        unitdir_intermediate = UnitDirection(base).from_angle(
            dir_intermediate_convergence
        )

        return unitdir_intermediate.as_vector()

    def directional_convergence_summing(
        self,
        convergence_vector: np.ndarray,
        reference_vector: np.ndarray,
        base: np.ndarray,
        gamma: float,
        nonlinear_velocity: Optional[np.ndarray] = None,
        convergence_radius: float = math.pi / 2,
    ) -> UnitDirection:
        """Rotating / modulating a vector by using directional space.

        Parameters
        ---------
        convergence_vector: vector (often linearized system) which ensures convergence
        reference_vector: vector towards the center (reference) of an obstacle
        weight: float in the range [0, 1] which gives influence on how important vector 2 is.
        nonlinear_velocity: (optional) the vector-field which converges

        Returns
        -------
        converging_velocity: Weighted summing in direction-space to 'emulate' the modulation.
        """
        # Put int range
        # unitdir_convergence = UnitDirection(base).from_vector(convergence_vector)
        # unitdir_reference = UnitDirection(base).from_vector(reference_vector)

        weight = self.get_smooth_continuation_weight(
            gamma, convergence_vector, reference_vector, base
        )

        if weight <= 0:
            return nonlinear_velocity

        vector_convergence_tangent = self.get_pseudo_tangent(
            convergence_vector, reference_vector, base=base
        )

        # This is now replacing the 'general_weighted_sum'
        # as it takes better into account the history
        direction_tree = VectorRotationTree()
        direction_tree.set_root(root_idx=-1, direction=convergence_vector)
        direction_tree.add_node(node_id=0, parent_id=-1, direction=nonlinear_velocity)
        direction_tree.add_node(
            node_id=1,
            parent_id=-1,
            direction=base[:, 0],
        )

        if convergence_radius > math.pi * 0.5:
            # Add intermediate convergence
            intermediate_vector = self.get_intermediate_convergence_direction(
                convergence_vector, reference_vector, base
            )

            direction_tree.add_node(
                node_id=2,
                parent_id=1,
                direction=intermediate_vector,
            )
            parent_id = 2
        else:
            parent_id = 1

        direction_tree.add_node(
            node_id=3,
            parent_id=parent_id,
            direction=vector_convergence_tangent,
        )

        averaged_direction = direction_tree.get_weighted_mean(
            node_list=[0, 3], weights=[(1 - weight), weight]
        )

        # Directional summing -> outdated
        # rotated_velocity = get_directional_weighted_sum(
        #         null_direction=unitdir_convergence.as_vector(),
        #         weights=np.array([weight, (1 - weight)]),
        #         directions=np.vstack(
        #             (
        #                 unitdir_convergence_tangent.as_vector(),
        #                 unitdir_initial.as_vector(),
        #             )
        #         ).T,
        #     )
        # averaged_direction = rotated_velocity

        # return rotated_velocity
        return averaged_direction
