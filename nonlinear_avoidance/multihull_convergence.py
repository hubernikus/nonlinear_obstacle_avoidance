"""
Library for the rotation and summing of linear Systems for learning purposes.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from typing import Optional

import numpy as np
from numpy import linalg as LA

# import numpy.typing as npt
# import matplotlib.pyplot as plt  # For debugging only (!)

from vartools.linalg import get_orthogonal_basis

# from vartools.directional_space import DirectionBase
from vartools.directional_space import get_directional_weighted_sum

# from vartools.directional_space import (
#     get_directional_weighted_sum_from_unit_directions,
# )

from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.multiboundary_container import (
    MultiBoundaryContainer,
)


def get_weight_gamma_for_hulls(
    gamma_array: np.ndarray, power_value: float = 1.0
) -> np.ndarray:
    """Returns weight-array based on input of gamma_array."""
    weights = np.maximum(gamma_array - 1, 0)

    if sum_weights := np.sum(weights):
        # Nonzero
        weights = weights / sum_weights
    return weights


def get_desired_radius(
    position: np.ndarray,
    gamma_value: float,
    it_obs: int,
    obstacle_list: MultiBoundaryContainer,
    dotprod_weight: float = 1.0,
    gamma_weight: float = 1.0,
) -> float:
    """Returns desired radius on which boundary a vector has to lie in direction space.
    e.g. radius=pi/2 for a vector to be tangent or pointing away.

    Parameters
    ---------
    gamma_position : The gamma value of this obstacle at the evluation position
    distance : distance between position & projected boundary position

    w_dist : weight-factor applied to the distance
    w_gamma : weight-factor applied to the gamm

    Return
    ------
    desired_radius
    """
    # gamma_value = 1/gamma_value
    radius_single_obstacle = np.pi * 2 ** (-gamma_value)

    # Get all graph-intersection-points with children & parents
    intersection_points = []
    if obstacle_list.get_parent(it_obs) >= 0:
        intersection_points.append(
            obstacle_list.get_parent_intersection(it_child=it_obs)
        )
        # intersection_points.append(obstacle_list.get_convergence_boundary_point(
        # it_child=it_obs, project_on_child=True))

    for it_child in obstacle_list.get_children(it_obs):
        intersection_points.append(
            obstacle_list.get_parent_intersection(it_child=it_child)
        )
        # intersection_points.append(obstacle_list.get_convergence_boundary_point(
        # it_child=it_child, project_on_child=False, it_parent=it_obs))

    boundary_projection = obstacle_list[it_obs].get_intersection_with_surface(
        direction=(position - obstacle_list[it_obs].center_position),
        in_global_frame=True,
    )

    intersection_weight = 1
    for ii in range(len(intersection_points)):
        for oo in range(len(obstacle_list)):
            if oo == it_obs:
                continue

            gamma = obstacle_list[oo].get_gamma(
                boundary_projection, in_global_frame=True
            )
            if gamma <= 1:
                continue

            center = obstacle_list[it_obs].center_position
            delta_pos = position - center
            # if intersection_points[ii] is None:
            #     breakpoint()

            delta_intersec = intersection_points[ii] - center
            dotprod = 1 - np.dot(delta_pos, delta_intersec) / (
                LA.norm(delta_pos) * LA.norm(delta_intersec)
            )

            dotprod_hat = dotprod * dotprod_weight
            gamma_hat = (gamma - 1) * gamma_weight

            new_intersection_weight = dotprod_hat / (dotprod_hat + gamma_hat)
            intersection_weight = min(intersection_weight, new_intersection_weight)

    return radius_single_obstacle * intersection_weight


def multihull_attraction(
    position: np.ndarray,
    initial_velocity: np.ndarray,
    obstacle_list: MultiBoundaryContainer,
    cutoff_gamma_high: float = 1e6,
    cutoff_gamma_low: float = 1e0,  #
    gamma_distance: Optional[float] = None,
) -> np.ndarray:
    """Learned trajectory can be forced to attracto a multi-hull environment.
    Obstacle avoidance is not ensured in order to allow a smooth space
    But the method is inspired by rotational obstacle avoidance (!)

    Parameters
    ----------
    position : array of the position at which the modulation is performed
        float-array of (dimension,)
    initial_velocity : Initial velocity which is modulated
    obstacle_list : Container which contains obstacles
    cutoff_gamma_low: Lower than 1 -> inside wall we're currently not considering

    Return
    ------
    modulated_velocity : array-like of shape (n_dimensions,)
    """
    n_obstacles = len(obstacle_list)
    if not n_obstacles:  # No obstacles in the environment
        return initial_velocity

    dimension = position.shape[0]

    gamma_array = np.zeros((n_obstacles))
    for ii in range(n_obstacles):
        gamma_array[ii] = obstacle_list[ii].get_gamma(position, in_global_frame=True)

    ind_obs = np.logical_and(
        gamma_array < cutoff_gamma_high, gamma_array > cutoff_gamma_low
    )
    if not any(ind_obs):
        return initial_velocity

    n_obs_close = np.sum(ind_obs)
    gamma_array = gamma_array[ind_obs]  # Only keep relevant ones
    inv_gamma_weight = get_weight_from_inv_of_gamma(gamma_array)

    # rotated_velocities = np.zeros((dimension, n_obs_close))
    avoider = RotationalAvoider()
    rotated_directions = np.zeros((dimension, n_obs_close))

    # for it, oo in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
    for it, oo in enumerate(np.arange(n_obstacles)[ind_obs]):
        # It is with respect to the close-obstacles -- oo ONLY to use in obstacle_list (whole)
        reference_dir = obstacle_list[oo].get_reference_direction(
            position, in_global_frame=True
        )
        normal_dir = obstacle_list[oo].get_normal_direction(
            position, in_global_frame=True
        )

        # It seems we only need to switch the normal
        # -> compare with RotationalAvoider
        # reference_dir = (-1) * reference_dir
        # Since the elements are boundary!
        # if obstacle_list[oo].is_boudary:
        normal_dir = (-1) * normal_dir

        normal_orthogonal_matrix = get_orthogonal_basis(normal_dir)

        if obstacle_list[oo].is_boundary:
            reference_dir = (-1) * reference_dir
            normal_orthogonal_matrix = normal_orthogonal_matrix * (-1)

        convergence_velocity = obstacle_list.get_convergence_direction(
            position=position, it_obs=oo
        )

        conv_vel_norm = np.linalg.norm(convergence_velocity)
        if conv_vel_norm:  # Zero value
            # base = DirectionBase(matrix=normal_orthogonal_matrix)
            base = normal_orthogonal_matrix
            rotated_directions[:, it] = initial_velocity

        # position, gamma_value, it_obs, obstacle_list, dotprod_weight=1, gamma_weight=1):
        convergence_radius = get_desired_radius(
            position, gamma_array[it], it_obs=oo, obstacle_list=obstacle_list
        )
        # For testing:
        # convergence_radius = pi*0.9
        # inv_gamma_weight[it] = 1
        # TODO: better weight-function (less doubling with the radius)

        rotated_directions[:, it] = avoider.directional_convergence_summing(
            convergence_vector=convergence_velocity,
            reference_vector=reference_dir,
            weight=inv_gamma_weight[it],
            # weight=1, #
            nonlinear_velocity=initial_velocity,
            # base=DirectionBase(matrix=normal_orthogonal_matrix),
            base=normal_orthogonal_matrix,
            convergence_radius=convergence_radius,
        )

    gamma_weight = get_weight_gamma_for_hulls(gamma_array)

    # rotated_velocities = np.array([vel.as_vector() for vel in rotated_directions])
    # rotated_velocity = get_directional_weighted_sum(
    #     null_direction=initial_velocity,
    #     # base=DirectionBase(vector=initial_velocity),
    #     directions=rotated_velocities,
    #     weights=inv_gamma_weight,
    #     )
    # base = DirectionBase(vector=initial_velocity)
    base = get_orthogonal_basis(initial_velocity)

    rotated_velocity = get_directional_weighted_sum(
        null_direction=base[:, 0], weights=gamma_weight, directions=rotated_directions
    )

    rotated_velocity = rotated_velocity * np.linalg.norm(initial_velocity)

    return rotated_velocity
