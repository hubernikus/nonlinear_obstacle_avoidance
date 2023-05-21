"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
from typing import Optional, Protocol

# import itertools as it

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_with_circle, CircleIntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.states import Pose

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.datatypes import Vector
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.geometry import get_intersection_of_obstacles
from nonlinear_avoidance.multi_obstacle_avoider import plot_multi_obstacle

# TODO:
#   - smoothing to ensure consistency at convergence limit, i.e., add lambda to each branch
#   - use reference direction to do correct decomposition of reference matrix


def get_intersection_with_ellipse(
    position,
    direction,
    ellipse: Ellipse,
    in_global_frame: bool = False,
    intersection_type=CircleIntersectionType.CLOSE,
) -> Optional[np.ndarray]:
    # Depreciated -> this has been integrated in the EllipseWithAxes class.
    if in_global_frame:
        # Currently only implemented for ellipse
        position = ellipse.pose.transform_position_to_relative(position)
        direction = ellipse.pose.transform_direction_to_relative(direction)

    # Stretch according to ellipse axes (radius)
    rel_pos = position / ellipse.axes_length
    rel_dir = direction / ellipse.axes_length

    # Intersection with unit circle
    surface_rel_pos = get_intersection_with_circle(
        start_position=rel_pos,
        direction=rel_dir,
        radius=0.5,
        intersection_type=intersection_type,
    )

    if surface_rel_pos is None:
        return None

    # Relative
    surface_pos = surface_rel_pos * ellipse.axes_length

    if in_global_frame:
        return ellipse.pose.transform_position_from_relative(surface_pos)

    else:
        return surface_pos


class MultiEllipseObstacle(Obstacle):
    def __init__(self):
        self._obstacle_list = []

        self._root_idx: Optional[int] = None
        self._parent_list: list[Optional[int]] = []
        self._children_list: list[list[int]] = []

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_idx(self) -> int:
        return self._root_idx

    def get_pose(self) -> Pose:
        if hasattr(self, "pose"):
            return self.pose
        else:
            return Pose.create_trivial(self._obstacle_list[0].dimension)

    # def get_obstacle_list(self) -> ObstacleContainer:
    #     return self._obstacle_list

    def get_component(self, idx_obs) -> Obstacle:
        return self._obstacle_list[idx_obs]

    def set_root(self, obs_id: int):
        if self._root_idx:
            raise NotImplementedError("Make sure to delete first.")
        self._root_idx = obs_id
        self._parent_list[obs_id] = -1

    def set_parent(self, obs_id: int, parent_id: int):
        # This should go automatically at run-time
        if self._parent_list[obs_id]:
            raise NotImplementedError("Make sure to delete first.")

        self._parent_list[obs_id] = parent_id
        self._children_list[parent_id].append(obs_id)

        # Set reference point
        intersection = get_intersection_of_obstacles(
            self._obstacle_list[obs_id], self._obstacle_list[parent_id]
        )

        self._obstacle_list[obs_id].set_reference_point(
            intersection, in_global_frame=True
        )

    def append(self, obstacle: Obstacle) -> None:
        self._obstacle_list.append(obstacle)
        self._children_list.append([])
        self._parent_list.append(None)

    def delete_item(self, obs_id: int):
        raise NotImplementedError()

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        return self._parent_list[idx_obs]

    def get_linearized_velocity(self, position):
        raise NotImplementedError()

    @staticmethod
    def get_normalized_tangent_component(
        vector: Vector, normal: Vector, reference: Vector
    ) -> Vector:
        """This function has similar properties as the
        'RotationalAvoider.get_projected_tangent_from_vectors'
        but is limited to a convergence-circle radius of pi/2."""
        basis = get_orthogonal_basis(normal)
        basis[:, 0] = reference

        tmp_tangent = LA.pinv(basis) @ vector
        tmp_tangent[0] = 0  # only in tangent plane
        tangent = basis @ tmp_tangent

        if not (norm_tangent := LA.norm(tangent)):
            raise Exception()

        return tangent / norm_tangent

    def _get_tangent_tree(self):
        pass

    def get_gamma(self, position: Vector, in_global_frame: bool = True) -> float:
        if not in_global_frame:
            raise NotImplementedError("For now we expect global frame..")

        gamma_values = np.zeros(self.n_components)
        for ii, obs in enumerate(self._obstacle_list):
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)

        return min(gamma_values)

    def is_inside(self, position: Vector, in_global_frame: bool = True) -> bool:
        return self.get_gamma(position, in_global_frame) <= 1

    def get_normal_direction(self, position):
        pass

    def weights(self):
        pass
