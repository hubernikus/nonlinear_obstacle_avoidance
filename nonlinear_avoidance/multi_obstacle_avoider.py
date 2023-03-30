"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
from typing import Optional, Protocol

import numpy as np
from numpy import linalg

from vartools.math import get_intersection_with_circle, CircleIntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem, LinearSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.nonlinear_rotation_avoider import ObstacleConvergenceDynamics
from nonlinear_avoidance.datatypes import Vector


class HierarchyObstacle(Protocol):
    # + all methods of a general obstacle(?)
    @property
    def n_components(self) -> int:
        ...

    @property
    def root_idx(self) -> int:
        ...

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        ...

    def get_component(self, idx_obs: int) -> Obstacle:
        ...


class MultiObstacleAvoider:
    """Obstacle Avoider which can take a 'multi-obstacle' as an input."""

    # TODO: future implementation should include multiple obstacles
    def __init__(
        self,
        obstacle: HierarchyObstacle,
        initial_dynamics: Optional[DynamicalSystem] = None,
        convergence_dynamics: Optional[ObstacleConvergenceDynamics] = None,
        convergence_radius: float = math.pi * 5e-2,
        smooth_continuation_power: float = 0.1,
    ):
        if initial_dynamics is not None:
            self.initial_dynamics = initial_dynamics

        self.convergence_dynamics = convergence_dynamics

        self.convergence_radius = convergence_radius
        self.smooth_continuation_power = smooth_continuation_power

        self.obstacle = obstacle

        # An ID number which does not co-inside with the obstacle
        self._BASE_VEL_ID = -1
        self.gamma_power_scaling = 0.5

        self._tangent_tree = VectorRotationTree()

    @property
    def n_components(self) -> int:
        return self.obstacle.n_components

    def evaluate(self, position: Vector) -> Vector:
        velocity = self.initial_dynamics.evaluate(position)
        # So far the convergence direction is only about the root-obstacle
        # in the future, this needs to be extended such that the rotation is_updating
        # ensured to be smooth (!)
        if self.convergence_dynamics is None:
            convergence_direction = velocity
        else:
            convergence_direction = (
                self.convergence_dynamics.evaluate_convergence_around_obstacle(
                    position,
                    obstacle=self.obstacle.get_component(self.obstacle.root_idx),
                )
            )

        return self.get_tangent_direction(position, velocity, convergence_direction)

    def get_tangent_direction(
        self,
        position: Vector,
        velocity: Vector,
        linearized_velocity: Vector,
        # linearized_velocity: Optional[Vector] = None,
    ) -> Vector:
        # if linearized_velocity is None:
        #     root_obs = self.obstacle.get_component(self.obstacle.root_id)

        #     base_velocity = self.get_linearized_velocity(
        #         # obstacle_list[self._root_id].get_reference_point(in_global_frame=True)
        #         root_obs.get_reference_point(in_global_frame=True)
        #     )
        # else:
        if not linalg.norm(velocity):
            return velocity

        base_velocity = linearized_velocity

        gamma_values = np.zeros(self.obstacle.n_components)
        for ii in range(self.obstacle.n_components):
            obs = self.obstacle.get_component(ii)
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)

        gamma_weights = compute_weights(gamma_values)

        # Evaluate rotation weight, to ensure smoothness in space (!)
        idx_root = self.obstacle.root_idx
        normal = self.obstacle.get_component(idx_root).get_normal_direction(
            position, in_global_frame=True
        )
        reference = self.obstacle.get_component(idx_root).get_reference_direction(
            position, in_global_frame=True
        )

        rotation_weight = RotationalAvoider.get_rotation_weight(
            normal_vector=normal,
            reference_vector=reference,
            convergence_vector=linearized_velocity,
            convergence_radius=self.convergence_radius,
            gamma_value=min(gamma_values),
            smooth_continuation_power=self.smooth_continuation_power,
        )

        if not (gamma_sum := sum(gamma_weights)) or not rotation_weight:
            return velocity

        gamma_weights = gamma_weights / gamma_sum * rotation_weight

        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_idx=self._BASE_VEL_ID,
            direction=velocity,
        )
        self._tangent_tree.add_node(
            parent_id=self._BASE_VEL_ID,
            node_id=self.obstacle.root_idx,
            direction=base_velocity,
        )

        # The base node should be equal to the (initial velocity)
        node_list = [self._BASE_VEL_ID]

        for obs_id in range(self.obstacle.n_components):
            if gamma_weights[obs_id] <= 0:
                continue

            node_list.append((obs_id, obs_id))
            self._update_tangent_branch(position, obs_id, base_velocity)

        weights = (
            gamma_weights[gamma_weights > 0]
            * (1 / np.min(gamma_values)) ** self.gamma_power_scaling
        )

        # Remaining weight to the initial velocity
        weights = np.hstack(([1 - np.sum(weights)], weights))
        weighted_tangent = self._tangent_tree.get_weighted_mean(
            node_list=node_list, weights=weights
        )
        return weighted_tangent

    def _update_tangent_branch(
        self,
        position: Vector,
        obs_id: int,
        base_velocity: np.ndarray,
    ) -> None:
        # TODO: predict at start the size (slight speed up)
        # normal_directions: list[Vector] = []
        # reference_directions: list[Vector] = []
        surface_points: list[Vector] = [position]
        parents_tree: list[int] = [obs_id]

        obs = self.obstacle.get_component(obs_id)
        normal_directions = [obs.get_normal_direction(position, in_global_frame=True)]
        reference_directions = [
            obs.get_reference_direction(position, in_global_frame=True)
        ]

        while parents_tree[-1] != self.obstacle.root_idx:
            obs = self.obstacle.get_component(parents_tree[-1])

            new_id = self.obstacle.get_parent_idx(parents_tree[-1])
            if new_id is None:
                # TODO: We should not reach this?! -> remove(?)
                breakpoint()
                break

            if len(parents_tree) > 10:
                # TODO: remove this debug check
                raise Exception()

            parents_tree.append(new_id)

            obs_parent = self.obstacle.get_component(new_id)
            ref_dir = obs.get_reference_point(in_global_frame=True) - surface_points[-1]

            # intersection = get_intersection_with_ellipse(
            #     surface_points[-1], ref_dir, obs_parent, in_global_frame=True
            # )
            intersection = obs_parent.get_intersection_with_surface(
                surface_points[-1], ref_dir, in_global_frame=True
            )

            if intersection is None:
                # TODO: This should probably never happen -> remove?
                # but for now easier to debug / catch (other) errors early
                breakpoint()
                raise Exception()

            surface_points.append(intersection)

            normal_directions.append(
                obs_parent.get_normal_direction(intersection, in_global_frame=True)
            )

            reference_directions.append(
                obs_parent.get_reference_direction(intersection, in_global_frame=True)
            )

        # Reversely traverse the parent tree - to project tangents
        # First node is connecting to the center-velocity
        tangent = RotationalAvoider.get_projected_tangent_from_vectors(
            base_velocity,
            normal=normal_directions[-1],
            reference=reference_directions[-1],
        )

        self._tangent_tree.add_node(
            node_id=(obs_id, parents_tree[-1]),
            parent_id=self._BASE_VEL_ID,
            direction=tangent,
        )

        # Iterate over all but last one
        for ii in reversed(range(len(parents_tree) - 1)):
            rel_id = parents_tree[ii]

            # Re-project tangent
            tangent = RotationalAvoider.get_projected_tangent_from_vectors(
                tangent,
                normal=normal_directions[ii],
                reference=reference_directions[ii],
            )

            self._tangent_tree.add_node(
                node_id=(obs_id, rel_id),
                parent_id=(obs_id, parents_tree[ii + 1]),
                direction=tangent,
            )


def plot_multi_obstacle(multi_obstacle, ax=None, **kwargs):
    plot_obstacles(
        obstacle_container=multi_obstacle._obstacle_list,
        ax=ax,
        **kwargs,
    )
