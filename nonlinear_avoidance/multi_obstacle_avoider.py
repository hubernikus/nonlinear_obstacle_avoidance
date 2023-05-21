"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
import warnings
from dataclasses import dataclass

from collections import namedtuple
from typing import Optional, Protocol, Hashable

import numpy as np
import numpy.typing as npt
from numpy import linalg

from vartools.math import get_intersection_with_circle, CircleIntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem, LinearSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from nonlinear_avoidance.datatypes import Vector
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.hierarchy_obstacle_protocol import HierarchyObstacle
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ObstacleConvergenceDynamics,
    ConvergenceDynamicsWithoutSingularity,
)
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

NodeType = Hashable
NodeKey = namedtuple("NodeKey", "obstacle component relative_level")
# IterationKey = namedtuple("IterationKey", "obstacle end_component current_component")


def compute_gamma_weights(
    distMeas: npt.ArrayLike,
    distMeas_lowerLimit: float = 1,
    weightPow: float = 1,
) -> np.ndarray:
    """Compute weights based on a distance measure (with no upper limit)"""
    distMeas = np.array(distMeas)
    n_points = distMeas.shape[0]

    critical_points = distMeas <= distMeas_lowerLimit

    if np.sum(critical_points):  # at least one
        if np.sum(critical_points) == 1:
            w = critical_points * 1.0
            return w
        else:
            # TODO: continuous weighting function
            warnings.warn("Implement continuity of weighting function.")
            w = critical_points * 1.0 / np.sum(critical_points)
            return w

    distMeas = distMeas - distMeas_lowerLimit
    w = (1 / distMeas) ** weightPow
    if np.sum(w) <= 1:
        return w

    return w / np.sum(w)  # Normalization


def compute_multiobstacle_relative_velocity(
    position: np.ndarray,
    environment: MultiObstacleContainer,
    cutoff_gamma: float = 10,
) -> np.ndarray:
    if position.shape[0] > 2:
        warnings.warn("No dynamic evaluation for higher dynensions.")
        return np.zeros_like(position)

    # Weights
    n_obstacles = len(environment)
    gammas = np.zeros(n_obstacles)
    for ii, obs in enumerate(environment):
        gammas[ii] = obs.get_gamma(position, in_global_frame=True)

    weights = compute_gamma_weights(gammas, 1.0)
    # weights = compute_weights(gammas, 1.0)
    influence_weight = np.exp(-1.0 * (np.maximum(gammas, 1.0) - 1))

    # weights = influence_weight * weights

    relative_velocity = np.zeros_like(position)
    for ii, obs in enumerate(environment):
        if weights[ii] <= 0:
            continue
        pose = obs.get_pose()

        if hasattr(obs, "twist") and obs.twist is not None:
            relative_velocity = relative_velocity + obs.twist.linear * weights[ii]

            if obs.twist.angular:
                angular_velocity = np.cross(
                    np.array([0, 0, obs.twist.angular]),
                    np.hstack((position - pose.position, 0)),
                )
                relative_velocity = (
                    relative_velocity
                    + weights[ii] * influence_weight[ii] * angular_velocity[:2]
                )

        if hasattr(obs, "deformation_rate"):
            relative_velocity = (
                relative_velocity
                + weights[ii] * (position - pose.position) * obs.deformation_rate
            )

    #     print("velocity", relative_velocity)

    # breakpoint()

    return relative_velocity


# @dataclass(slots=True)
# class NodeKey:
#     """Returns simple (and reversible) hash-key."""

#     # TODO: what should actually be stored (!?)
#     # Maybe lighten / simplify -> share min / supr among classes
#     # don't store input
#     _hash: int = -1

#     MINIMUM: int = -10
#     SUPRENUM: int = 100

#     def __post_init__(self):
#         val_range = self.SUPRENUM - self.MINIMUM
#         o_val = self.obstacle - self.MINIMUM
#         c_val = self.component - self.MINIMUM
#         r_val = self.relative_level - self.MINIMUM
#         self._hash = o_val * (val_range * val_range) + c_val * val_range + r_val

#     def __hash__(self) -> int:
#         return self._hash


class MultiObstacleAvoider:
    """Obstacle Avoider which can take a 'multi-obstacle' as an input."""

    # TODO: future implementation should include multiple obstacles
    def __init__(
        self,
        obstacle: Optional[HierarchyObstacle] = None,
        initial_dynamics: Optional[DynamicalSystem] = None,
        convergence_dynamics: Optional[ObstacleConvergenceDynamics] = None,
        convergence_radius: float = math.pi * 0.5,
        smooth_continuation_power: float = 0.1,
        obstacle_container: Optional[list[HierarchyObstacle]] = None,
        create_convergence_dynamics: bool = False,
    ):
        if initial_dynamics is not None:
            self.initial_dynamics = initial_dynamics

        if create_convergence_dynamics:
            # TODO: [Refactor] -> this implementation should happens somewhere else..
            self.convergence_dynamics = (
                MultiObstacleAvoider.create_local_convergence_dynamics(initial_dynamics)
            )
        else:
            self.convergence_dynamics = convergence_dynamics

        self.convergence_radius = convergence_radius
        self.smooth_continuation_power = smooth_continuation_power

        # self.obstacle = obstacle
        if obstacle is None:
            self.obstacle_list = obstacle_container
        else:
            self.obstacle_list = [obstacle]

        # An ID number which does not co-inside with the obstacle
        self._BASE_VEL_ID = NodeKey(-1, -1, -1)
        self.gamma_power_scaling = 0.5

        self._tangent_tree: VectorRotationTree

    @classmethod
    def create_local_convergence_dynamics(
        initial_dynamics: DynamicalSystem,
        reference_dynamics: Optional[DynamicalSystem] = None,
    ) -> ObstacleConvergenceDynamics:
        """So far this"""
        if hasattr(initial_dynamics, "attractor_position"):
            # TODO: so far this is for stable systems only (!)
            if reference_dynamics is None:
                reference_functor = lambda x: x - initial_dynamics.attractor
            else:
                reference_functor = reference_dynamics.evaluate

            return ProjectedRotationDynamics(
                attractor_position=initial_dynamics.attractor_position,
                initial_dynamics=initial_dynamics,
                reference_velocity=reference_functor,
            )
        else:
            if reference_dynamics is None:
                reference_velocity = np.zeros(self.initial_dynamics.dimension)
                reference_velocity[0] = 1.0
                reference_dynamics = ConstantValue(reference_velocity)

            return ConvergenceDynamicsWithoutSingularity(
                initial_dynamics, reference_dynamics
            )

    def evaluate(self, position: Vector) -> Vector:
        # breakpoint()
        relative_velocity = compute_multiobstacle_relative_velocity(
            position, self.obstacle_list
        )
        velocity = self.initial_dynamics.evaluate(position)
        # So far the convergence direction is only about the root-obstacle
        # in the future, this needs to be extended such that the rotation is_updating
        # ensured to be smooth (!)
        if self.convergence_dynamics is None:
            convergence_direction = velocity
        else:
            convergence_direction = None

        # velocity = velocity - relative_velocity

        final_velocity = self.get_tangent_direction(
            position, velocity, convergence_direction
        )

        final_velocity = final_velocity + relative_velocity
        # breakpoint()
        return final_velocity

    def get_tangent_direction(
        self,
        position: Vector,
        velocity: Vector,
        # linearized_velocity: Vector,
        linearized_velocity: Optional[Vector] = None,
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

        # base_velocity = linearized_velocity
        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_idx=self._BASE_VEL_ID,
            direction=velocity,
        )

        # The base node should be equal to the (initial velocity)
        node_list: list[NodeType] = []
        component_weights: list[list[np.ndarray]] = []
        obstacle_gammas = np.zeros(len(self.obstacle_list))

        # for obs_idx, obstacle in enumerate([self.obstacle]):
        for obs_idx, obstacle in enumerate(self.obstacle_list):
            if linearized_velocity is None:
                local_velocity = (
                    self.convergence_dynamics.evaluate_convergence_around_obstacle(
                        position,
                        obstacle=obstacle.get_component(obstacle.root_idx),
                    )
                )
            else:
                local_velocity = linearized_velocity

            gamma_values, gamma_weights = self.compute_gamma_and_weights(
                obstacle=obstacle, position=position, base_velocity=local_velocity
            )

            # obstacle, base_velocity, position, obs_idx: int, gamma_weights
            node_list += self.populate_tangent_tree(
                obstacle=obstacle,
                base_velocity=local_velocity,
                position=position,
                obs_idx=obs_idx,
                gamma_weights=gamma_weights,
            )

            # print("gamma_values", gamma_values)
            # print("gamma_weights", gamma_weights)

            # component_weights.append(
            #     gamma_weights[gamma_weights > 0]
            #     * (1 / np.maximum(gamma_values, 1.0)) ** self.gamma_power_scaling
            # )
            # Make the weight sum is below 1
            component_weights.append(gamma_weights[gamma_weights > 0])

            if np.sum(component_weights[-1]) > 1.0:
                # breakpoint()
                # TODO: remove when never called (for debugging only...)
                raise ValueError("Unexpected weights...")

            obstacle_gammas[obs_idx] = np.min(gamma_values)

        # Flatten weights over the obstacles
        obstacle_weights = compute_weights(obstacle_gammas)
        weights = np.concatenate(
            [ww * cc for ww, cc in zip(obstacle_weights, component_weights)]
        )

        # Remaining weight to the initial velocity
        node_list.append(self._BASE_VEL_ID)
        weights = np.hstack((weights, [1 - np.sum(weights)]))
        weighted_tangent = self._tangent_tree.get_weighted_mean(
            node_list=node_list, weights=weights
        )

        # print("node list", node_list)
        # print("weights", weights)

        # breakpoint()
        return weighted_tangent

    def compute_gamma_and_weights(
        self, position, obstacle, base_velocity
    ) -> tuple[np.ndarray, np.ndarray]:
        gamma_values = np.zeros(obstacle.n_components)
        for ii in range(obstacle.n_components):
            obs = obstacle.get_component(ii)
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)

        gamma_weights = compute_weights(gamma_values)

        normal = obstacle.get_component(obstacle.root_idx).get_normal_direction(
            position, in_global_frame=True
        )
        reference = obstacle.get_component(obstacle.root_idx).get_reference_direction(
            position, in_global_frame=True
        )

        rotation_weight = RotationalAvoider.get_rotation_weight(
            normal_vector=normal,
            reference_vector=reference,
            convergence_vector=base_velocity,
            convergence_radius=self.convergence_radius,
            gamma_value=min(gamma_values),
            smooth_continuation_power=self.smooth_continuation_power,
        )

        if not (gamma_sum := sum(gamma_weights)) or not rotation_weight:
            return np.zeros_like(gamma_weights), np.zeros_like(gamma_weights)

        gamma_weights = gamma_weights / gamma_sum * rotation_weight
        return gamma_values, gamma_weights

    def populate_tangent_tree(
        self, obstacle, base_velocity, position, obs_idx: int, gamma_weights
    ) -> list[NodeType]:
        # Evaluate rotation weight, to ensure smoothness in space (!)
        node_list = []
        self._tangent_tree.add_node(
            node_id=NodeKey(obs_idx, -1, -1),
            parent_id=self._BASE_VEL_ID,
            direction=base_velocity,
        )

        for comp_id in range(obstacle.n_components):
            if gamma_weights[comp_id] <= 0:
                continue

            node_list.append((obs_idx, comp_id, comp_id))
            self._update_tangent_branch(
                position, comp_id, base_velocity, obstacle, obs_idx
            )

        return node_list

    def _update_tangent_branch(
        self,
        position: Vector,
        comp_id: int,
        base_velocity: np.ndarray,
        obstacle,
        obs_idx: NodeType,
    ) -> None:
        # TODO: predict at start the size (slight speed up)
        # normal_directions: list[Vector] = []
        # reference_directions: list[Vector] = []

        surface_points: list[Vector] = [position]
        parents_tree: list[int] = [comp_id]

        obs = obstacle.get_component(comp_id)
        normal_directions = [obs.get_normal_direction(position, in_global_frame=True)]
        reference_directions = [
            obs.get_reference_direction(position, in_global_frame=True)
        ]

        while parents_tree[-1] != obstacle.root_idx:
            obs = obstacle.get_component(parents_tree[-1])

            new_id = obstacle.get_parent_idx(parents_tree[-1])
            if new_id is None:
                # TODO: We should not reach this?! -> remove(?)
                breakpoint()
                break

            if len(parents_tree) > 10:
                # TODO: remove this debug check
                raise Exception()

            parents_tree.append(new_id)

            obs_parent = obstacle.get_component(new_id)
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
            convergence_radius=np.pi * 0.5,
        )

        # Should this not be the normal parent ?
        self._tangent_tree.add_node(
            node_id=NodeKey(obs_idx, comp_id, parents_tree[-1]),
            parent_id=NodeKey(obs_idx, -1, -1),
            direction=tangent,
        )
        # print("New node", NodeKey(obs_idx, comp_id, parents_tree[-1]))
        # print(f"tangent={tangent}")

        # Iterate over all but last one
        for ii in reversed(range(len(parents_tree) - 1)):
            # print(f"Add Node {ii}")
            rel_id = parents_tree[ii]

            tangent = RotationalAvoider.get_projected_tangent_from_vectors(
                tangent,
                normal=normal_directions[ii],
                reference=reference_directions[ii],
                convergence_radius=self.convergence_radius,
            )

            self._tangent_tree.add_node(
                # node_id=NodeKey(obs_idx, comp_id, rel_id),
                node_id=NodeKey(obs_idx, comp_id, parents_tree[ii]),
                parent_id=NodeKey(obs_idx, comp_id, parents_tree[ii + 1]),
                direction=tangent,
            )

            # print("New node", NodeKey(obs_idx, comp_id, parents_tree[ii]))
            # print(f"tangent={tangent}")
        # breakpoint()


def plot_multi_obstacle(multi_obstacle, ax=None, **kwargs):
    plot_obstacles(
        obstacle_container=multi_obstacle._obstacle_list,
        ax=ax,
        **kwargs,
    )


def test_hashable_node_key():
    node_key = NodeKey(-10, -10, -10, minimum_value=-10)
    assert 0 == hash(node_key)

    node_key1 = NodeKey(8, 7, 8)
    node_key2 = NodeKey(8, 7, 8)
    assert node_key1 == node_key2


# def _test_named_tuple():
#     aa = NodeKey(1, 1, 1)
#     bb = NodeKey(1, 1, 1)

#     assert aa == bb

#     import networkx as nx

#     graph = nx.Graph()
#     graph.add_node(aa)

#     graph.add_node(bb)

#     out_aa = graph.nodes[bb]
#     print("All good.")
#     print(out_aa)
#     breakpoint()


if (__name__) == "__main__":
    # test_hashable_node_key()

    # _test_named_tuple()

    print("Done")
