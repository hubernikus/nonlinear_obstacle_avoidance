"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""
from __future__ import annotations  # To be removed in future python versions

import math
import warnings
from dataclasses import dataclass

from collections import namedtuple
from typing import Optional, Protocol, Hashable

import numpy as np
import numpy.typing as npt

from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.datatypes import Vector
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.dynamics.sequenced_dynamics import evaluate_dynamics_sequence
from nonlinear_avoidance.hierarchy_obstacle_protocol import HierarchyObstacle
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ObstacleConvergenceDynamics,
    ConvergenceDynamicsWithoutSingularity,
)
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence


NodeType = Hashable
NodeKey = namedtuple("NodeKey", "obstacle component relative_level")
# IterationKey = namedtuple("IterationKey", "obstacle end_component current_component")


def plot_multi_obstacle(multi_obstacle, ax=None, **kwargs):
    plot_obstacles(
        obstacle_container=multi_obstacle._obstacle_list,
        ax=ax,
        **kwargs,
    )


def get_limited_weights_to_max_sum(weights: npt.ArrayLike) -> float | np.ndarray:
    """Makes sure weights with value of 1 get preferencial treatment.
    Changes the weights if larger > 1
    Returns the total weight if below 1.0, otherwise the updated list"""
    if (weight_sum := sum(weights)) < 1:
        return (np.array(weights), weight_sum)

    if np.any(ind_max := np.array(weights) >= 1.0):
        new_weights = np.zeros_like(weights)
        new_weights[ind_max] = 1
        # Assign to list
        return (new_weights, 1.0)

    # if zero -> remains zero / otherwise scaling
    new_weights = np.array(weights)
    new_weights = 1.0 / (1 - new_weights) - 1
    new_weights = new_weights / np.sum(new_weights)

    return (new_weights, 1.0)


def compute_gamma_weights(
    distances: npt.ArrayLike,
    min_distance: float = 1.0,
    power_factor: int | float = 1,
) -> np.ndarray:
    """Compute weights based on a distance measure (with no upper limit)"""
    distances = np.array(distances)
    n_points = distances.shape[0]

    critical_points = distances <= min_distance
    if np.sum(critical_points):  # at least one
        if np.sum(critical_points) >= 2:
            # TODO: continuous weighting function
            warnings.warn("Implement continuity of weighting function.")
            return critical_points * 1.0 / np.sum(critical_points)

        return critical_points * 1.0

    distances = distances - min_distance
    weights = (1.0 / distances) ** power_factor

    if np.sum(weights) <= 1:
        return weights
    else:
        return weights / np.sum(weights)


def compute_multiobstacle_relative_velocity(
    position: np.ndarray,
    environment: MultiObstacleContainer,
    cutoff_gamma: float = 10,
) -> np.ndarray:
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
                if position.shape[0] == 2:
                    angular_velocity = np.cross(
                        np.array([0, 0, obs.twist.angular]),
                        np.hstack((position - pose.position, 0)),
                    )
                elif position.shape[0] == 3:
                    angular_velocity = np.cross(
                        obs.twist.angular, (position - pose.position)
                    )
                else:
                    warnings.warn("No dynamic evaluation for higher dynensions.")

                relative_velocity = (
                    relative_velocity
                    + weights[ii] * influence_weight[ii] * angular_velocity[:2]
                )

        if hasattr(obs, "deformation_rate"):
            relative_velocity = (
                relative_velocity
                + weights[ii] * (position - pose.position) * obs.deformation_rate
            )
    return relative_velocity


class MultiObstacleAvoider:
    """Obstacle Avoider which can take a 'multi-obstacle' as an input.


    default_dynamics: Are the 'fall-back' dynamics if they exist..
    """

    # TODO: clean up to remove old functions
    # TODO: refactoring for speed-up
    # TODO: Move outwards at position behind obstacle.
    def __init__(
        self,
        obstacle: Optional[HierarchyObstacle] = None,
        initial_dynamics: Optional[DynamicalSystem] = None,
        convergence_dynamics: Optional[ObstacleConvergenceDynamics] = None,
        convergence_radius: float = math.pi * 0.5,
        gamma_maximum_repulsion: Optional[float] = 0.8,
        smooth_continuation_power: float = 0.3,
        obstacle_container: Optional[list[HierarchyObstacle]] = None,
        create_convergence_dynamics: bool = False,
        default_dynamics: Optional[DynamicalSystem] = None,
    ):
        if initial_dynamics is not None:
            self.initial_dynamics = initial_dynamics

        # The fall-back dynamics
        self.default_dynamics = default_dynamics

        if create_convergence_dynamics:
            # TODO: [Refactor] -> this implementation should happens somewhere else..
            self.convergence_dynamics = (
                MultiObstacleAvoider.create_local_convergence_dynamics(initial_dynamics)
            )
        else:
            self.convergence_dynamics = convergence_dynamics

        if not (math.pi * 0.5 <= convergence_radius <= math.pi):
            raise ValueError(
                f"Convergence_radius of {convergence_radius} is out of bound."
            )

        self.convergence_radius = convergence_radius
        self.gamma_maximum_repulsion = gamma_maximum_repulsion
        self.smooth_continuation_power = smooth_continuation_power

        # self.obstacle = obstacle
        if obstacle is None:
            self.tree_list: MultiObstacleContainer = obstacle_container
        else:
            self.tree_list: MultiObstacleContainer = [obstacle]

        # An ID number which does not co-inside with the obstacle
        self._BASE_VEL_ID = NodeKey(-1, -1, -1)
        self._ROOT_ID = -100

        self.gamma_power_scaling = 0.5

        self._tangent_tree: VectorRotationTree

        # Normals and corresponding weights
        self._normal_vectors: list[float]
        self._cluster_weights: list[float]
        self._rotation_weights: list[float]
        self._convergence_reference_proximity: list[float]

        self._old_relative_velocity = None

    @property
    def singularity(self) -> Optional[np.ndarray]:
        try:
            return self.initial_dynamics.attractor_position
        except AttributeError:
            return None

    @property
    def obstacle_list(self):
        return self.tree_list

    @obstacle_list.setter
    def obstacle_list(self, value):
        self.tree_list = value

    @classmethod
    def create_with_convergence_dynamics(
        cls,
        obstacle_container: list[HierarchyObstacle],
        initial_dynamics: DynamicalSystem,
        reference_dynamics: Optional[DynamicalSystem] = None,
        **kwargs,
    ) -> Self:
        convergence_dynamics = cls.create_local_convergence_dynamics(
            initial_dynamics, reference_dynamics=reference_dynamics
        )
        return cls(
            obstacle_container=obstacle_container,
            initial_dynamics=initial_dynamics,
            convergence_dynamics=convergence_dynamics,
            **kwargs,
        )

    @staticmethod
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

            if initial_dynamics.attractor_position is None:
                breakpoint()

            return ProjectedRotationDynamics(
                attractor_position=initial_dynamics.attractor_position,
                initial_dynamics=initial_dynamics,
                reference_velocity=reference_functor,
            )

        else:
            if reference_dynamics is None:
                raise ValueError("What reference dynamics should be taken !?")

            return ConvergenceDynamicsWithoutSingularity(
                initial_dynamics, reference_dynamics
            )

    @property
    def obstacle_container(self) -> list[HierarchyObstacle]:
        return self.tree_list

    def evaluate(self, position: Vector) -> Vector:
        return self.evaluate_sequence(position)

    def evaluate_sequence(self, position: Vector) -> Vector:
        if hasattr(self.initial_dynamics, "evaluate_magnitude"):
            initial_magnitude = self.initial_dynamics.evaluate_magnitude(position)

        else:
            initial_velocity = self.initial_dynamics.evaluate(position)
            initial_magnitude = np.linalg.norm(initial_velocity)

        initial_sequence = evaluate_dynamics_sequence(
            position,
            self.initial_dynamics,
            # default_dynamics=self.default_dynamics
        )

        if not len(self.obstacle_container):
            return initial_sequence.get_end_vector() * initial_magnitude

        relative_velocity = compute_multiobstacle_relative_velocity(
            position, self.tree_list
        )

        if initial_sequence is None:
            return np.zeros(self.initial_dynamics.dimension)

        # if self.default_dynamics is None:
        convergence_sequence = self.compute_convergence_sequence(
            position, initial_sequence
        )
        # else:
        #     convergence_sequence = evaluate_dynamics_sequence(
        #         position, self.default_dynamics
        #     )

        final_sequence = self.evaluate_avoidance_from_sequence(
            position, convergence_sequence
        )

        # Move velocity to relative (moving) frame
        # final_velocity = final_sequence.get_end_vector() + relative_velocity
        final_velocity = final_sequence.get_end_vector()

        if len(self._cluster_weights) <= 0:
            return final_velocity + relative_velocity

        # averaged_normal, gamma = self.compute_averaged_normal_and_gamma(position)
        averaged_normal = np.sum(
            np.array(self._normal_vectors).T
            * np.tile(self._cluster_weights, (position.shape[0], 1)),
            axis=1,
        )

        scaling_proximity = RotationalAvoider.compute_safe_scaling(
            velocity=final_velocity,
            averaged_normal=averaged_normal,
            gamma=(1.0 / max(self._cluster_weights)),
        )
        scaling_convergence = self.slowfactor_from_convergence()

        slowed_velocity = (
            final_velocity
            / np.linalg.norm(final_velocity)
            * initial_magnitude
            * min(scaling_convergence, scaling_proximity)
        )

        if False:
            print("position", position)
            print("final_velocity", final_velocity)
            print("slowed_velocity", slowed_velocity)
            print("averaged_normal", averaged_normal)
            print("magnitude", magnitude)
            print("self._cluster_weights", self._cluster_weights)

        # breakpoint()
        return slowed_velocity + relative_velocity

    def slowfactor_from_convergence(self, power_factor: float = 0.2) -> float:
        """Returns scaling factor between [0, 1]"""
        rot_weight = np.array(self._rotation_weights)
        conv_prox = np.array(self._convergence_reference_proximity)
        ind_pos = np.logical_and(rot_weight > 0.0, 0.0 < conv_prox, conv_prox < 1.0)

        scaling = np.ones_like(rot_weight)
        scaling[ind_pos] = ((1 - conv_prox[ind_pos]) / rot_weight[ind_pos]) ** (
            power_factor
        )
        # Ensure a max of 1 (!)
        return min(np.min(scaling), 1)

    def evaluate_old(self, position: Vector) -> Vector:
        warnings.warn("This function is currently outdated and does not work well.")
        if False:
            # TODO: remove soon...
            raise NotImplementedError(
                "This function is outdated and does not work well."
            )

        relative_velocity = compute_multiobstacle_relative_velocity(
            position, self.tree_list
        )
        initial_velocity = self.initial_dynamics.evaluate(position)

        # So far the convergence direction is only about the root-obstacle
        # in the future, this needs to be extended such that the rotation is_updating
        # ensured to be smooth (!)
        if self.convergence_dynamics is None:
            convergence_direction = initial_velocity
        else:
            convergence_direction = None

        # velocity = velocity - relative_velocity

        final_velocity = self.evaluate_avoidance_direction(
            position, initial_velocity, convergence_direction
        )
        final_velocity = final_velocity + relative_velocity

        averaged_normal, gamma = self.compute_averaged_normal_and_gamma(position)

        # TODO: revert to sequence evaluation
        slowed_velocity = RotationalAvoider.compute_safe_magnitude(
            rotated_velocity=final_velocity,
            initial_norm=np.linalg.norm(initial_velocity),
            averaged_normal=averaged_normal,
            gamma=gamma,
        )

        if np.any(np.isnan(position)) or np.any(np.isnan(slowed_velocity)):
            breakpoint()
        # breakpoint()
        return slowed_velocity

    def compute_averaged_normal_and_gamma(
        self, position: Vector, weight_power: float = 2.0
    ) -> tuple[Vector, float]:
        # TODO: this recomputation could be done directly during the algorithm
        normal_vectors: list[np.ndarray] = []
        gamma_values: list[float] = []
        for tree in self.tree_list:
            for obs in tree._obstacle_list:
                # TODO: make this NOT private anymore (!)
                gamma_values.append(obs.get_gamma(position, in_global_frame=True))

                normal_vectors.append(
                    obs.get_normal_direction(position, in_global_frame=True)
                )

        min_gamma = min(gamma_values)
        if min_gamma <= 1:
            weights = np.array(gamma_values) <= 1
            weights = weights / np.sum(weights)
        else:
            weights = 1.0 / np.array(gamma_values) ** weight_power
            if np.sum(weights) > 1.0:
                weights = weights / np.sum(weights)

        averaged_normal = np.sum(
            np.array(normal_vectors).T * np.tile(weights, (position.shape[0], 1)),
            axis=1,
        )

        return averaged_normal, min_gamma

    def get_tangent_direction(self, *args, **kwargs):
        return self.evaluate_avoidance_direction(*args, **kwargs)

    def evaluate_avoidance_direction(
        self,
        position: Vector,
        initial_velocity: Vector,
        linearized_velocity: Optional[Vector] = None,
    ) -> Vector:
        """Evaluates the preferred direction to go around the obstacle."""
        if not np.linalg.norm(initial_velocity):
            return initial_velocity

        sequence = self.evaluate_avoidance_sequence(
            position,
            initial_velocity=initial_velocity,
            linearized_velocity=linearized_velocity,
        )
        return sequence.get_end_vector()

    def compute_convergence_direction(self, position):
        init_seq = evaluate_dynamics_sequence(position, self.initial_dynamics)
        final_seq = self.compute_convergence_sequence(position, init_seq)
        return final_seq.get_end_vector()

    def compute_tree_weight(
        self,
        obstacle,
        gamma_decreasing_influence: float = 2.0,
    ) -> float:
        """Returns the weight with which the full avoidance is getting neglected,
        over prioritizing the smoothness across the directional-singularity point.

        gamma_decreasing_influence : float = Defines the upper limit of gamma at
            which this is started to be activated.
        """
        if self.singularity is None:
            return 1.0
        tree_gamma = obstacle.get_gamma(self.singularity, in_global_frame=True)

        if tree_gamma <= 1.0:
            # Attractor inside obstacle -> no convergence direction
            return 0.0

        if tree_gamma >= gamma_decreasing_influence:
            return 1.0

        return (tree_gamma - 1.0) / (gamma_decreasing_influence - 1.0)

    def compute_convergence_sequence(
        self,
        position: np.ndarray,
        initial_sequence: VectorRotationSequence,
    ) -> VectorRotationSequence:
        """Computes and averages the convergence sequence."""

        # Create sequence and populate it
        root_id = -10
        init_id = -1
        self.conv_tree = VectorRotationTree.from_sequence(
            root_id=root_id,
            node_id=init_id,
            sequence=initial_sequence,
        )

        node_list = []
        weight_list = []
        for ii_tree, obstacle_tree in enumerate(self.tree_list):
            tree_weight = self.compute_tree_weight(obstacle_tree)
            if tree_weight <= 1e-6:
                continue

            root = obstacle_tree.get_root()
            node_id = (ii_tree, obstacle_tree.root_idx)

            weight = self.convergence_dynamics.evaluate_projected_weight(position, root)

            trafo_pos_to_root = (
                self.convergence_dynamics.evaluate_rotation_position_to_transform(
                    position, root
                )
            )

            if self.default_dynamics is None:
                # Check if there are 'fall-back' global dynamics
                convergence_sequence = evaluate_dynamics_sequence(
                    root.get_reference_point(in_global_frame=True),
                    self.initial_dynamics,
                )
            else:
                convergence_sequence = evaluate_dynamics_sequence(
                    root.get_reference_point(in_global_frame=True),
                    self.default_dynamics,
                )

            try:
                continuous_sequence = self.vector_rotation_reduction(
                    initial_sequence, trafo_pos_to_root, convergence_sequence, weight
                )
            except:
                breakpoint()

            self.conv_tree.add_sequence(
                sequence=continuous_sequence,
                node_id=node_id,
                parent_id=root_id,
            )

            # Nonzero weight expected, since weight > 0
            if weight > 1e-6:
                # Each node forms a n individual branch with root velocity
                # -> no need to store / calculate
                # in case of 0 weight
                node_list.append(node_id)
                weight_list.append(weight * tree_weight)

            for ii_com, component in enumerate(obstacle_tree):
                if ii_com == obstacle_tree.root_idx:
                    continue

                weight = self.convergence_dynamics.evaluate_projected_weight(
                    position, component
                )

                if weight <= 1e-2:
                    continue

                # The constructed 'tree' is reverse to the obstacle tree
                # To avoid confusion we use pred(ecessor) & node for the directions & graph
                # and obstalce / parent for the obstacles-tree
                ii_obs = ii_com
                pred_id = root_id
                pred_point = position
                for pp in range(self.conv_tree.maximum_level):
                    node_id = (ii_tree, ii_com, pp)
                    obs_point = obstacle_tree.get_component(
                        ii_obs
                    ).global_reference_point

                    trafo_comp_to_parent = (
                        self.convergence_dynamics.evaluate_rotation_start_to_end(
                            pred_point,
                            obs_point,
                            center=self.singularity,
                        )
                    )
                    if trafo_comp_to_parent is None:
                        breakpoint()  # TODO: debugging -> remove

                    # TODO: this could be the vector directly...
                    self.conv_tree.add_node_orientation(
                        orientation=trafo_comp_to_parent,
                        node_id=node_id,
                        parent_id=pred_id,
                    )

                    # Update direction points & id
                    pred_id = node_id
                    pred_point = obs_point

                    ii_obs = obstacle_tree.get_parent_idx(ii_obs)
                    if ii_obs is None:
                        # Root reached
                        break

                # Add convergence transformation to branch
                node_id = (ii_tree, ii_com)
                try:
                    self.conv_tree.add_sequence(
                        sequence=convergence_sequence,
                        node_id=node_id,
                        parent_id=pred_id,
                    )
                except:
                    breakpoint()

                node_list.append(node_id)
                weight_list.append(weight * tree_weight)

        # if hasattr(self.initial_dynamics, "attractor_position"):
        #     # Ensure convergence around attractor
        #     node_list.append(init_id)
        #     dist_norm = np.linalg.norm(
        #         position - self.initial_dynamics.attractor_position
        #     )
        #     weight_list.append(1.0 / (1 + dist_norm))

        node_list.append(init_id)
        weight_list.append(0.0)

        # print(weight_list)
        # print(node_list)

        # Normalize weight [add to initial if small]
        weight_list, tot_weight = get_limited_weights_to_max_sum(weight_list)
        weight_list[-1] = weight_list[-1] + (1 - tot_weight)

        weighted_sequence = self.conv_tree.reduce_weighted_to_sequence(
            node_list=node_list, weights=weight_list
        )

        if np.any(np.isnan(weighted_sequence.basis_array)):
            breakpoint()  # TODO: remove after DEBUG

        return weighted_sequence

    @staticmethod
    def vector_rotation_reduction(
        sequence1: VectorRotationSequence,
        trafo_seq1_to_seq2: Optional[VectorRotationXd],
        sequence2: VectorRotationSequence,
        weight2: float,
        # weight_factor: bool = 4,
    ) -> VectorRotationSequence:
        # Start effect drop off later
        # weight2 = min(1, weight2 * weight_factor)

        if weight2 <= 0 or trafo_seq1_to_seq2 is None:
            return sequence1

        # Assumptions that weight1
        tmp_tree = VectorRotationTree.from_sequence(
            node_id=1, sequence=sequence1, root_id=0
        )
        # tmp_tree.add_node_orientation(
        #     orientation=trafo_seq1_to_seq2, node_id=-1, parent_id=1
        # )
        tmp_tree.add_node_orientation(
            orientation=trafo_seq1_to_seq2, node_id=-1, parent_id=0
        )
        tmp_tree.add_sequence(node_id=2, sequence=sequence2, parent_id=-1)

        return tmp_tree.reduce_weighted_to_sequence([1, 2], [(1 - weight2), weight2])

    def add_convergence_directions(
        self, position, component, parent_position, node_id, parent_id: NodeType
    ) -> Optional[float]:
        weight = self.convergence_dynamics.evaluate_projected_weight(
            position, component
        )
        # Start this weight reduction later
        # weight = max(1, weight * 2)
        if math.isclose(weight, 0, abs_tol=1e-3):
            return None

        trafo_pos_to_attr = (
            self.convergence_dynamics.evaluate_rotation_position_to_transform(
                parent_position, component
            )
        )
        convergence_sequence = evaluate_dynamics_sequence(
            component.get_reference_point(in_global_frame=True),
            self.initial_dynamics,
        )

        if convergence_sequence is None:
            raise NotImplementedError("Obstacle at center.")

        self.conv_tree.add_node_orientation(
            orientation=trafo_pos_to_attr,
            node_id=self.get_reference_node(node_id),
            parent_id=parent_id,
        )
        self.conv_tree.add_sequence(
            sequence=convergence_sequence,
            node_id=node_id,
            parent_id=self.get_reference_node(node_id),
        )
        return weight

    @staticmethod
    def get_reference_node(node_id):
        return (node_id, 0)

    def evaluate_avoidance_from_sequence(self, position, initial_sequence):
        """Keep track of the rotation sequence when evaluating the full avoidance.
        This is advantageous when trying to follow highly nonlinear dynamics and
        large obstacle-trees.
        """
        # Only last vector is really important
        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_idx=self._BASE_VEL_ID,
            direction=initial_sequence.get_end_vector(),
        )

        # Reset normal vectors
        self._normal_vectors = []
        self._rotation_weights = []
        self._convergence_reference_proximity = []

        node_list: list[NodeType] = []
        component_weights: list[list[np.ndarray]] = []
        # obstacle_gammas = np.zeros(len(self.tree_list))
        tree_influence_weights = np.zeros(len(self.tree_list))

        for ii_tree, obstacle_tree in enumerate(self.tree_list):
            gamma_values, gamma_weights = self.compute_gamma_and_weights(
                obstacle=obstacle_tree,
                position=position,
                base_velocity=initial_sequence.get_end_vector(),
            )
            occlusion_weights = self.compute_parent_occlusion_weight(
                position, obstacle_tree
            )
            obstacle_weights = gamma_weights * occlusion_weights

            node_list += self.simple_population_tangent_tree(
                obstacle=obstacle_tree,
                base_velocity=initial_sequence.get_end_vector(),
                position=position,
                obs_idx=ii_tree,
                obstacle_weights=obstacle_weights,
                start_id=self._BASE_VEL_ID,
                gamma_values=gamma_values,
            )
            # obstacle_gammas[ii_tree] = np.min(gamma_values)

            # Normalize component weight
            tree_influence_weights[ii_tree] = np.sum(obstacle_weights)

            if np.isnan(np.sum(obstacle_weights)):
                breakpoint()

            # Normalize weights
            obstacle_weights = obstacle_weights[obstacle_weights > 0]
            if len(obstacle_weights):
                obstacle_weights = obstacle_weights / np.sum(obstacle_weights)
            else:
                # At least one weight should be added
                # component_weights.append(np.array([0.0]))
                pass

            component_weights.append(obstacle_weights)

        # print(tree_influence_weights)
        normalized_weights, tot_weight = get_limited_weights_to_max_sum(
            tree_influence_weights
        )
        # print(tree_influence_weights)

        # Flatten weights over the obstacles
        # obstacle_weights = compute_weights(obstacle_gammas)
        self._cluster_weights = np.concatenate(
            [wo * wc for wo, wc in zip(normalized_weights, component_weights)]
        )

        # Remaining weight to the initial velocity
        node_list.append(self._BASE_VEL_ID)
        final_weights = np.hstack(
            (self._cluster_weights, [1 - np.sum(self._cluster_weights)])
        )
        weighted_sequence = self._tangent_tree.reduce_weighted_to_sequence(
            node_list=node_list, weights=final_weights
        )

        # print("rotation_weight", self._rotation_weights)
        # print("cluster weights", self._cluster_weights)
        # print("final_weights", final_weights)
        # print("node list", node_list)
        return weighted_sequence

    @staticmethod
    def compute_weights_from_distances(
        distances: np.ndarray, distance_min: float = 1.0, normalize: bool = True
    ) -> np.ndarray:
        ind_low = distances <= distance_min
        if np.sum(ind_low):
            return ind_low / np.sum(ind_low)

        weights = 1.0 / (distances - distance_min)
        if normalize and (weights_sum := np.sum(weights)) > 1.0:
            weights = weights / weights_sum

        if np.any(np.isinf(weights)):
            raise NotImplementedError("TODO")

        return weights

    def evaluate_avoidance_sequence(
        self,
        position: Vector,
        initial_velocity: Vector,
        linearized_velocity: Optional[Vector],
    ) -> VectorRotationSequence:
        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_idx=self._BASE_VEL_ID,
            direction=initial_velocity,
        )

        # The base node should be equal to the (initial velocity)
        node_list: list[NodeType] = []
        component_weights: list[list[np.ndarray]] = []
        obstacle_gammas = np.zeros(len(self.tree_list))

        # for obs_idx, obstacle in enumerate([self.obstacle]):
        for obs_idx, obstacle in enumerate(self.tree_list):
            if linearized_velocity is None:
                local_velocity = (
                    self.convergence_dynamics.evaluate_convergence_around_obstacle(
                        position,
                        obstacle=obstacle.get_component(obstacle.root_idx),
                    )
                )
            else:
                local_velocity = linearized_velocity

            if np.any(np.isnan(local_velocity)):
                breakpoint()

            gamma_values, gamma_weights = self.compute_gamma_and_weights(
                obstacle=obstacle, position=position, base_velocity=local_velocity
            )

            # obstacle, base_velocity, position, obs_idx: int, gamma_weights
            new_nodes = self.populate_tangent_tree(
                obstacle=obstacle,
                base_velocity=local_velocity,
                position=position,
                obs_idx=obs_idx,
                obstacle_weights=gamma_weights,
            )
            node_list += new_nodes
            # component_weights.append(
            #     gamma_weights[gamma_weights > 0]
            #     * (1 / np.maximum(gamma_values, 1.0)) ** self.gamma_power_scaling
            # )

            # Make the weight sum is below 1
            ind_weights = gamma_weights > 0
            if np.sum(ind_weights) > 0:
                component_weights.append(gamma_weights[gamma_weights > 0])
            else:
                # At least one weight should be nonzero
                component_weights.append(np.array([0.0]))

            if np.sum(component_weights[-1]) > 1 + 1e-6:
                breakpoint()
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
        self.final_weights = np.hstack((weights, [1 - np.sum(weights)]))

        weighted_sequence = self._tangent_tree.reduce_weighted_to_sequence(
            node_list=node_list, weights=self.final_weights
        )

        if np.any(np.isnan(weighted_sequence.get_end_vector())):
            breakpoint()

        return weighted_sequence

    def compute_gamma_and_weights(
        self,
        position: np.ndarray,
        obstacle: Obstacle,
        base_velocity: np.ndarray,
        weight_power: float = 1 / 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the gammas and weights based on the convergence-rotation and the distance for
        a obstacle(-tree)."""
        gamma_values = np.zeros(obstacle.n_components)
        for ii in range(obstacle.n_components):
            obs = obstacle.get_component(ii)
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)

        # TODO: maybe use 'convergence_dynamics.evaluate_projected_weight' instead
        gamma_weights = self.compute_weights_from_distances(
            gamma_values, normalize=False
        )

        normal = obstacle.get_component(obstacle.root_idx).get_normal_direction(
            position, in_global_frame=True
        )
        reference = obstacle.get_component(obstacle.root_idx).get_reference_direction(
            position, in_global_frame=True
        )

        # TODO: rotation weight could be done outside...
        # Single rotation weight, as it's for the root-normal and the velocity
        rotation_weight = RotationalAvoider.get_rotation_weight(
            normal_vector=normal,
            reference_vector=reference,
            convergence_vector=base_velocity,
            gamma_value=min(gamma_values),
            smooth_continuation_power=self.smooth_continuation_power,
            # convergence_radius=self.convergence_radius,
        )

        # Store the weights for using it to slow down the velocity
        self._rotation_weights.append(rotation_weight)
        conv_norml = base_velocity / np.linalg.norm(base_velocity)
        self._convergence_reference_proximity.append(conv_norml @ reference)

        if not (gamma_sum := sum(gamma_weights)) or not rotation_weight:
            return np.ones_like(gamma_weights), np.zeros_like(gamma_weights)

        total_weights = gamma_weights / gamma_sum * rotation_weight
        if np.any(np.isnan(gamma_weights)):
            breakpoint()

        return gamma_values, total_weights

    def simple_population_tangent_tree(
        self,
        obstacle,
        base_velocity,
        position,
        obs_idx: int,
        obstacle_weights,
        start_id: Hashable,
        gamma_values: Optional[np.ndarray] = None,
    ) -> list[NodeType]:
        """Returns the node-list and the (normalized) occlusion weights."""
        # Evaluate rotation weight, to ensure smoothness in space (!)
        if start_id is None:
            start_id = self._BASE_VEL_ID

        node_list = []
        self._tangent_tree.add_node(
            node_id=NodeKey(obs_idx, -1, -1),
            parent_id=start_id,
            direction=base_velocity,
        )

        convergence_radiuses = self.compute_convergence_radiuses(
            gamma_values, position=position, obstacle_tree=obstacle
        )

        for comp_id in range(obstacle.n_components):
            if obstacle_weights[comp_id] <= 0:
                continue

            node_list.append((obs_idx, comp_id, comp_id))
            self._simple_tangent_branch_update(
                position=position,
                comp_id=comp_id,
                base_velocity=base_velocity,
                obstacle=obstacle,
                obs_idx=obs_idx,
                convergence_radius=convergence_radiuses[comp_id],
            )
        return node_list

    def compute_convergence_radiuses(
        self,
        gamma_values: npt.ArrayLike,
        position: np.ndarray,
        obstacle_tree: MultiObstacle,
        convergence_radius_inside: float = math.pi,
    ) -> np.ndarray:
        """Returns the convergence radiuses based on a lower-gamma limit, at which
        there is full repulsion,"""
        # TODO: compute as absolute distance from surface (?) /
        # BUT needs to be aware of the reference position (if it's close...)
        conv_radii = self.convergence_radius * np.ones_like(gamma_values)
        if self.gamma_maximum_repulsion is None:
            return conv_radii

        ind_small = np.array(gamma_values) < 1.0
        if not np.sum(ind_small):
            return conv_radii

        # weights = np.ones(ind_small.shape[0])
        for ii in np.arange(ind_small.shape[0])[ind_small]:
            component = obstacle_tree.get_component(ii)
            # center = component.global_reference_point
            center = component.center_position
            if np.allclose(position, center):
                conv_radii[ii] = convergence_radius_inside
                continue

            surface_point = component.get_intersection_with_surface(
                center, position - center, in_global_frame=True
            )

            inside_gamma = np.linalg.norm(position - center) / np.linalg.norm(
                surface_point - center
            )
            if inside_gamma < self.gamma_maximum_repulsion:
                conv_radii[ii] = convergence_radius_inside
                continue

            weight = (inside_gamma - self.gamma_maximum_repulsion) / (
                1 - self.gamma_maximum_repulsion
            )
            conv_radii[ii] = conv_radii[ii] * weight + convergence_radius_inside * (
                1.0 - weight
            )

        if np.any(np.isnan(conv_radii)):
            breakpoint()

        return conv_radii

    def populate_tangent_tree(
        self,
        obstacle,
        base_velocity,
        position,
        obs_idx: int,
        obstacle_weights,
        start_id: Optional[Hashable] = None,
    ) -> list[NodeType]:
        """Returns the node-list and the (normalized) occlusion weights."""
        # Evaluate rotation weight, to ensure smoothness in space (!)
        if start_id is None:
            start_id = self._BASE_VEL_ID

        node_list = []
        self._tangent_tree.add_node(
            node_id=NodeKey(obs_idx, -1, -1),
            parent_id=start_id,
            direction=base_velocity,
        )

        for comp_id in range(obstacle.n_components):
            if obstacle_weights[comp_id] <= 0:
                continue

            node_list.append((obs_idx, comp_id, comp_id))
            self._update_tangent_branch(
                position, comp_id, base_velocity, obstacle, obs_idx
            )
        return node_list

    def compute_parent_occlusion_weight(
        self, position: Vector, obstacle_tree: MultiObstacle
    ) -> np.ndarray:
        occlusion_weights = np.ones(obstacle_tree.n_components)
        for ii, component in enumerate(obstacle_tree):
            if ii == obstacle_tree.root_idx:
                # Root cannot be occluded by parent
                continue

            reference_directions = component.get_reference_direction(
                position, in_global_frame=True
            )
            if component.get_gamma(position, in_global_frame=True) < 1.0:
                reference_directions = (-1) * reference_directions

            surf_point = component.get_intersection_with_surface(
                position, reference_directions, in_global_frame=True
            )
            # Check if its occluded by parent
            comp_parent = obstacle_tree.get_component(obstacle_tree.get_parent_idx(ii))
            occlusion_gamma = comp_parent.get_gamma(surf_point, in_global_frame=True)

            if occlusion_gamma >= 1:
                continue

            ref_parent = comp_parent.get_reference_point(in_global_frame=True)
            ref_comp = component.get_reference_point(in_global_frame=True)
            vec_pos = position - ref_comp
            vec_parent = ref_parent - ref_comp
            dot_prod = (vec_parent @ vec_pos) / (
                np.linalg.norm(vec_pos) * np.linalg.norm(vec_parent)
            )

            if dot_prod >= 1.0:
                occlusion_weights[ii] = 0
                continue
            occlusion_weights[ii] = occlusion_gamma ** (1 / (1 - dot_prod))

        if np.any(np.isnan(occlusion_weights)):
            breakpoint()
        return occlusion_weights

    def compute_occlusion_weights(
        self,
        position: np.ndarray,
        obstacle: MultiObstacle,
        gamma_margin: float = 0.1,
        max_gamma: Optional[float] = 2.0,
    ) -> np.ndarray:
        """Returns the weights of the components of the obstacle at position.
        The gamma_margin ensures smooth transition if walls are overlapping."""
        occlusion_gammas = np.zeros(obstacle.n_components)
        for comp_id in range(obstacle.n_components):
            obs = obstacle.get_component(comp_id)
            reference_directions = obs.get_reference_direction(
                position, in_global_frame=True
            )
            surface_point = obs.get_intersection_with_surface(
                position, reference_directions, in_global_frame=True
            )
            occlusion_gammas[comp_id] = obstacle.get_gamma_except_components(
                surface_point, [comp_id], in_global_frame=True
            )
            # # Compute occlusion gamma
            # if comp_id == obstacle.root_idx:
            #     return 1.0

            # new_id = obstacle.get_parent_idx(comp_id)
            # obs_parent = obstacle.get_component(new_id)
            # Get minimum gamma of all obstacles
            # occlusion_gamma = obstacle.get_gamma_except_components(
            #     surface_points[0], [comp_id], in_global_frame=True
            # )

        if max_gamma is not None:
            occlusion_gammas = np.maximum(occlusion_gammas, max_gamma)

        # print("occlusion_gammas", occlusion_gammas)
        if np.any(occlusion_gammas > 1 + gamma_margin):
            ind_free = occlusion_gammas > 1
            occlusion_weights = np.zeros_like(occlusion_gammas)
            occlusion_weights[ind_free] = (
                occlusion_gammas[ind_free] - 1
            ) / occlusion_gammas[ind_free]
            occlusion_weights = occlusion_weights / np.sum(occlusion_weights)
            return occlusion_weights

        # Otherwise give the weight to the least occluded,
        # add margin for slightly more smooth transition
        max_gamma = np.max(occlusion_gammas)
        occlusion_weights = np.maximum(occlusion_gammas - max_gamma + gamma_margin, 0)
        occlusion_weights = occlusion_weights / np.sum(occlusion_weights)

        return occlusion_weights

    @staticmethod
    def compute_single_occlusion_weight(
        gamma: float, power_factor: float = 1.0 / 4
    ) -> float:
        if gamma <= 1:
            # Power factor is not possible for negative numbers
            return gamma - 1
        return ((gamma - 1) / gamma) ** power_factor

    @staticmethod
    def evaluate_occlusion_array(occlusion_weights):
        # Evaluate occlusion weights
        occlusion_weights = np.array(occlusion_weights)
        ind_pos = occlusion_weights > 0
        if any(ind_pos):
            final_weights = np.zeros_like(occlusion_weights)
            final_weights[ind_pos] = occlusion_weights[ind_pos] / np.sum(
                occlusion_weights[ind_pos]
            )
            return final_weights

        arg_max = np.argsort(occlusion_weights)
        final_weights = np.zeros_like(occlusion_weights)
        final_weights[-1] = 1.0
        for ii in range(final_weights.shape[0] - 1, 0):
            # Take the closest ones and divide (!)
            if occlusion_weights[ii] < occlusion_weights[-1]:
                break
            final_weights[ii] = 1.0
        return final_weights / np.sum(final_weights)

    def _compute_normal_and_reference_of_branch(
        self,
        position: Vector,
        comp_id: int,
        obstacle_tree: MultiObstacle,
        level_max: int = 100,
    ):
        obs = obstacle_tree.get_component(comp_id)

        # surface_points: list[Vector] = [position]
        parents_tree: list[int] = [comp_id]
        reference_directions = [
            obs.get_reference_direction(position, in_global_frame=True)
        ]

        if obs.get_gamma(position, in_global_frame=True) > 1.0:
            # TODO: use maybe gamma instead of distance (?)
            intersection = obs.get_intersection_with_surface(
                obs.global_reference_point,
                (-1) * reference_directions[-1],
                in_global_frame=True,
            )
            distance_to_surf = np.linalg.norm(position - intersection)
        else:
            distance_to_surf = 0
            intersection = obs.get_intersection_with_surface(
                position, (-1) * reference_directions[-1], in_global_frame=True
            )

        surface_points: list[Vector] = [intersection]

        normal_directions = [
            self.get_normal_at_distance(obs, surface_points[-1], distance_to_surf)
        ]

        for ii in range(level_max):
            if parents_tree[-1] == obstacle_tree.root_idx:
                break
            obs = obstacle_tree.get_component(parents_tree[-1])
            new_id = obstacle_tree.get_parent_idx(parents_tree[-1])
            parents_tree.append(new_id)
            obs_parent: Obstacle_tree = obstacle_tree.get_component(new_id)

            if new_id is None:
                # TODO: We should not reach this?! -> remove(?)
                breakpoint()
                break

            if len(parents_tree) > 10:
                # TODO: remove this debug check
                raise Exception()

            # Go from reference point outwards to avoid numerical errors
            ref_point = obs.get_reference_point(in_global_frame=True)
            ref_dir = surface_points[-1] - ref_point

            # if obs_parent.get_gamma(surface_points[-1], in_global_frame=True) < 1:
            #     ref_dir = (-1) * ref_dir

            intersection = obs_parent.get_intersection_with_surface(
                ref_point, ref_dir, in_global_frame=True
            )
            if intersection is None:
                # TODO: This should probably never happen -> remove?
                # but for now easier to debug / catch (other) errors early
                breakpoint()
                raise Exception()
            surface_points.append(intersection)

            normal_directions.append(
                self.get_normal_at_distance(
                    obs_parent, surface_points[-1], distance_to_surf
                )
            )
            reference_directions.append(
                obs_parent.get_reference_direction(intersection, in_global_frame=True)
            )

        # # print("surf-points \n", np.array(surface_points))
        # if comp_id > 0:
        #     breakpoint()

        return parents_tree, normal_directions, reference_directions

    def _simple_tangent_branch_update(
        self,
        position: Vector,
        comp_id: int,
        base_velocity: np.ndarray,
        obstacle: MultiObstacle,
        obs_idx: NodeType,
        convergence_radius: float = None,
    ) -> NodeType:
        (
            parents_tree,
            normal_directions,
            reference_directions,
        ) = self._compute_normal_and_reference_of_branch(
            position, comp_id, obstacle_tree=obstacle
        )

        velocity = base_velocity
        parent_id = NodeKey(obs_idx, -1, -1)
        for ii in reversed(range(1, len(parents_tree))):
            rotation = VectorRotationXd.from_directions(
                reference_directions[ii], reference_directions[ii - 1]
            )
            velocity = rotation.rotate(velocity)
            # print("velocity", velocity)

            node_id = NodeKey(obs_idx, comp_id, parents_tree[ii])
            self._tangent_tree.add_node(
                node_id=node_id,
                parent_id=parent_id,
                direction=velocity,
            )
            parent_id = node_id

        if (
            obstacle.get_component(comp_id).tail_effect
            or np.dot(normal_directions[0], velocity) < 0
        ):
            if convergence_radius > math.pi * 0.5:
                # Add an intermediary node to ensure angle < math.pi
                tangent = RotationalAvoider.get_projected_tangent_from_vectors(
                    velocity,
                    normal=normal_directions[0],
                    reference=reference_directions[0],
                    convergence_radius=math.pi * 0.5,
                )
                node_id = NodeKey(obs_idx, comp_id, -1)
                self._tangent_tree.add_node(
                    node_id=node_id,
                    parent_id=parent_id,
                    direction=tangent,
                )
                parent_id = node_id

            # Tail effecto or pointing towards obstacle
            avoidance_direction = RotationalAvoider.get_projected_tangent_from_vectors(
                velocity,
                normal=normal_directions[0],
                reference=reference_directions[0],
                convergence_radius=convergence_radius,
            )
        else:
            # Just keep 'rotated' velocity
            avoidance_direction = velocity

        node_id = NodeKey(obs_idx, comp_id, parents_tree[0])
        self._tangent_tree.add_node(
            node_id=node_id,
            parent_id=parent_id,
            direction=avoidance_direction,
        )

        # Store normal vectors for 'averaged-normal' computation
        self._normal_vectors.append(normal_directions[0])

        # # np.set_printoptions(precision=6)
        # print("obs_idx", obs_idx)
        # print("comp_id", comp_id)
        # # print(f"tree {obs_idx} | comp {comp_id}")
        # # # print("reference_directions \n", np.array(reference_directions))
        # print("normals \n", np.array(normal_directions))

        # print("baseVe", base_velocity)
        # print("velocity", velocity)
        # print("normal", normal_directions[0])
        # try:
        #     print("tangent", tangent)
        # except:
        #     pass
        # print("avoidance_direction", avoidance_direction)

        return node_id

    def _update_tangent_branch(
        self,
        position: Vector,
        comp_id: int,
        base_velocity: np.ndarray,
        obstacle: MultiObstacle,
        obs_idx: NodeType,
    ):
        """Updates the tangent of the specified obstacle and returns the 'interection' weight.

        WARNING: This function is not used anymore..
        """
        (
            parents_tree,
            normal_directions,
            reference_directions,
        ) = self._compute_normal_and_reference_of_branch(
            position, comp_id, obstacle_tree=obstacle
        )

        # np.set_printoptions(precision=6)
        # print("comp_id", comp_id)
        # print("baseVe", np.round(base_velocity, 3))
        # print("normals \n", np.array(normal_directions))
        # print("reference_directions \n", np.array(reference_directions))

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
        # print(f"tangent={tangent}")

        if np.any(np.isnan(tangent)):
            # TODO: remove DEBUG check
            breakpoint()

        # Iterate over all but last one
        for ii in reversed(range(len(parents_tree) - 1)):
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
            # print(f"tangent={tangent}")
            # breakpoint()

            if np.any(np.isnan(tangent)):
                # print("New node", NodeKey(obs_idx, comp_id, parents_tree[ii]))
                print(f"tangent={tangent}")
                breakpoint()

    def get_normal_at_distance(
        self, obs: Obstacle, surface_point: np.ndarray, distance: float
    ):
        """Returns the normal direction.
        For non-smooth surface polygon, the normal is evaluated at distance"""
        if not isinstance(obs, Cuboid):
            return obs.get_normal_direction(surface_point, in_global_frame=True)

        direction = surface_point - obs.center_position
        if not (dir_norm := np.linalg.norm(direction)):
            raise ValueError("Position is at the surface of the obstacle.")

        position = surface_point + direction / dir_norm * distance
        return obs.get_normal_direction(position, in_global_frame=True)
