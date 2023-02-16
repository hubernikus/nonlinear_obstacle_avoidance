"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
from typing import Optional, Protocol

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_with_circle, CircleIntersectionType
from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from roam.multi_ellipse_obstacle import MultiEllipseObstacle
from roam.rotational_avoider import RotationalAvoider
from roam.vector_rotation import VectorRotationTree
from roam.nonlinear_rotation_avoider import ObstacleConvergenceDynamics
from roam.datatypes import Vector


class HierarchyObstacle(Protocol):
    # + all methods of a general obstacle(?)
    @property
    def n_components(self) -> int:
        ...

    @property
    def root_id(self) -> int:
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
        convergence_radius: float = math.pi * 0.2,
        smooth_continuation_power: float = 0.0,
    ):
        if initial_dynamics is not None:
            self.initial_dynamics = initial_dynamics

        if convergence_dynamics is not None:
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
        convergence_direction = (
            self.convergence_dynamics.evaluate_convergence_around_obstacle(
                position, obstacle=self.obstacle.get_component(self.obstacle.root_id)
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
        base_velocity = linearized_velocity

        gamma_values = np.zeros(self.obstacle.n_components)
        for ii in range(self.obstacle.n_components):
            obs = self.obstacle.get_component(ii)
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)

        gamma_weights = compute_weights(gamma_values)

        # Get convergence direction
        idx_root = self.obstacle.root_id
        normal = self.obstacle.get_component(idx_root).get_normal_direction(
            position, in_global_frame=True
        )
        reference = self.obstacle.get_component(idx_root).get_reference_direction(
            position, in_global_frame=True
        )

        # Evaluate rotation weight, to ensure smoothness in space (!)
        rotation_weight = RotationalAvoider.get_rotation_weight(
            normal_vector=normal,
            reference_vector=reference,
            convergence_vector=linearized_velocity,
            convergence_radius=self.convergence_radius,
            gamma_value=min(gamma_values),
            smooth_continuation_power=self.smooth_continuation_power,
        )
        # print("rotation_weight", rotation_weight)
        # gamma_weights = gamma_weights * rotation_weight

        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_id=self._BASE_VEL_ID,
            direction=velocity,
        )
        self._tangent_tree.add_node(
            parent_id=self._BASE_VEL_ID,
            node_id=self.obstacle.root_id,
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

        while parents_tree[-1] != self.obstacle.root_id:
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


def test_triple_ellipse_environment(visualize=False, savefig=False):
    triple_ellipses = MultiEllipseObstacle()
    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([-3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        )
    )

    triple_ellipses.set_root(obs_id=0)
    triple_ellipses.set_parent(obs_id=1, parent_id=0)
    triple_ellipses.set_parent(obs_id=2, parent_id=0)

    multibstacle_avoider = MultiObstacleAvoider(obstacle=triple_ellipses)

    velocity = np.array([1.0, 0])
    linearized_velociy = np.array([1.0, 0])

    if visualize:
        x_lim = [-14, 14]
        y_lim = [-8, 16]
        fig, ax = plt.subplots(figsize=(8, 5))

        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            # reference_point_number=True,
            show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                triple_ellipses.get_gamma(x, in_global_frame=True) <= 1
            ),
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            n_grid=30,
            # vectorfield_color=vf_color,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_sidewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-3.37931034, 0.27586207])
    gamma_value = triple_ellipses.get_gamma(position, in_global_frame=True)
    assert gamma_value <= 1, "Is in one of the obstacles"

    # Testing various position around the obstacle
    position = np.array([-1.5, 5])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, 5])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0

    position = np.array([5.5, 5])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, -0.9])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0


def test_tripple_ellipse_in_the_face(visualize=False, savefig=False):
    triple_ellipses = MultiEllipseObstacle()

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([3.4, 4.0]),
            axes_length=np.array([9.0, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 7.8]),
            axes_length=np.array([8, 3.0]),
            orientation=0 * math.pi / 180.0,
        )
    )

    triple_ellipses.set_root(obs_id=0)
    triple_ellipses.set_parent(obs_id=1, parent_id=0)
    triple_ellipses.set_parent(obs_id=2, parent_id=0)

    multibstacle_avoider = MultiObstacleAvoider(obstacle=triple_ellipses)

    velocity = np.array([1.0, 0.0])
    linearized_velociy = np.array([1.0, 0.0])

    if visualize:
        # figsize=(7, 8)
        # x_lim = [-6, 6]
        # y_lim = [-3.8, 11]

        # figsize = (5, 6)
        # x_lim = [-7, 7]
        # y_lim = [-5, 12.5]

        figsize = (10, 5)
        x_lim = [-12, 12]
        y_lim = [-5, 12.5]

        # n_grid = 120
        n_grid = 20
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=True,
            # reference_point_number=True,
            # show_obstacle_number=True,
            # ** kwargs,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_facewards"
            plt.savefig(
                "figures/" + "obstacles_only_" + figname + figtype,
                bbox_inches="tight",
            )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                triple_ellipses.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
            # vectorfield_color=vf_color,
        )

        if savefig:
            figname = "triple_ellipses_obstacle_facewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-5.0, 0.5])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([6.0, 6.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5.0, 9.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0


def test_orthonormal_tangent_finding():
    # Do we really want this test (?!) ->  maybe remove or make it better
    normal = np.array([0.99306112, 0.11759934])
    reference = np.array([-0.32339489, -0.9462641])

    initial = np.array([0.93462702, 0.35562949])
    tangent_rot = RotationalAvoider.get_projected_tangent_from_vectors(
        initial,
        normal=normal,
        reference=reference,
    )

    tangent_matr = MultiEllipseObstacle.get_normalized_tangent_component(
        initial, normal=normal, reference=reference
    )
    assert np.allclose(tangent_rot, tangent_matr)

    normal = np.array([0.99306112, 0.11759934])
    reference = np.array([-0.32339489, -0.9462641])
    initial = np.array([0.11759934, -0.99306112])
    tangent_rot = RotationalAvoider.get_projected_tangent_from_vectors(
        initial,
        normal=normal,
        reference=reference,
    )

    tangent_matr = MultiEllipseObstacle.get_normalized_tangent_component(
        initial, normal=normal, reference=reference
    )
    assert np.allclose(tangent_rot, tangent_matr)


def test_tree_with_two_children(visualize=False, savefig=False):
    """This is a rather uncommon configuration as the vectorfield has to traverse back
    since the root obstacle is not at the center."""
    triple_ellipses = MultiEllipseObstacle()
    triple_ellipses.append(
        Ellipse(
            center_position=np.array([-3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        )
    )

    triple_ellipses.set_root(obs_id=0)
    triple_ellipses.set_parent(obs_id=1, parent_id=0)
    triple_ellipses.set_parent(obs_id=2, parent_id=1)

    multibstacle_avoider = MultiObstacleAvoider(obstacle=triple_ellipses)

    velocity = np.array([1.0, 0])
    linearized_velociy = np.array([1.0, 0])

    if visualize:
        figsize = (10, 5)
        x_lim = [-12, 12]
        y_lim = [-5, 12.5]

        n_grid = 40
        fig, ax = plt.subplots(figsize=figsize)

        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=True,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                triple_ellipses.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
            # vectorfield_color=vf_color,
        )

        if savefig:
            figname = "triple_ellipses_childeschild_obstacle_facewards"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    # On the surface of the first obstacle
    position = np.array([-4.84, 4.68])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0

    # On the surface of the last one -> velocity has to go up to the root (!)
    position = np.array([2.15, 5.77])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] < 0 and averaged_direction[1] < 0


def test_rectangle_obstacle():
    """This is a rather uncommon configuration as the vectorfield has to traverse back
    since the root obstacle is not at the center."""
    triple_ellipses = MultiEllipseObstacle()
    triple_ellipses.append(
        Ellipse(
            center_position=np.array([-3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        )
    )

    triple_ellipses.set_root(obs_id=0)
    triple_ellipses.set_parent(obs_id=1, parent_id=0)
    triple_ellipses.set_parent(obs_id=2, parent_id=1)


if (__name__) == "__main__":
    # figtype = ".png"
    figtype = ".pdf"

    import matplotlib.pyplot as plt

    from dynamic_obstacle_avoidance.visualization import plot_obstacles
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )

    # plt.close("all")
    plt.ion()

    test_tree_with_two_children(visualize=False, savefig=False)
    test_orthonormal_tangent_finding()
    test_tripple_ellipse_in_the_face(visualize=True, savefig=False)
    test_triple_ellipse_environment(visualize=False)

    print("Tests done.")
