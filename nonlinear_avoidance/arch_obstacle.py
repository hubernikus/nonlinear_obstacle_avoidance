"""
Create 'Arch'-Obstacle which might be often used
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

import networkx as nx

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

# from nonlinear_avoidance.multi_obstacle_avoider import HierarchyObstacle
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.nonlinear_rotation_avoider import (
    SingularityConvergenceDynamics,
)
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


class BlockArchObstacle:
    # TODO: remove - as a parallel factory has been created
    def __init__(
        self,
        wall_width: float,
        axes_length: np.ndarray,
        pose: Pose,
        margin_absolut: float = 0,
    ):
        self.dimension = 2

        self.pose = pose
        self.axes_length = axes_length
        self._graph = nx.DiGraph()

        self._local_poses = []
        self._obstacle_list = []

        self._root_idx = 0
        self.set_root(
            Cuboid(
                axes_length=np.array([wall_width, axes_length[1]]),
                pose=Pose(np.zeros(self.dimension), orientation=0.0),
                margin_absolut=margin_absolut,
            ),
        )

        delta_pos = (axes_length - wall_width) * 0.5
        self.add_component(
            Cuboid(
                axes_length=np.array([axes_length[0], wall_width]),
                pose=Pose(delta_pos, orientation=0.0),
                margin_absolut=margin_absolut,
            ),
            reference_position=np.array([-delta_pos[0], 0.0]),
            parent_ind=0,
        )

        self.add_component(
            Cuboid(
                axes_length=np.array([axes_length[0], wall_width]),
                pose=Pose(np.array([delta_pos[0], -delta_pos[1]]), orientation=0.0),
                margin_absolut=margin_absolut,
            ),
            reference_position=np.array([-delta_pos[0], 0.0]),
            parent_ind=0,
        )

    def get_pose(self) -> Pose:
        return self.pose

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_idx(self) -> int:
        return self._root_idx

    def __len__(self) -> int:
        return len(self._obstacle_list)

    def __iter__(self):
        return iter(self._obstacle_list)

    def __getitem__(self, idx: int) -> Obstacle:
        return self._obstacle_list[idx]

    def get_root(self) -> Obstacle:
        return self._obstacle_list[self._root_idx]

    def is_collision_free(self, position: np.ndarray) -> bool:
        for obstacle in self._obstacle_list:
            if obstacle.get_gamma(position, in_global_frame=True) <= 1:
                return False

        return True

    def get_gamma(self, position, in_global_frame: bool = True):
        if not in_global_frame:
            position = self.pose.transform_pose_from_relative(position)

        gammas = [
            obs.get_gamma(position, in_global_frame=True) for obs in self._obstacle_list
        ]
        return np.min(gammas)

    def get_gamma_except_components(
        self,
        position: np.ndarray,
        excluded_components: list[int],
        in_global_frame: bool = True,
    ) -> float:
        if not in_global_frame:
            position = self._pose.transform_pose_from_relative(position)

        gammas = []
        for ii, obs in enumerate(self._obstacle_list):
            if ii in excluded_components:
                continue

            gammas.append(obs.get_gamma(position, in_global_frame=True))

        if not len(gammas):
            raise ValueError("No components left to evaluate gamma.")

        return min(gammas)

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_component(self, idx_obs: int) -> Obstacle:
        return self._obstacle_list[idx_obs]

    def set_root(self, obstacle: Obstacle) -> None:
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self.pose.transform_pose_from_relative(self._local_poses[-1])

        self._obstacle_list.append(obstacle)
        obs_id = 0  # Obstacle ID
        self._graph.add_node(obs_id, references_children=[], indeces_children=[])

    def add_component(
        self,
        obstacle: Obstacle,
        reference_position: npt.ArrayLike,
        parent_ind: int,
    ) -> None:
        """Create and add an obstacle container in the local frame of reference."""
        reference_position = np.array(reference_position)
        obstacle.set_reference_point(reference_position, in_global_frame=False)

        new_id = len(self._obstacle_list)
        # Put obstacle to 'global' frame, but store local pose
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self.pose.transform_pose_from_relative(self._local_poses[-1])
        self._obstacle_list.append(obstacle)

        self._graph.add_node(
            new_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )
        # self._graph.nodes[parent_ind]["references_children"].append(
        #     self._local_poses[-1]
        # )
        self._graph.nodes[parent_ind]["indeces_children"].append(new_id)
        self._graph.add_edge(parent_ind, new_id)

    def update_obstacles(self, delta_time):
        # Update all positions of the moved obstacles
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            obs.shape = self.pose.transform_pose_from_relative(pose)


def create_arch_obstacle(
    wall_width: float,
    axes_length: np.ndarray,
    pose: Pose,
    margin_absolut: float = 0.0,
    distance_scaling: float = 1.0,
) -> BlockArchObstacle:
    multi_block = MultiObstacle(pose, margin_absolut)

    multi_block.set_root(
        Cuboid(
            axes_length=np.array([wall_width, axes_length[1]]),
            pose=Pose(np.zeros(multi_block.dimension), orientation=0.0),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
    )

    delta_pos = (axes_length - wall_width) * 0.5
    ref_dist = 0.5 * axes_length[0] - 0.1 * wall_width

    multi_block.add_component(
        Cuboid(
            axes_length=np.array([axes_length[0], wall_width]),
            pose=Pose(delta_pos, orientation=0.0),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
        reference_position=np.array([-ref_dist, 0.0]),
        parent_ind=0,
    )

    multi_block.add_component(
        Cuboid(
            axes_length=np.array([axes_length[0], wall_width]),
            pose=Pose(np.array([delta_pos[0], -delta_pos[1]]), orientation=0.0),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        ),
        reference_position=np.array([-ref_dist, 0.0]),
        parent_ind=0,
    )
    return multi_block
