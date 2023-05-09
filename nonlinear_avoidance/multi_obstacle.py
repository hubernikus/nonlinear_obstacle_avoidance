import copy
from typing import Optional
from dataclasses import dataclass, field

import networkx as nx

import numpy as np
from numpy import typing as npt

from vartools.states import Pose, Twist
from dynamic_obstacle_avoidance.obstacles import Obstacle


@dataclass
class MultiObstacle:
    _pose: Pose
    margin_absolut: float = 0

    _graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    _local_poses: list[Pose] = field(default_factory=list)
    _obstacle_list: list[Obstacle] = field(default_factory=list)

    _root_idx: int = 0

    twist: Optional[Twist] = None

    @property
    def dimension(self) -> int:
        return self._pose.dimension

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_idx(self) -> int:
        return self._root_idx

    def get_pose(self) -> Pose:
        """Returns a (copy) of the pose."""
        return copy.deepcopy(self._pose)

    def update_pose(self, new_pose: Pose) -> None:
        self._pose = new_pose
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            obs.pose = self._pose.transform_pose_from_relative(pose)

    def get_gamma(self, position, in_global_frame: bool = True):
        if not in_global_frame:
            position = self._pose.transform_pose_from_relative(position)

        gammas = [
            obs.get_gamma(position, in_global_frame=True) for obs in self._obstacle_list
        ]
        return np.min(gammas)

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_component(self, idx_obs: int) -> Obstacle:
        return self._obstacle_list[idx_obs]

    def set_root(self, obstacle: Obstacle) -> None:
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self._pose.transform_pose_from_relative(self._local_poses[-1])

        self._obstacle_list.append(obstacle)
        self._root_idx = 0  # Obstacle ID
        self._graph.add_node(
            self._root_idx, references_children=[], indeces_children=[]
        )

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
        obstacle.pose = self._pose.transform_pose_from_relative(self._local_poses[-1])
        self._obstacle_list.append(obstacle)

        self._graph.add_node(
            new_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )

        self._graph.nodes[parent_ind]["indeces_children"].append(new_id)
        self._graph.add_edge(parent_ind, new_id)

    # def update_obstacles(self, delta_time):
    #     # Update all positions of the moved obstacles
    #     for pose, obs in zip(self._local_poses, self._obstacle_list):
    #         obs.shape = self._pose.transform_pose_from_relative(pose)
