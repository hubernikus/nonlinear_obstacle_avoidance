"""
Multi Obstacle Containers
"""
# Author: Lukas Huber
# Created: 2023-05-09
# Github: huberniuks

from dataclasses import dataclass, field
from typing import Iterator
import numpy as np

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.hierarchy_obstacle_protocol import HierarchyObstacle


def plot_multi_obstacle_container(container, ax=None, **kwargs):
    for obstacle_tree in container:
        plot_obstacles(
            obstacle_container=obstacle_tree,
            ax=ax,
            **kwargs,
        )


@dataclass(slots=True)
class MultiObstacleContainer:
    _obstacle_list: list[HierarchyObstacle] = field(default_factory=list)

    def is_collision_free(self, position: np.ndarray) -> bool:
        for obstacle_tree in self._obstacle_list:
            if obstacle_tree.is_collision_free(position):
                continue
            return False

        return True

    def get_gamma(self, position: np.ndarray, in_global_frame: bool = True) -> float:
        """Returns minimum gamma across the container."""
        if not in_global_frame:
            raise NotImplementedError()
        gammas = np.zeros(len(self._obstacle_list))
        for oo, obs in enumerate(self._obstacle_list):
            gammas[oo] = obs.get_gamma(position, in_global_frame=True)

        return min(gammas)

    def get_obstacle_tree(self, index: int) -> HierarchyObstacle:
        return self._obstacle_list[index]

    def append(self, obstacle: HierarchyObstacle) -> None:
        self._obstacle_list.append(obstacle)

    def __iter__(self) -> Iterator[int]:
        return iter(self._obstacle_list)

    # def __getitem__(self, key: int) -> HierarchyObstacle:
    #     return self._obstacle_list[key]

    def get_tree(self, key: int) -> HierarchyObstacle:
        return self._obstacle_list[key]

    def __len__(self) -> int:
        return len(self._obstacle_list)

    def is_in_free_space(
        self, position: np.ndarray, in_global_frame: bool = True
    ) -> bool:
        if not in_global_frame:
            raise NotImplementedError()
        for obs in self:
            if obs.is_in_free_space(position, in_global_frame=True):
                continue
            return False

        return True
