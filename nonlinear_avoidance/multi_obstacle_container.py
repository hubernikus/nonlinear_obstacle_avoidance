"""
Multi Obstacle Containers
"""
# Author: Lukas Huber
# Created: 2023-05-09
# Github: huberniuks

from dataclasses import dataclass, field
from typing import Iterator
import numpy as np

from nonlinear_avoidance.multi_obstacle_avoider import HierarchyObstacle


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
