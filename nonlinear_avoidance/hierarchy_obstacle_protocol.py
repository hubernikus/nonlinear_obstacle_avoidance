from typing import Optional, Protocol

from dynamic_obstacle_avoidance.obstacles import Obstacle


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

    # def get_gamma(self) -> float:
    #     ...
