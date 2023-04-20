import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamics import Dynamics
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)


def rotate(orientation, vector: np.ndarray) -> np.ndarray:
    cos_ = math.cos(orientation)
    sin_ = math.sin(orientation)

    return np.array([[cos_, sin_], [-sin_, cos_]]) @ vector


@dataclass
class DynamicsSegment:
    start: np.ndarray
    end: np.ndarray

    # Defined in post-init
    direction: np.ndarray = None
    length: np.ndarray = None

    def __post_init__(self):
        self.start = np.array(self.start)
        self.end = np.array(self.end)

        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)

        if not (self.length):
            raise ValueError("Points are the same")

        self.direction = self.direction / self.length
        self.orientation = np.arctan2(self.direction[1], self.direction[0])

    def get_distance(self, position: np.ndarray) -> np.ndarray:
        tt = (position - self.start) @ self.direction
        tt = min(self.length, max(tt, 0))
        projection = self.start + self.direction * tt

        return np.linalg.norm(position - projection)

    def evaluate(self, position):
        tt = (position - self.start) @ self.direction
        projection = self.start + self.direction * tt
        dir_perp = projection - position

        vector = dir_perp + self.direction
        vector = vector / np.linalg.norm(vector)
        # return rotate(self.orientation, vector)
        return vector


class RotationTransformation2D:
    center: np.ndarray

    def get_weight(self, position):
        pass

    def inverse_transform_position(self, position):
        relative_position = position - self.center

    def transform_velocity(self, position, velocity):
        pass


class WavyPathFollowing(Dynamics):
    def __init__(self):
        self.segments = []
        self.segments.append(DynamicsSegment([0.0, 0], [3.0, 0]))
        self.segments.append(DynamicsSegment([4.0, 1], [4.0, 5]))
        self.segments.append(DynamicsSegment([5.0, 6], [8.0, 6.0]))

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def dimension(self):
        return 2

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        distances = np.zeros(self.n_segments)

        for ii, segment in enumerate(self.segments):
            distances[ii] = segment.get_distance(position)

        zero_dist = distances == 0
        if np.any(zero_dist):
            weights = 1.0 * zero_dist
        else:
            weight_power = 2
            weight_factor = 4
            weights = (weight_factor / distances) ** weight_power

        if (weight_sum := np.sum(weights)) > 1:
            weights = weights / np.sum(weights)
        else:
            # Add everything to last one
            weights[-1] = weights[-1] + (1 - weight_sum)

        directions = np.zeros((self.dimension, self.n_segments))
        for ii, segment in enumerate(self.segments):
            directions[:, ii] = segment.evaluate(position)

        # directional sum (!)
        averaged_direction = get_directional_weighted_sum(
            null_direction=self.segments[-1].direction,
            weights=weights,
            directions=directions,
        )

        return averaged_direction


def main(n_grid=30):
    x_lim = [-1, 10]
    y_lim = [-1, 10]

    dynamics = WavyPathFollowing()

    fig, ax = plt.subplots(figsize=(6, 5))

    for segment in dynamics.segments:
        ax.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            marker="o",
        )
    # position = np.array([4.0, 0])
    position = np.array([5.2, 7.75])
    velocity = dynamics.evaluate(position)
    position = np.array([5.2, 8.25])
    velocity = dynamics.evaluate(position)

    plot_obstacle_dynamics(
        obstacle_container=[],
        dynamics=dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=n_grid,
        ax=ax,
        # attractor_position=dynamic.attractor_position,
        do_quiver=True,
        # show_ticks=False,
    )

    # position =


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    main()
