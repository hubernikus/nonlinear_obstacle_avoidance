import math
from dataclasses import dataclass

import numpy as np

from vartools.dynamics import Dynamics, LinearSystem
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.vector_rotation import VectorRotationTree


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
            raise ValueError(f"Points {self.start} and {self.end} are the same")

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


def create_segment_from_points(points, margin=0.5):
    segments = []
    for pp in range(1, len(points)):
        point0 = np.array(points[pp - 1])
        point1 = np.array(points[pp])

        dir_point = point1 - point0
        if not (point_norm := np.linalg.norm(dir_point)):
            raise ValueError("Zero value.")

        dir_point = dir_point / point_norm
        point0 = point0 + dir_point * margin
        point1 = point1 - dir_point * margin

        segments.append(DynamicsSegment(point0, point1))

    return WavyPathFollowing(segments)


class WavyPathFollowing(Dynamics):
    def __init__(self, segments):
        self.segments = segments

        self.attractor_position = segments[-1].end

        self.global_dynamics = LinearSystem(
            attractor_position=self.attractor_position, maximum_velocity=1.0
        )

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def dimension(self):
        return 2

    def get_weights(self, position, weight_power=2, weight_factor=4):
        distances = np.zeros(self.n_segments)

        for ii, segment in enumerate(self.segments):
            distances[ii] = segment.get_distance(position)

        zero_dist = distances == 0
        if np.any(zero_dist):
            weights = 1.0 * zero_dist
            return np.append(weights, 0)

        weights = (weight_factor / distances) ** weight_power
        if (weight_sum := np.sum(weights)) < 1:
            return np.append(weights, (1 - weight_sum))

        weights = weights / np.sum(weights)
        return np.append(weights, 0)

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        weights = self.get_weights(position)

        direction_tree = VectorRotationTree()
        direction_tree.set_root(
            root_idx=(1, self.n_segments),
            direction=self.segments[-1].evaluate(position)
            # Attractor currently does not work, as the weight would need to be adapted
            # direction=self.global_dynamics.evaluate(position),
        )

        directions = np.zeros((self.dimension, self.n_segments))
        for ii, segment in reversed(list(enumerate(self.segments))):
            directions[:, ii] = segment.evaluate(position)

            direction_tree.add_node(
                node_id=(1, ii),
                parent_id=(1, ii + 1),
                direction=segment.direction,
            )

            direction_tree.add_node(
                node_id=ii,
                parent_id=(1, ii),
                direction=directions[:, ii],
            )

        # directional sum (!)
        averaged_direction = get_directional_weighted_sum(
            null_direction=self.segments[-1].direction,
            weights=weights[:-1],
            directions=directions,
        )

        tree_average = direction_tree.get_weighted_mean(
            node_list=[ii for ii in range(self.n_segments)] + [(1, self.n_segments)],
            weights=weights,
        )

        # return averaged_direction
        return tree_average


def test_archy_dynamics(visualize=False, n_grid=20):
    dynamics = create_segment_from_points(
        [[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]],
        margin=0.1,
    )

    if visualize:
        x_lim = [-2, 5]
        y_lim = [-2, 5]

        fig, ax = plt.subplots(figsize=(6, 5))

        for segment in dynamics.segments:
            ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                marker="o",
            )

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

    position = np.array([4.0, 0.0])
    weights = dynamics.get_weights(position)
    # Max weight is the second weight
    assert np.isclose(np.max(weights), weights[2])

    velocity = dynamics.evaluate(position)
    assert velocity[1] < 0
    assert velocity[0] < 0


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    # main()
    test_archy_dynamics(visualize=True)
