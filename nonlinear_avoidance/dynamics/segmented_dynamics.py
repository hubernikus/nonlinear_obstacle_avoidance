import math
from dataclasses import dataclass

import numpy as np

from vartools.dynamics import Dynamics, LinearSystem
from vartools.directional_space import get_directional_weighted_sum
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence


def rotate(orientation, vector: np.ndarray) -> np.ndarray:
    cos_ = math.cos(orientation)
    sin_ = math.sin(orientation)

    return np.array([[cos_, sin_], [-sin_, cos_]]) @ vector


@dataclass
class DynamicsSegment:
    start: np.ndarray
    end: np.ndarray

    # Defined in post-init
    # direction: np.ndarray = None
    # length: np.ndarray = None
    perpendicular_factor = 3.0

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

        vector = self.perpendicular_factor * dir_perp + self.direction
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
    def __init__(self, segments):
        self.segments = segments

        self.attractor_position = segments[-1].end
        self.maximum_velocity = 1.0
        self.slowdown_distance = 1.0

        self.global_dynamics = LinearSystem(
            attractor_position=self.attractor_position, maximum_velocity=1.0
        )

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def dimension(self):
        return 2

    def evaluate_global_dynamics(self, position: np.ndarray) -> np.ndarray:
        direction = self.attractor_position - position
        if dir_norm := np.linalg.norm(direction):
            return direction / dir_norm
        else:
            return direction

    def get_weights(
        self,
        position,
        weight_power: float = 2,
        weight_factor: float = 3,
        dotprod_power: float = 1.0,
    ) -> np.ndarray:
        # TODO: Opposite (directional singularity) reduction weights
        distances = np.zeros(self.n_segments)

        for ii, segment in enumerate(self.segments):
            distances[ii] = segment.get_distance(position)

        zero_dist = distances == 0
        if np.any(zero_dist):
            weights = 1.0 * zero_dist
            return np.append(weights, 0)

        weights = (weight_factor / distances) ** weight_power

        # Remove weight from final direction
        global_direction = self.evaluate_global_dynamics(position)
        dot_prod = np.dot(global_direction, self.segments[-1].direction)
        if dot_prod < 0:
            weights = weights * (1 + dot_prod) ** dotprod_power

        if (weight_sum := np.sum(weights)) < 1:
            return np.append(weights, (1 - weight_sum))
        else:
            weights = weights / np.sum(weights)
            return np.append(weights, 0)

    def evaluate_dynamics_sequence(
        self, position: np.ndarray
    ) -> VectorRotationSequence:
        weights = self.get_weights(position)

        direction_tree = VectorRotationTree()
        direction_tree.set_root(
            root_idx=(1, self.n_segments),
            # direction=self.segments[-1].evaluate(position)
            direction=self.evaluate_global_dynamics(position),
            # Attractor currently does not work, as the weight would need to be adapted
            # direction=self.global_dynamics.evaluate(position),
        )

        directions = np.zeros((self.dimension, self.n_segments))
        for ii, segment in reversed(list(enumerate(self.segments))):
            direction = segment.evaluate(position)

            direction_tree.add_node(
                node_id=(1, ii),
                parent_id=(1, ii + 1),
                direction=segment.direction,
            )

            direction_tree.add_node(
                node_id=ii,
                parent_id=(1, ii),
                direction=direction,
            )

        sequence = direction_tree.reduce_weighted_to_sequence(
            node_list=[ii for ii in range(self.n_segments)] + [(1, self.n_segments)],
            weights=weights,
        )
        return sequence

    def evaluate_magnitude(self, position: np.ndarray) -> float:
        dist_attr = np.linalg.norm(position - self.attractor_position)

        if dist_attr > self.slowdown_distance:
            return self.maximum_velocity

        return dist_attr / self.slowdown_distance * self.maximum_velocity

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        vel_attractor = self.evaluate_magnitude(position)
        if not np.linalg.norm(vel_attractor):
            return vel_attractor

        sequence = self.evaluate_dynamics_sequence(position)
        return sequence.get_end_vector() * vel_attractor


def create_segment_from_points(points, margin=0.5) -> WavyPathFollowing:
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


def test_archy_dynamics(visualize=False, n_grid=20):
    # TODO: create test-script
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

    #  Weight below last one
    position = np.array([3.1, -1.7])
    weights = dynamics.get_weights(position)
    assert np.isclose(weights[-1], 1, atol=0.1)
    assert np.isclose(np.sum(weights), 1)

    velocity = dynamics.evaluate(position)
    assert velocity[1] > 0, "Pointing towards attractor"

    #  Weight below last one
    position = dynamics.attractor_position + dynamics.segments[-1].direction * 5.0
    weights = dynamics.get_weights(position)
    assert np.isclose(weights[-1], 1)
    assert np.isclose(np.sum(weights), 1)

    # Zero velocity at attractor
    velocity = dynamics.evaluate(dynamics.attractor_position)
    assert np.allclose(velocity, np.zeros_like(velocity)), "Zero velocity at attractor."

    # Max weight is the second weight
    position = np.array([4.0, 0.0])
    weights = dynamics.get_weights(position)
    assert np.isclose(np.max(weights), weights[2])
    assert np.isclose(np.sum(weights), 1)

    # Velocities
    velocity = dynamics.evaluate(position)
    assert velocity[1] < 0
    assert velocity[0] < 0


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    # main()
    test_archy_dynamics(visualize=True)
