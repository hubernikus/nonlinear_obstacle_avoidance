from __future__ import annotations  # To be removed in future python versions
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from vartools.states import Pose
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle

from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics

Vector = np.ndarray


@dataclass
class SpiralingDynamics3D:
    """Dynamics consisting of 2D rotating and linear direction.

    The dynamics are spiraling around the center in the y-z plane, while mainting"""

    pose: Pose
    direction: Vector
    circular_dynamics: SimpleCircularDynamics = SimpleCircularDynamics(
        pose=Pose.create_trivial(2)
    )
    speed: float = 1.0

    @classmethod
    def create_from_direction(
        cls,
        center: Vector,
        direction: Vector,
        radius: float = 1.0,
        speed: float = 1.0,
    ) -> Self:
        direction = direction
        basis = get_orthogonal_basis(direction)
        rotation = Rotation.from_matrix(basis)

        circular = SimpleCircularDynamics(pose=Pose.create_trivial(2), radius=radius)

        # rotation = Rotation.from_matrix(basis.T)
        return cls(Pose(center, rotation), direction, circular, speed)

    def evaluate(self, position: Vector) -> Vector:
        local_position = self.pose.transform_position_to_relative(position)
        rotating_vel2d = self.circular_dynamics.evaluate(local_position[1:])

        rotating_velocity = np.hstack((0.0, rotating_vel2d))
        rotating_velocity = self.pose.transform_position_from_relative(
            rotating_velocity
        )

        combined_velocity = rotating_velocity + self.direction
        combined_velocity = combined_velocity / np.linalg.norm(combined_velocity)
        combined_velocity = combined_velocity * self.speed
        return combined_velocity

    def evaluate_convergence_around_obstacle(
        self, position: Vector, obstacle: Obstacle
    ) -> Vector:
        return self.direction


@dataclass
class SpiralingAttractorDynamics3D:
    """Dynamics consisting of 2D rotating and linear direction.

    The dynamics are spiraling around the center in the y-z plane, while mainting"""

    pose: Pose
    direction: Vector
    circular_dynamics: SimpleCircularDynamics = SimpleCircularDynamics(
        pose=Pose.create_trivial(2)
    )
    speed: float = 1.0

    @property
    def attractor_position(self):
        return self.pose.position

    @attractor_position.setter
    def attractor_position(self, value: np.ndarray) -> None:
        self.pose.position = value

    @classmethod
    def create_from_direction(
        cls,
        center: Vector,
        direction: Vector,
        radius: float = 1.0,
        speed: float = 1.0,
    ) -> Self:
        direction = direction
        basis = get_orthogonal_basis(direction)
        rotation = Rotation.from_matrix(basis)

        circular = SimpleCircularDynamics(pose=Pose.create_trivial(2), radius=radius)

        # rotation = Rotation.from_matrix(basis.T)
        return cls(Pose(center, rotation), direction, circular, speed)

    def evaluate(self, position: Vector) -> Vector:
        # local_position = self.pose.transform_position_to_relative(position)
        local_position = position

        # Rotation in the in the y-z plane
        rotating_vel2d = self.circular_dynamics.evaluate(
            [local_position[0], local_position[2]]
        )
        rotating_velocity = np.array([rotating_vel2d[0], 0.0, rotating_vel2d[1]])
        # rotating_velocity = self.pose.transform_position_from_relative(
        #     rotating_velocity
        # )

        velocity_to_attractor = self.pose.position - position
        base_vel_norm = np.linalg.norm(velocity_to_attractor)
        if base_vel_norm > self.speed:
            velocity_to_attractor = velocity_to_attractor / base_vel_norm * self.speed

        combined_velocity = rotating_velocity + velocity_to_attractor
        combined_velocity = combined_velocity / np.linalg.norm(combined_velocity)
        combined_velocity = combined_velocity * self.speed
        return combined_velocity

    def evaluate_convergence_around_obstacle(
        self, position: Vector, obstacle: Obstacle
    ) -> Vector:
        return self.direction
