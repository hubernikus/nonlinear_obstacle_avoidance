from typing import Optional
import math

import numpy as np

# from vartools.math import get_intersection_with_circle
from vartools.linalg import get_orthogonal_basis
from vartools.angle_math import get_orientation_from_direction
from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse


def deg_to_euler(value):
    return tuple([ii * math.pi / 180.0 for ii in value])


def get_quat_from_direction(direction, null_vector=np.array([0, 0, 1.0])):
    rotation = get_orientation_from_direction(direction, null_vector=null_vector)
    quat = rotation.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]
    # mathutils.Quaternion((0.7071068, 0.0, 0.7071068, 0.0))


class DirectionalSpaceTransformer:
    def __init__(self, basis, center: Optional[np.ndarray] = None):
        self.basis = basis

        # The center 'of the directional space in 3d'
        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = center

    @classmethod
    def from_vector(cls, vector, center):
        return cls(basis=get_orthogonal_basis(vector, normalize=True), center=center)

    def transform_to_direction_space(self, direction):
        direction = np.array(direction, dtype=float)
        ang_space = get_angle_space(direction, null_matrix=self.basis, normalize=True)

        # Assuming alignement with x-y-plane
        ang_space = np.hstack(([0], ang_space))

        return ang_space + self.center

    def transform_from_direction_space(self, ang_space):
        ang_space = np.array(ang_space) - self.center
        ang_space = ang_space[1:]
        direction = get_angle_space_inverse(
            ang_space,
            null_matrix=self.basis,
        )
        return direction


# def test_quaternion():
#     quat = get_quat_from_direction([0.0, 0.0, 1.0])
#     breakpoint()


# if (__name__) == "__main__":
#     test_quaternion()
