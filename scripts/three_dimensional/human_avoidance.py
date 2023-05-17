from dataclasses import dataclass

import numpy as np

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

from vartools.states import Pose

from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics

Vector = np.ndarray


def get_perpendicular_vector(initial: Vector, nominal: Vector) -> Vector:
    perp_vector = initial - (initial @ nominal) * nominal

    if not (perp_norm := np.linalg.norm(perp_vector)):
        return np.zeros_like(initial)

    return perp_vector / perp_norm


@dataclass
class SpiralingDynamics3D:
    """Dynamics consisting of 2D rotating and linear direction."""

    center: Vector
    direction: Vector
    circular_dynamics: SimpleCircularDynamics

    @classmethod
    def create_from_direction(cls, center: Vector, direction: Vector) -> Self:
        circular_dynamics = SimpleCircularDynamics(pose=Pose(position=center))
        return cls(center, direction, circular_dynamics)

    def evaluate(self, position: Vector):
        pass


def main():
    print("Done")


if (__name__) == "__main__":
    main()
