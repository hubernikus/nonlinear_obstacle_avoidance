from vartools.dynamics import Dynamics

import numpy as np
import numpy.typing as npt

from nonlinear_avoidance.vector_rotation import VectorRotationSequence


class ConstantValueWithSequence(Dynamics):
    """Returns constant velocity based on the DynamicalSystem parent-class"""

    def __init__(self, velocity):
        self.constant_velocity = velocity

    def evaluate(self, *args, **kwargs):
        """Random input arguments, but always ouptuts same vector-field"""
        return self.constant_velocity

    def evaluate_dynamics_sequence(self, position: np.ndarray):
        velocity = self.evaluate(position)

        rotation = VectorRotationSequence.create_from_vector_array(
            np.vstack((velocity, velocity)).T
        )
        return rotation
