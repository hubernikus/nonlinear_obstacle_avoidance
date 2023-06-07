from typing import Optional

import numpy as np

from vartools.dynamics import Dynamics
from nonlinear_avoidance.vector_rotation import VectorRotationSequence


def evaluate_dynamics_sequence(
    position: np.ndarray, dynamics: Dynamics
) -> Optional[VectorRotationSequence]:
    """Evaluate dynamical system as a sequence of 'rotations'."""
    if hasattr(dynamics, "evaluate_dynamics_sequence"):
        return dynamics.evaluate_dynamics_sequence(position)

    # Otherwise create based on direction towards stable-attractor
    base = dynamics.attractor_position - position
    final = dynamics.evaluate(position)

    if not np.linalg.norm(base):
        return None

    rotation = VectorRotationSequence.create_from_vector_array(
        np.vstack((base, final)).T
    )

    if np.any(np.isnan(rotation.rotation_angles)):
        breakpoint()

    return rotation
