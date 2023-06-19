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
    if hasattr(dynamics, "attractor_position"):
        base = dynamics.attractor_position - position
        final = dynamics.evaluate(position)
    else:
        base = dynamics.evaluate(position)
        final = base

    if not np.linalg.norm(base):
        return None

    rotation = VectorRotationSequence.create_from_vector_array(
        np.vstack((base, final)).T
    )

    if np.any(np.isnan(rotation.rotation_angles)):
        breakpoint()  # TODO: remove after debugging
    return rotation
