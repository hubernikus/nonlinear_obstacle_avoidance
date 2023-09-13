from typing import Optional

import numpy as np

from vartools.dynamics import Dynamics
from nonlinear_avoidance.vector_rotation import VectorRotationSequence


def evaluate_dynamics_sequence(
    position: np.ndarray,
    dynamics: Dynamics,
    default_dynamics: Optional[Dynamics] = None,
) -> Optional[VectorRotationSequence]:
    """Evaluate dynamical system as a sequence of 'rotations'."""
    if hasattr(dynamics, "evaluate_dynamics_sequence"):
        return dynamics.evaluate_dynamics_sequence(position)

    if default_dynamics is not None:
        base = default_dynamics.evaluate(position)
        final = dynamics.evaluate(position)

    elif hasattr(dynamics, "attractor_position"):
        base = dynamics.attractor_position - position
        final = dynamics.evaluate(position)

    else:
        final = dynamics.evaluate(position)
        # Otherwise create based on direction towards stable-attractor
        base = final

    if not np.linalg.norm(base) or not np.linalg.norm(final):
        return None

    rotation = VectorRotationSequence.create_from_vector_array(
        np.vstack((base, final)).T
    )

    if np.any(np.isnan(rotation.rotation_angles)):
        breakpoint()  # TODO: remove after debugging
    return rotation
