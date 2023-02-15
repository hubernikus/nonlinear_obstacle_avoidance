""" Sipmlest of all rigid body - the datatype we are expect to receive.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RigidBody:
    """Optitrack RigidBody as recieved from the interace."""

    obs_id: int
    position: np.ndarray
    rotation: Rotation
