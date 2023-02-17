"""
The :mod:`avoidance` with various types of base-dynamics.
"""

# Various Obstacle Descriptions
from .rotational_avoidance import obstacle_avoidance_rotational
from .rotational_avoider import RotationalAvoider

__all__ = [
    "obstacle_avoidance_rotational",
    "RotationalAvoider",
]
