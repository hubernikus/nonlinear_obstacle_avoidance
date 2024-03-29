""" Container to describe obstacles & wall environment"""
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021
import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import (
    ConstantValue,
    LinearSystem,
    LocallyRotated,
)
from vartools.directional_space import get_angle_space
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import BaseContainer


class RotationContainer(BaseContainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convergence_dynamics = [None for ii in range(len(self))]
        self.convergence_radiuses: list[float] = []

    def append(self, value, convergence_radius: Optional[float] = None):
        super().append(value)
        self._convergence_dynamics.append(None)

    def __delitem__(self, key):
        """Obstacle is not part of the workspace anymore."""
        super().__delitem__(self._obstacle_list[key])
        del self._convergence_dynamics

    def __setitem__(self, key, value):
        super().__setitem__(self._obstacle_list[key])
        self._convergence_dynamics[key] = None

    def set_convergence_directions(
        self, nonlinear_dynamics=None, converging_dynamics=None
    ):
        """Define a convergence direction / mode.
        It is implemented as 'locally-linear' for a multi-boundary-environment.

        Parameters
        ----------
        attractor_position: if non-none value: linear-system is chosen as desired function
        dynamical_system: if non-none value: linear-system is chosen as desired function
        """
        if converging_dynamics is not None:
            # Converging dynamics are for all the same
            self._convergence_dynamics = [
                converging_dynamics for ii in range(self.n_obstacles)
            ]
            return

        if nonlinear_dynamics.attractor_position is None:
            # WARNING: non-knowlege about local topology leads to weird behavior (!)
            for it_obs in range(self.n_obstacles):
                position = self[it_obs].center_position
                local_velocity = nonlinear_dynamics.evaluate(position)
                if np.linalg.norm(local_velocity):  # Nonzero
                    self._convergence_dynamics[it_obs] = ConstantValue(
                        velocity=local_velocity
                    )
                else:
                    # Make converge towards center
                    self._convergence_dynamics[it_obs] = LinearSystem(
                        attractor_position=position
                    )

            return

        attractor = nonlinear_dynamics.attractor_position

        for it_obs in range(self.n_obstacles):
            position = self[it_obs].center_position
            pos_norm = LA.norm(position)
            if not pos_norm:
                # Make it converge to attractor either way, as evaluation might
                # be numerically bad.
                self._convergence_dynamics[it_obs] = LinearSystem(
                    attractor_position=attractor
                )
                continue

            local_velocity = nonlinear_dynamics.evaluate(position)
            dot_prod = np.dot(position, local_velocity) / (
                pos_norm * LA.norm(local_velocity)
            )

            if dot_prod == (-1):
                # Simpler and faster DS -> no rotation need, hence linear is sufficient
                self._convergence_dynamics[it_obs] = LinearSystem(
                    attractor_position=attractor
                )
                continue

            # Nonzero / not at attractor
            reference_radius = self[it_obs].get_characteristic_length()

            ds_direction = get_angle_space(
                direction=local_velocity,
                null_direction=(attractor - position),
            )

            self._convergence_dynamics[it_obs] = LocallyRotated(
                max_rotation=ds_direction,
                influence_pose=ObjectPose(position=position),
                influence_radius=reference_radius,
                attractor_position=attractor,
            )

    def get_convergence_direction(self, position, it_obs: int) -> np.ndarray:
        """Return 'convergence direction' at input 'position'."""
        # if self._convergence_dynamics[it_obs] is not None:
        return self._convergence_dynamics[it_obs].evaluate(position)
        # else:
        #     return self.attractor_position - position

    def get_intersection_position(self, it_obs):
        """Get the position where two boundary-obstacles intersect."""
        if hasattr(self._convergence_dynamics[it_obs], "attractor_position"):
            return self._convergence_dynamics[it_obs].attractor_position

        else:
            raise NotImplementedError(
                "Create 'intersection-with-surface' from local DS"
            )
