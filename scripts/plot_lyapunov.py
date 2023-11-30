import numpy as np

import matplotlib.pyplot as plt

from vartools.dynamics import LinearSystem


def lyapunov_function(xx):
    return xx @ xx


def main():
    dynamics_angle = 0.3 * np.pi
    cos_ = np.cos(dynamics_angle)
    sin_ = np.sin(dynamics_angle)

    rot_matrix = np.array([[cos_, sin_], [-cos_, sin_]])
    linear_dynamics = LinearSystem(attractor_position=np.zeros(2), A_matrix=rot_matrix)


if (__name__) == "__main__":
    main()
