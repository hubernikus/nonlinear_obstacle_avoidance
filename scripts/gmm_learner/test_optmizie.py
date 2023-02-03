#
# Trial and error to create the correct optimization problem
#

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize


def sed_solver_mse():
    pass


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def input_value(x):
    pass


def penalty_function(x):
    pass


if (__name__) == "__main__":
    x0 = np.array([1.3, 0.7, 0.1, 1.9, 1.2])

    res = minimize(
        rosen, x0, method="nelder-mead", options={"xatol": 1e-8, "disp": True}
    )

    pass
