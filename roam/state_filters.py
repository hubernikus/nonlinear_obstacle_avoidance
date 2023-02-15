"""
Container which smoothenes position (and rotation) of incoming obstacles.
"""
import warnings

import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


def get_angular_velocity_from_quaterions(
    q1: Rotation, q2: Rotation, dt: float
) -> np.ndarray:
    """Returns the angular velocity required to reach quaternion 'q2'
    from quaternion 'q1' within the timestep 'dt'."""
    delta_q = (q2 * q1.inv()).as_quat().flatten()
    delta_q = delta_q / LA.norm(delta_q)

    delta_q_norm = LA.norm(delta_q[1:])
    delta_q_angle = 2 * np.arctan2(delta_q_norm, delta_q[0])

    return delta_q[1:] * (delta_q_angle / dt)


class SimpleOrientationFilter:
    """Very simplified rotational velocity filter"""

    def __init__(self, update_frequency: float, initial_orientation: Rotation = None):
        self._transition_weight = 0.95

        if initial_orientation is None:
            self._rotation = Rotation.from_euler("xyz", [0, 0, 0])
        else:
            self._rotation = initial_orientation
        self.angular_velocity = np.zeros(3)

        self.dt = 1 / update_frequency

    def run_once(self, rotation_measurement: Rotation):
        ang_vel_estimate = get_angular_velocity_from_quaterions(
            self._rotation,
            rotation_measurement,
            self.dt,
        )
        # Assumption of new rotation
        self._rotation = rotation_measurement

        self.angular_velocity = (
            1 - self._transition_weight
        ) * self.angular_velocity + self._transition_weight * ang_vel_estimate

    @property
    def quaternion(self) -> np.ndarray:
        """Returns [x, y, z, w] quaternion."""
        return self._rotation.as_quat()

    @property
    def rotation(self) -> Rotation:
        return self._rotation


class OrientationFilter:
    """Orientation filter easy to use with optitrack.

    Explanation found on:
    https://ahrs.readthedocs.io/en/latest/filters/ekf.html
    """

    def __init__(self, update_frequency: float = 100.0):
        # Measure Quaternion - Estimate Quaternion and Position
        # self._kf = KalmanFilter(dim_x=7, dim_z=4)
        self._kf = KalmanFilter(dim_x=7, dim_z=7)

        self.dim_quat = 4
        self.dim_vel = 3

        self.dt = 1.0 / update_frequency

        self._kf.x = np.zeros(self._kf.dim_x)

        # State transition matrix (dummy matrix to start with)
        self._kf.F = np.eye(self._kf.dim_x)

        # Measurement function (measures position only)
        # self._kf.H = np.vstack((np.eye(4), np.zeros((3, 4))))
        self._kf.H = np.eye(self._kf.dim_x)

        # Covariance matrix
        self._kf.P = np.eye(self._kf.dim_x) * 1

        # Measurement noise
        self._kf.R = np.eye(self._kf.dim_x) * 0.1

        # Process noise [it is cut to fit 4 quaternions - 3 rotation-matrix]
        Q = Q_discrete_white_noise(
            dim=2, dt=self.dt, var=1e-3, block_size=4, order_by_dim=False
        )
        self._kf.Q = Q[: self._kf.dim_x, : self._kf.dim_x]

    def run_once(self, rotation_measurement: Rotation):
        ang_vel_estimate = get_angular_velocity_from_quaterions(
            self.orientation,
            rotation_measurement,
            self.dt,
        )

        # Move to local frame
        ang_vel_estimate = self.orientation.inv().apply(ang_vel_estimate)

        self._normalize_quaternion()
        self.update_state_transition(ang_vel_estimate)

        self._kf.predict()
        self._kf.update(np.hstack((rotation_measurement.as_quat(), ang_vel_estimate)))

    def _normalize_quaternion(self) -> None:
        quat_norm = LA.norm(self.quaternion)
        if quat_norm:
            self.reset_quaternion(self.quaternion / quat_norm)
        else:
            qq = np.zeros(4)
            qq[-1] = 1
            self.reset_quaternion(qq)

    # def update_process_noise(self):
    # dim=2, dt=self.dt, var=0.01, block_size=4, order_by_dim=False

    def update_state_transition(self, velocity_estimate: np.ndarray):
        self._kf.F = np.eye(self._kf.dim_x)

        # Quaternion / Get rotation matrix
        # wx = self.velocity[0]
        # wy = self.velocity[1]
        # wz = self.velocity[2]
        wx, wy, wz = tuple(velocity_estimate)

        # Calculate omega-matrix for quaternions given by (q_x, q_y, q_z, q_w)
        Omega_t = np.zeros((4, 4))
        Omega_t[0, :] = [0, -wx, -wy, -wz]
        Omega_t[1, :] = [wx, 0, wz, -wy]
        Omega_t[2, :] = [wy, -wz, 0, wx]
        Omega_t[3, :] = [wz, wy, -wx, 0]

        # Transform (q_w, q_x, q_y, q_z) to (q_x, q_y, q_z, q_w)
        Omega_t = np.vstack((Omega_t[1:, :], Omega_t[0, :].reshape(1, -1)))
        Omega_t = np.hstack((Omega_t[:, 1:], Omega_t[:, 0].reshape(-1, 1)))

        self._kf.F[:4, :4] = self._kf.F[:4, :4] + 0.5 * self.dt * Omega_t

    @property
    def quaternion(self) -> np.ndarray:
        return self._kf.x[:4]

    def reset_quaternion(self, value: np.ndarray) -> None:
        # Flatten to matrix as scipy.transform.Rotation outputs a matrix
        self._kf.x[:4] = value.flatten()

    @property
    def orientation(self) -> Rotation:
        return Rotation.from_quat(self._kf.x[:4])

    def reset_rotation(self, value: Rotation) -> None:
        self._kf.x[:4] = value.as_quat().flatten()

    @property
    def velocity(self) -> np.ndarray:
        return self._kf.x[4:]

    def reset_velocity(self, value) -> None:
        self._kf.x[4:] = value


class PositionFilter:
    """Implementation of kalman filter for position
    -> x-y-z could be separated / simpler filter as they are independant."""

    def __init__(
        self, update_frequency: float = 100.0, initial_position: np.ndarray = None
    ):
        self.dimension = 3

        # Measure Position - Estimate Velocity
        # self._kf = KalmanFilter(dim_x=self.dimension * 2, dim_z=self.dimension)
        self._kf = KalmanFilter(dim_x=self.dimension * 2, dim_z=self.dimension * 2)

        self.dt = 1.0 / update_frequency

        self._kf.x = np.zeros(self._kf.dim_x)

        # State transition matrix
        self._kf.F = np.eye(self._kf.dim_x)
        self._kf.F[: self.dimension, self.dimension :] = (
            np.eye(self.dimension) * self.dt
        )

        # Measurement function (measures position only)
        # self._kf.H = np.hstack((np.eye(self.dimension), np.zeros(self.dimension)))
        self._kf.H = np.eye(self._kf.dim_x)

        # Covariance matrix
        self._kf.P = np.eye(self._kf.dim_x)

        # Measurement noise
        self._kf.R = np.eye(self._kf.dim_x) * 0.001

        # Process noise
        self._kf.Q = Q_discrete_white_noise(
            dim=2, dt=self.dt, var=1e-1, block_size=3, order_by_dim=False
        )
        self._kf.Q[self.dimension :, self.dimension :] = np.eye(self.dimension) * 5e-4

        if initial_position is not None:
            self.reset_position(initial_position)

    def run_once(self, position_measurement: np.ndarray) -> None:
        velocity_estimate = (position_measurement - self.position) / self.dt

        self._kf.predict()
        self._kf.update(np.hstack((position_measurement, velocity_estimate)))

    def reset_position(self, value: np.ndarray):
        self._kf.x[: self.dimension] = value

    @property
    def position(self) -> np.ndarray:
        return self._kf.x[: self.dimension]

    def reset_velocity(self, value: np.ndarray) -> None:
        self._kf.x[self.dimension :] = value.flatten()

    @property
    def velocity(self) -> np.ndarray:
        return self._kf.x[self.dimension :]


class UnfinishedFilter:
    # class UnfinishedFilter(ExtendedKalmanFilter):
    def __init__(self, update_rate: float):
        self.dim_x = 7 + 6 + 6
        # Position and Orientation
        self.dim_z = 7

        self.delta_time = 1 / update_rate

    def get_jacobian(self, pos, dt):
        J = np.eye(self.dim_x)

        # Position
        J[0:3, :][:, 3:6] = dt

        # Linear Velocity
        J[3:6, :][:, 6:9] = dt

        # Quaternion / Get rotation matrix
        q_omega_matrix = np.zeros((4, 4))
        # Column 0
        q_omega_matrix[1, 0] = self.angular_velocity[0]
        q_omega_matrix[2, 0] = self.angular_velocity[1]
        q_omega_matrix[3, 0] = self.angular_velocity[2]
        # Column 1
        q_omega_matrix[0, 1] = -self.angular_velocity[0]
        q_omega_matrix[2, 1] = -self.angular_velocity[2]
        q_omega_matrix[3, 1] = self.angular_velocity[1]
        # Column 2
        q_omega_matrix[0, 2] = -self.angular_velocity[1]
        q_omega_matrix[1, 2] = self.angular_velocity[2]
        q_omega_matrix[3, 2] = -self.angular_velocity[0]
        # Column 3
        q_omega_matrix[0, 3] = -self.angular_velocity[2]
        q_omega_matrix[1, 3] = -self.angular_velocity[1]
        q_omega_matrix[2, 3] = self.angular_velocity[0]
        # Assign
        J[9:13, :][:, 13:16] = 0.5 * q_omega_matrix * dt

        # Angular Velocity
        J[13:16, :][:, 16:19] = dt

    def _normalize_quaternion(self) -> None:
        quat_norm = LA.norm(self.quaternion)
        if quat_norm:
            self.quaternion = self.quaternion / quat_norm
        else:
            qq = np.zeros(4)
            qq[0] = 1
            self.quaternion = qq
            return

    @property
    def position(self):
        """Returns the orientation"""
        return self.x[0:3]

    @property
    def linear_velocity(self):
        return self.x[3:6]

    @property
    def linear_acceleration(self):
        return self.x[6:9]

    def get_orientation(self) -> Rotation:
        """Returns the orientation as a quaternion"""
        return Rotation.from_quat(self.x[9:13])

    @property
    def quaternion(self):
        """Returns the orientation as a quaternion"""
        return self.x[9:13]

    @quaternion.setter
    def quaternion(self, value: np.ndarray):
        """Returns the orientation as a quaternion"""
        self.x[9:13] = value

    @property
    def angular_velocity(self):
        return self.x[13:16]

    @property
    def angular_acceleration(self) -> np.ndarray:
        return self.x[16:19]


def test_position_filter(debug_print=False):
    n_measurements = 21
    pos_x = np.linspace(0, 2, n_measurements)
    pos_y = np.linspace(1, 3, n_measurements)
    pos_z = np.linspace(0, -2, n_measurements)

    position_measurements = np.vstack((pos_x, pos_y, pos_z))

    vel_estimated = [1, 1, -1]

    pos_filter = PositionFilter(update_frequency=10.0)
    pos_filter.reset_position(np.array(position_measurements[:, 0]))

    pos_filter.run_once(position_measurements[:, 0])

    # Directly after standstill, velocity is estimated to be at 0
    assert not np.allclose(vel_estimated, pos_filter.velocity, atol=1e-2)

    if debug_print:
        print("measurement", position_measurements[:, 0])
        print("position", np.round(pos_filter.position, 5))
        print("velocity", np.round(pos_filter.velocity, 5))

    for ii in range(1, position_measurements.shape[1]):
        pos_filter.run_once(position_measurements[:, ii])

        if debug_print:
            print("measurement", position_measurements[:, ii])
            print("position", np.round(pos_filter.position, 5))
            print("velocity", np.round(pos_filter.velocity, 5))

    # After many loops velocity ends up at expected
    assert np.allclose(vel_estimated, pos_filter.velocity, atol=1e-2)


def test_orientation_filter(debug_print=False):
    n_measurements = 11

    # Do the euler angles
    rho = np.linspace(0, 0, n_measurements)
    phi = np.linspace(0, 0, n_measurements)
    gamma = np.linspace(0, np.pi / 2, n_measurements)
    orientation_measurements = np.vstack((rho, phi, gamma))

    rot_filter = OrientationFilter(update_frequency=10.0)
    rot_filter.reset_rotation(
        Rotation.from_euler("zyx", orientation_measurements[:, 0])
    )
    rot_filter.run_once(Rotation.from_euler("xyz", orientation_measurements[:, 0]))
    # breakpoint()

    # Directly after standstill, velocity is estimated to be at 0
    # assert not np.allclose(vel_estimated, pos_filter.velocity, atol=1e-2)

    if debug_print:
        print("measurement", orientation_measurements[:, 0])
        print("position", np.round(rot_filter.quaternion, 5))
        print("velocity", np.round(rot_filter.velocity, 5))

    for ii in range(1, orientation_measurements.shape[1]):
        rotation = Rotation.from_euler("zyx", orientation_measurements[:, ii])
        rot_filter.run_once(rotation)

        if debug_print:
            print()
            print("measurement", rotation.as_quat())
            print("quaternion", np.round(rot_filter.quaternion, 5))
            print("ang velocity", np.round(rot_filter.velocity, 5))

    # After many loops velocity ends up at expected
    # assert np.allclose(, pos_filter.velocity, atol=1e-2)


def test_orientation_filter(debug_print=False):
    n_measurements = 11

    # Do the euler angles
    rho = np.linspace(0, 0, n_measurements)
    phi = np.linspace(0, 0, n_measurements)
    gamma = np.linspace(0, np.pi / 2, n_measurements)
    orientation_measurements = np.vstack((rho, phi, gamma))

    rot_filter = SimpleOrientationFilter(
        update_frequency=100,
        initial_orientation=Rotation.from_euler("zyx", orientation_measurements[:, 0]),
    )

    rot_filter.run_once(Rotation.from_euler("xyz", orientation_measurements[:, 0]))
    # breakpoint()

    # Directly after standstill, velocity is estimated to be at 0
    # assert not np.allclose(vel_estimated, pos_filter.velocity, atol=1e-2)

    if debug_print:
        print("measurement", orientation_measurements[:, 0])
        print("position", np.round(rot_filter.quaternion, 5))
        print("velocity", np.round(rot_filter.angular_velocity, 5))

    for ii in range(1, orientation_measurements.shape[1]):
        rotation = Rotation.from_euler("zyx", orientation_measurements[:, ii])
        rot_filter.run_once(rotation)

        if debug_print:
            print()
            print("measurement", rotation.as_quat())
            print("quaternion", np.round(rot_filter.quaternion, 5))
            print("ang velocity", np.round(rot_filter.angular_velocity, 5))

    # After many loops velocity ends up at expected
    # assert np.allclose(, pos_filter.velocity, atol=1e-2)


if (__name__) == "__main__":
    # test_position_filter(debug_print=True)
    test_orientation_filter(debug_print=True)
