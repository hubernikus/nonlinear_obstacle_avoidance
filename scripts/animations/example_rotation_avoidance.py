import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)


from nonlinear_avoidance.arch_obstacle import MultiObstacleContainer
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer


class AnimatorRotationAvoidanceEllipse(Animator):
    # def setup(self, n_traj: int =  4):
    def setup(self, x_lim=[-10, 10], y_lim=[-10, 10]):
        self.fig, self.ax = plt.subplots(figsize=(6, 5))

        # self.avoider = MultiObstacleAvoider(
        #     obstacle_container=self.container,
        #     initial_dynamics=initial_dynamics,
        #     create_convergence_dynamics=True,
        # )
        self.n_grid = 10
        self.attractor = np.array([8.0, 0])
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.delta_time = 0.01
        # self.trajectories = [np.zeros((self.dimension, self.it_max))) for _ in range(n_traj)]

        self.trajectory = np.zeros((self.dimension, self.it_max))

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.initial_dynamics = QuadraticAxisConvergence(
            stretching_factor=3,
            maximum_velocity=1.0,
            dimension=self.dimension,
            attractor_position=self.attractor,
        )

        self.environment = RotationContainer()
        self.environment.append(
            Ellipse(
                center_position=np.array([0, 0]),
                axes_length=np.array([3, 3]),
                orientation=0.0 / 180 * math.pi,
                is_boundary=False,
                tail_effect=False,
            )
        )

        self.avoider = RotationalAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.environment,
        )

        self.environment.set_convergence_directions(LinearSystem(self.attractor))

        self.trajectory_color = "green"

    def update_step(self, ii: int) -> None:
        rotated_velocity = self.avoider.evaluate(self.position)
        self.position = self.position + rotated_velocity * self.delta_time

        self.ax.clear()

        self.trajectory[:, ii] = self.position
        self.ax.plot(self.trajectory[0, 0], self.trajectory[1, 0], "ko")
        self.ax.plot(
            self.trajectory[0, :ii],
            self.trajectory[1, :ii],
            "--",
            color=self.trajectory_color,
        )
        self.ax.plot(
            self.trajectory[0, ii],
            self.trajectory[1, ii],
            "o",
            color=self.trajectory_color,
        )

        # Plot backgroundg
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.environment,
            x_range=self.x_lim,
            y_range=self.y_lim,
            # noTicks=True,
            showLabel=False,
            alpha_obstacle=0.9,
        )

        plot_vectorfield = False
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=self.environment,
                collision_check_functor=lambda x: (
                    self.environment.get_minimum_gamma(x) <= 1
                ),
                dynamics=self.avoider.evaluate,
                attractor_position=self.initial_dynamics.attractor_position,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                ax=self.ax,
                do_quiver=True,
                n_grid=self.n_grid,
                show_ticks=True,
            )

        # Plot Trajectory


if (__name__) == "__main__":
    # def main():
    animator = AnimatorRotationAvoidanceEllipse(dt_simulation=0.2, dt_sleep=0.001)
    animator.setup()
    animator.run()
