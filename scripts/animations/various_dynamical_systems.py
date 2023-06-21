import math
import numpy as np

from typing import Optional

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.states import Pose
from vartools.dynamics import WavyRotatedDynamics
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence

from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse


from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer


class AnimatorDynamics(Animator):
    def setup(
        self,
        dynamics,
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(12, 8.0),
        attractor=None,
        n_traj: int = 10,
        start_positions: Optional[np.ndarray] = None,
    ):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=150)  # Kind-of HD
        self.x_lim = x_lim
        self.y_lim = y_lim

        if start_positions is None:
            self.n_traj = n_traj
            self.start_positions = np.vstack(
                (
                    np.ones(self.n_traj) * x_lim[0],
                    np.linspace(y_lim[0], y_lim[1], self.n_traj),
                )
            )
        else:
            self.start_positions = start_positions
            self.n_traj = self.start_positions.shape[1]

        self.n_grid = 15
        if attractor is None:
            self.attractor = np.array([8.0, 0])
        else:
            self.attractor = attractor
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.trajectories = []
        for tt in range(self.n_traj):
            self.trajectories.append(np.zeros((self.dimension, self.it_max + 1)))
            self.trajectories[tt][:, 0] = self.start_positions[:, tt]

        self.dynamics = dynamics

        # self.trajectory_color = "green"
        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"Iteration {ii}")

        for tt in range(self.n_traj):
            pos = self.trajectories[tt][:, ii]
            rotated_velocity = self.dynamics.evaluate(pos)
            self.trajectories[tt][:, ii + 1] = (
                pos + rotated_velocity * self.dt_simulation
            )

        self.ax.clear()

        for tt in range(self.n_traj):
            trajectory = self.trajectories[tt]
            self.ax.plot(
                trajectory[0, 0],
                trajectory[1, 0],
                "ko",
                linewidth=2.0,
            )
            self.ax.plot(
                trajectory[0, :ii],
                trajectory[1, :ii],
                "--",
                color=self.color_list[tt],
                linewidth=2.0,
            )
            self.ax.plot(
                trajectory[0, ii],
                trajectory[1, ii],
                "o",
                color=self.color_list[tt],
                markersize=8,
            )

        if hasattr(self.dynamics, "attractor_position"):
            self.ax.scatter(
                self.dynamics.attractor_position[0],
                self.dynamics.attractor_position[1],
                marker="*",
                s=200,
                color="white",
                zorder=5,
            )

        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=self.dynamics.evaluate,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            ax=self.ax,
            do_quiver=True,
            n_grid=self.n_grid,
            show_ticks=False,
            vectorfield_color="#808080",
            quiver_scale=30,
        )


def animation_straight_dynamics(save_animation=False):
    animator = AnimatorDynamics(
        dt_simulation=0.03,
        dt_sleep=0.03,
        it_max=170,
        animation_name="straight_dynamics",
        file_type=".gif",
    )
    animator.setup(
        figsize=(7.0, 6.0),
        x_lim=[-10, 2.0],
        y_lim=[-5, 5.0],
        dynamics=LinearSystem(
            attractor_position=np.array([0, 0]), maximum_velocity=2.0
        ),
    )
    animator.run(save_animation=save_animation)


def animation_wavy_dynamics(save_animation=False):
    dynamics = WavyRotatedDynamics(
        pose=Pose(position=[3, -3], orientation=0),
        maximum_velocity=2.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.3 * math.pi,
    )

    n_traj = 6
    x_lim = [-7, 5.0]
    y_lim = [-5, 5.0]
    start_left = np.vstack(
        (
            np.ones(n_traj) * x_lim[0],
            np.linspace(y_lim[0], y_lim[1], n_traj, endpoint=False),
        )
    )
    start_top = np.vstack(
        (
            np.linspace(x_lim[0], x_lim[1], n_traj),
            np.ones(n_traj) * y_lim[1],
        )
    )

    animator = AnimatorDynamics(
        dt_simulation=0.03,
        dt_sleep=0.03,
        it_max=250,
        animation_name="wavy_dynamics",
        file_type=".gif",
    )
    animator.setup(
        figsize=(7.0, 6.0),
        x_lim=x_lim,
        y_lim=y_lim,
        dynamics=dynamics,
        start_positions=np.hstack((start_left, start_top)),
    )

    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    # def main():
    plt.style.use("dark_background")
    animation_straight_dynamics(save_animation=True)
    animation_wavy_dynamics(save_animation=True)
