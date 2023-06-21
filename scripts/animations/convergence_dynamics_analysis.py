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


def evaluate_global_converence(save_animation=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    n_resolution = 120
    vf_color = "blue"
    dynamics, avoider = get_environment_obstacle_top_right()

    plot_obstacle_dynamics(
        obstacle_container=avoider.obstacle_environment,
        # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
        dynamics=avoider.evaluate,
        x_lim=setup["x_lim"],
        y_lim=setup["y_lim"],
        ax=ax,
        n_grid=n_resolution,
        do_quiver=False,
        show_ticks=False,
        # do_quiver=True,
        vectorfield_color=vf_color,
        attractor_position=dynamics.attractor_position,
    )

    plot_obstacles(
        ax=ax,
        obstacle_container=avoider.obstacle_environment,
        alpha_obstacle=1.0,
        draw_reference=True,
        draw_center=False,
    )

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


if (__name__) == "__main__":
    setup = {
        "attractor_color": "#BD5E11",
        "opposite_color": "#4E8212",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3ff",
        "final_color": "#30a0b3ff",
        # "figsize": (5, 4),
        "figsize": (10, 8),
        "x_lim": [-3.0, 3.4],
        "y_lim": [-3.0, 3.0],
        "n_resolution": 100,
        "n_vectors": 10,
        "linestyle": "--",
        "linewidth": 10,
        "figure_name": "linear_spiral_motion",
        "weights_alpha": 0.7,
    }

    figtype = ".pdf"

    evaluate_global_converence()
