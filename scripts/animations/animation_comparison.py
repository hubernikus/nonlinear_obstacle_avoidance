import os
import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.animator import Animator
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.comparison.comparison_multiobstacle_vectorfield import (
    plot_trajectory_comparison,
    create_six_obstacle_environment,
    create_initial_dynamics,
)


class ComparisonAnimator(Animator):
    def setup(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 5))

        colors = ["red", "green", "blue", "gray"]
        datafolders = [
            "nonlinear_avoidance",
            "modulation_avoidance",
            "guiding_field",
            "original_trajectories",
        ]
        labels = ["ROAM", "MuMo", "VC-CAPF", "Original"]

        self.my_animators = []
        for color, folder, label in zip(colors, datafolders, labels):
            my_animator = ComparisonSingleAnimator(
                dt_sleep=0.005, dt_simulation=0.005, it_max=501
            )
            my_animator.setup(folder, color, datapath, shared_figure=True)
            self.my_animators.append(my_animator)

    def update_step(self, ii):
        self.ax.clear()

        for my_animator in self.my_animators:
            my_animator.update_step(ii=ii, ax=self.ax)

        plot_obstacles(
            ax=self.ax,
            obstacle_container=my_animator.obstacle_environment,
            x_lim=my_animator.x_lim,
            y_lim=my_animator.y_lim,
            noTicks=True,
            # show_ticks=False,
        )


class ComparisonSingleAnimator(Animator):
    def setup(
        self,
        datafolder: str,
        trajectory_color: str,
        datapath: str,
        shared_figure=False,
        figsize: tuple[float, float] = (6, 5),
    ):
        if not shared_figure:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=200)
            self.fig.tight_layout()

        self.trajectory_color = trajectory_color

        # Create base environment

        initial_ds = create_initial_dynamics()

        line_kwargs = {
            "linewidth": 2,
            # "linestyle": "dashed"
        }

        # Create legend
        # for ii, color in enumerate(colors):
        #     ax.plot([], [], colors[ii], **line_kwargs, label=labels[ii])
        # ax.legend(loc="upper left")

        if datafolder in [
            "nonlinear_avoidance",
            "modulation_avoidance",
            "guiding_field",
        ]:
            filename = datafolder + "_" + "cycle.csv"
            self.limit_trajectory = np.loadtxt(
                os.path.join(datapath, filename),
                delimiter=",",
                dtype=float,
                skiprows=0,
            ).T
            self.obstacle_environment = create_six_obstacle_environment()
        else:
            radius = 2.0
            angles = np.linspace(0, 2 * math.pi, 100)
            self.limit_trajectory = np.vstack((np.cos(angles), np.sin(angles))) * radius
            # Empty container
            self.obstacle_environment = MultiObstacleContainer()
        # ax.plot(trajectory[0, :], trajectory[1, :], colors[ii], **line_kwargs)

        # Plot base circle
        angles = np.linspace(0, 2 * math.pi, 100)
        self.traj_base = np.vstack((np.cos(angles), np.sin(angles))) * initial_ds.radius
        # self.ax.plot(
        #     self.traj_base[0, :],
        #     self.traj_base[1, :],
        #     color="gray",
        #     **line_kwargs,
        #     zorder=-1,
        # )

        self.center_position = np.zeros(2)

        positions = [
            [-0.4, 1.25],
            # [-0.5, -1.4],
            [-0.45, -0.44],
            # [0.45, 0.4],
            # [-3.07, 1.31],
            # [3.13, -1.35],
            [2.25, 3.94],
            # [-1.33, -3.94],
            # [-4, -4],
            [-2.2, -4],
            # [4, -2.22],
            [4, -4],
            [4, 0.41],
            [-4, 0.4],
            [0.48, -0.46],
            # Additional
            [0, 0.5],
            [0, -3],
            [-3, 3],
            [0, 3],
            [-3, -2],
        ]

        start_positions = np.loadtxt(
            os.path.join(datapath, "initial_positions.csv"),
            delimiter=",",
            dtype=float,
            skiprows=1,
        )
        start_positions = start_positions.T

        self.all_trajectories = []

        abs_tol = 0.4
        indexes = np.zeros(len(positions), dtype=int)
        for pp, pos in enumerate(positions):
            value_index = np.ones(start_positions.shape[1], dtype=bool)
            indexes[pp] = np.argmin(
                np.linalg.norm(
                    start_positions - np.tile(pos, (start_positions.shape[1], 1)).T,
                    axis=0,
                )
            )
            filename = f"trajectory{indexes[pp]:03d}.csv"
            # for aa, folder in enumerate(datafolders):
            trajectory = np.loadtxt(
                os.path.join(datapath, datafolder, filename),
                delimiter=",",
                dtype=float,
                skiprows=0,
            )

            self.all_trajectories.append(trajectory.T)
            # ax.plot(trajectory[0, -1], trajectory[1, -1], ".", color=colors[aa])

            # They all have the same start position

        # visualize_trajectories(
        #     datapath, datafolder="nonlinear_avoidance", uni_color="blue", ax=ax
        # )
        # visualize_trajectories(
        #     datapath, datafolder="modulation_avoidance", uni_color="red", ax=ax
        # )
        # visualize_trajectories(
        #     datapath, datafolder="guiding_field", uni_color="green", ax=ax
        # )

        self.x_lim = [-4, 4]
        self.y_lim = [-4, 4]

    def update_step(self, ii: int, ax=None):
        if ax is None:
            ax = self.ax
            ax.clear()

            plot_obstacles(
                ax=ax,
                obstacle_container=self.obstacle_environment,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                noTicks=True,
                # show_ticks=False,
            )

        # self.ax.clear()
        # print("ii", ii)

        for trajectory in self.all_trajectories:
            ax.plot(
                trajectory[0, :ii],
                trajectory[1, :ii],
                self.trajectory_color,
                linewidth=2,
                # linestyle=":",
                zorder=0,
            )

            it = min(ii, trajectory.shape[1] - 1)
            ax.plot(trajectory[0, it], trajectory[1, it], ".", color="black")

        ax.plot(
            self.limit_trajectory[0, :],
            self.limit_trajectory[1, :],
            color=self.trajectory_color,
            # color="black",
            linestyle=":",
            alpha=0.5,
        )
        ax.plot(self.center_position[0], self.center_position[1], "*", color="black")

        ax.grid(True, zorder=-2)


def main(save_animation=None) -> None:
    datapath = "/home/lukas/Code/nonlinear_obstacle_avoidance/nonlinear_avoidance/comparison/data/"

    colors = ["red", "green", "blue", "gray"]
    datafolders = [
        "nonlinear_avoidance",
        "modulation_avoidance",
        "guiding_field",
        "original_trajectories",
    ]
    labels = ["ROAM", "MuMo", "VC-CAPF", "Original"]

    it = 0
    for color, folder, label in zip(colors, datafolders, labels):
        # if it < 0:
        #     it += 1
        #     continue
        my_animator = ComparisonSingleAnimator(
            dt_sleep=1e-5,
            dt_simulation=1 / 60.0,
            it_max=501,
            animation_name=f"comparison_multi_obstacle_{label}",
            file_type=".gif",
        )
        my_animator.setup(folder, color, datapath, figsize=(4.0, 3.5))
        my_animator.run(save_animation=save_animation)

        print("Done one")
        # break

    print("Now were out.")


def run_single(save_animation=False) -> None:
    datapath = "/home/lukas/Code/nonlinear_obstacle_avoidance/nonlinear_avoidance/comparison/data/"

    colors = [
        # "red", "green", "blue",
        "gray"
    ]
    datafolders = [
        # "nonlinear_avoidance",
        # "modulation_avoidance",
        # "guiding_field",
        "original_trajectories",
    ]
    labels = [
        # "ROAM", "MuMo", "VC-CAPF",
        "Original"
    ]

    it = 0
    for color, folder, label in zip(colors, datafolders, labels):
        # if it < 0:
        #     it += 1
        #     continue
        my_animator = ComparisonSingleAnimator(
            dt_sleep=1e-5,
            dt_simulation=1 / 60.0,
            it_max=501,
            animation_name=f"comparison_multi_obstacle_{label}",
            file_type=".gif",
        )
        my_animator.setup(folder, color, datapath, figsize=(4.0, 3.5))
        my_animator.run(save_animation=save_animation)

        print("Done one")
        # break

    print("Now were out.")


def run_all_animation_in_figure() -> None:
    my_animator = ComparisonAnimator(dt_sleep=0.005, dt_simulation=1e-2, it_max=501)
    my_animator.setup()
    my_animator.run()


if (__name__) == "__main__":
    plt.close("all")
    # main(save_animation=True)
    run_single(save_animation=True)
