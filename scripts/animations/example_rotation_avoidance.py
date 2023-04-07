import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence

from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer


class AnimatorRotationAvoidanceEllipse(Animator):
    # def setup(self, n_traj: int =  4):
    def setup(
        self,
        environment,
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        attractor=(8.0, 0),
        n_traj=10,
    ):
        self.n_grid = 15
        self.n_traj = n_traj
        self.attractor = np.array(attractor)

        # self.fig, self.ax = plt.subplots(figsize=(12, 9 / 4 * 3))
        self.fig, self.ax = plt.subplots(figsize=(19.20, 10.80))  # Kind-of HD

        self.environment = environment
        # self.avoider = MultiObstacleAvoider(
        #     obstacle_container=self.container,
        #     initial_dynamics=initial_dynamics,
        #     create_convergence_dynamics=True,
        # )

        self.start_positions = np.vstack(
            (
                np.ones(self.n_traj) * x_lim[0],
                np.linspace(y_lim[0], y_lim[1], self.n_traj),
            )
        )

        self.dimension = 2
        self.trajectories = []
        for tt in range(self.n_traj):
            self.trajectories.append(np.zeros((self.dimension, self.it_max + 1)))
            self.trajectories[tt][:, 0] = self.start_positions[:, tt]
        # for ii in range(self.n_traj):
        #     self.trajectory = np.zeros((self.dimension, self.it_max))

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.initial_dynamics = QuadraticAxisConvergence(
            stretching_factor=10,
            maximum_velocity=1.0,
            dimension=self.dimension,
            attractor_position=self.attractor,
        )

        self.avoider = RotationalAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.environment,
            convergence_system=LinearSystem(self.attractor),
            convergence_radius=math.pi * 0.7,
        )

        self.environment.set_convergence_directions(LinearSystem(self.attractor))

        # self.trajectory_color = "green"
        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"Iteration {ii}")
        for tt in range(self.n_traj):
            pos = self.trajectories[tt][:, ii]
            rotated_velocity = self.avoider.evaluate(pos)
            self.trajectories[tt][:, ii + 1] = (
                pos + rotated_velocity * self.dt_simulation
            )

        for obs in self.environment:
            obs.pose.position = (
                self.dt_simulation * obs.twist.linear + obs.pose.position
            )
            obs.pose.orientation = (
                self.dt_simulation * obs.twist.angular + obs.pose.orientation
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

        # Plot backgroundg
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.environment,
            x_range=self.x_lim,
            y_range=self.y_lim,
            # noTicks=True,
            showLabel=False,
            alpha_obstacle=1.0,
        )

        self.ax.scatter(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            marker="*",
            s=200,
            color="white",
            zorder=5,
        )

        plot_vectorfield = True
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=self.environment,
                collision_check_functor=lambda x: (
                    self.environment.get_minimum_gamma(x) <= 1
                ),
                dynamics=self.avoider.evaluate,
                # attractor_position=self.initial_dynamics.attractor_position,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                ax=self.ax,
                do_quiver=True,
                n_grid=self.n_grid,
                show_ticks=False,
                # vectorfield_color="#808080",
                vectorfield_color="#99ccff",
                quiver_alpha=0.5,
                quiver_scale=30,
            )

        # Plot Trajectory


def animation_ellipse(save_animation=False):
    environment = RotationContainer()
    environment.append(
        Ellipse(
            center_position=np.array([0, -0.3]),
            axes_length=np.array([3, 9]),
            orientation=0.0 / 180 * math.pi,
            is_boundary=False,
            tail_effect=False,
            distance_scaling=0.3,
            angular_velocity=0.2,
        )
    )
    animator = AnimatorRotationAvoidanceEllipse(
        dt_simulation=0.1,
        dt_sleep=0.001,
        it_max=550,
        animation_name="rotating_ellipse",
        file_type=".gif",
    )
    animator.setup(environment=environment)
    animator.run(save_animation=save_animation)


def animation_starshape(save_animation=False):
    environment = RotationContainer()
    environment.append(
        StarshapedFlower(
            radius_magnitude=1.5,
            radius_mean=3,
            number_of_edges=3,
            center_position=np.array([0, -0.15]),
            orientation=0.0 / 180 * math.pi,
            is_boundary=False,
            tail_effect=False,
            distance_scaling=0.8,
            angular_velocity=0.1,
        )
    )
    animator = AnimatorRotationAvoidanceEllipse(
        dt_simulation=0.1,
        dt_sleep=0.001,
        it_max=600,
        animation_name="rotating_starshape",
        file_type=".gif",
    )
    animator.setup(environment=environment)
    animator.run(save_animation=save_animation)


def animation_comparison_repulsion(save_animation=False):
    environment = RotationContainer()
    environment.convergence_radiuses = [math.pi / 2.0, math.pi]
    environment.append(
        Cuboid(
            center_position=np.array([0.0, 10]),
            axes_length=np.array([10, 6]),
            orientation=0.0 / 180 * math.pi,
            is_boundary=False,
            tail_effect=False,
            distance_scaling=0.3,
        )
    )

    environment.append(
        Cuboid(
            center_position=np.array([0, -10.0]),
            axes_length=np.array([10, 6]),
            orientation=0.0 / 180 * math.pi,
            is_boundary=False,
            tail_effect=False,
            distance_scaling=0.3,
        )
    )
    animator = AnimatorRotationAvoidanceEllipse(
        dt_simulation=0.2,
        dt_sleep=0.001,
        it_max=300,
        animation_name="comparison_repulsion",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        attractor=np.array([100, 0.0]),
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        n_traj=15,
    )
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    # def main():
    plt.style.use("dark_background")
    animation_ellipse(save_animation=True)
    animation_starshape(save_animation=True)
    # animation_comparison_repulsion(save_animation=True)
