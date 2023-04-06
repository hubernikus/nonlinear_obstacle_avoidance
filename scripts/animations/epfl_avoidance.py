import math
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt


import matplotlib.pyplot as plt

import networkx as nx

from vartools.states import Pose
from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from nonlinear_avoidance.arch_obstacle import MultiObstacleContainer
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider


def epfl_logo_parser():
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    plt.ion()

    im = plt.imread(Path("scripts") / "animations" / "epfl_logo.png")
    fig, ax = plt.subplots()
    # im = ax.imshow(im, extent=[0, 300, 0, 300])
    x_lim = [0, 640]
    y_lim = [0, 186]
    im = ax.imshow(im, extent=[0, 640, 0, 186])

    # Replicate obstacles
    obstacle_e = create_obstacle_e()
    plot_obstacles(
        obstacle_container=obstacle_e._obstacle_list, x_lim=x_lim, y_lim=y_lim, ax=ax
    )

    obstacle_p = create_obstacle_p()
    plot_obstacles(
        obstacle_container=obstacle_p._obstacle_list, x_lim=x_lim, y_lim=y_lim, ax=ax
    )
    obstacle_f = create_obstacle_f()
    plot_obstacles(
        obstacle_container=obstacle_f._obstacle_list, x_lim=x_lim, y_lim=y_lim, ax=ax
    )

    obstacle_l = create_obstacle_l()
    plot_obstacles(
        obstacle_container=obstacle_l._obstacle_list, x_lim=x_lim, y_lim=y_lim, ax=ax
    )


def create_obstacle_e(margin_absolut=0.0, scaling=1.0):
    """Letter-obstacle creator."""
    dimension = 2

    height = 186 * scaling
    breadth = 135 * scaling
    breadth_center = 126 * scaling
    wall_back = 40 * scaling
    wall_width = 35 * scaling

    letter_obstacle = MultiObstacle(
        pose=Pose(position=np.array([wall_back * 0.5, height * 0.5]))
    )
    letter_obstacle.set_root(
        Cuboid(
            axes_length=np.array([wall_back, height]),
            pose=Pose(np.zeros(dimension), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
    )

    delta_pos = np.array([breadth - wall_back, height - wall_width]) * 0.5
    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth, wall_width]),
            pose=Pose(delta_pos, orientation=0.0),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth_center, wall_width]),
            pose=Pose(
                np.array([0.5 * (breadth_center - wall_back), 0]), orientation=0.0
            ),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth, wall_width]),
            pose=Pose(np.array([delta_pos[0], -delta_pos[1]]), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    return letter_obstacle


def create_obstacle_p(margin_absolut=0.0, scaling=1.0):
    """Letter-obstacle creator."""
    dimension = 2

    height = 186 * scaling
    breadth = 135 * scaling
    breadth_center = 126 * scaling
    wall_back = 40 * scaling
    wall_width = 35 * scaling
    start_x = 169 * scaling

    p_base = 85 * scaling
    p_radius = 64 * scaling
    belly_height = height * 0.5 + wall_width * 0.5

    letter_obstacle = MultiObstacle(
        pose=Pose(position=np.array([start_x + wall_back * 0.5, height * 0.5]))
    )
    letter_obstacle.set_root(
        Cuboid(
            axes_length=np.array([wall_back, height]),
            pose=Pose(np.zeros(dimension), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
    )

    belly_center_y = (height - belly_height) * 0.5
    pose = Pose(np.array([(p_base - wall_back) * 0.5, belly_center_y]), orientation=0.0)
    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([p_base, belly_height]),
            pose=pose,
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-(p_base - wall_width) * 0.5, 0.0]),
        parent_ind=0,
    )

    letter_obstacle.add_component(
        Ellipse(
            axes_length=np.array([p_radius * 2, belly_height]),
            pose=Pose(
                np.array([p_base - wall_back * 0.5, belly_center_y]), orientation=0.0
            ),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-p_radius * 0.5, 0.0]),
        parent_ind=0,
    )

    return letter_obstacle


def create_obstacle_f(margin_absolut=0.0, scaling=1.0):
    """Letter-obstacle creator."""
    dimension = 2

    height = 186 * scaling
    breadth = 135 * scaling
    breadth_center = 126 * scaling
    wall_back = 40 * scaling
    wall_width = 35 * scaling

    start_x = 342 * scaling

    letter_obstacle = MultiObstacle(
        pose=Pose(position=np.array([start_x + wall_back * 0.5, height * 0.5]))
    )
    letter_obstacle.set_root(
        Cuboid(
            axes_length=np.array([wall_back, height]),
            pose=Pose(np.zeros(dimension), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
    )

    delta_pos = np.array([breadth - wall_back, height - wall_width]) * 0.5
    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth, wall_width]),
            pose=Pose(delta_pos, orientation=0.0),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth_center, wall_width]),
            pose=Pose(
                np.array([0.5 * (breadth_center - wall_back), 0]), orientation=0.0
            ),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    return letter_obstacle


def create_obstacle_l(margin_absolut=0.0, scaling=1.0):
    """Letter-obstacle creator."""
    dimension = 2

    height = 186 * scaling
    breadth = 135 * scaling
    # breadth_center = 126
    wall_back = 40 * scaling
    wall_width = 35 * scaling
    start_x = 505 * scaling

    letter_obstacle = MultiObstacle(
        pose=Pose(position=np.array([start_x + wall_back * 0.5, height * 0.5]))
    )
    letter_obstacle.set_root(
        Cuboid(
            axes_length=np.array([wall_back, height]),
            pose=Pose(np.zeros(dimension), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
    )

    delta_pos = np.array([breadth - wall_back, height - wall_width]) * 0.5
    letter_obstacle.add_component(
        Cuboid(
            axes_length=np.array([breadth, wall_width]),
            pose=Pose(np.array([delta_pos[0], -delta_pos[1]]), orientation=0.0),
            margin_absolut=margin_absolut,
        ),
        reference_position=np.array([-delta_pos[0], 0.0]),
        parent_ind=0,
    )

    return letter_obstacle


@dataclass
class MultiObstacle:
    pose: Pose
    margin_absolut: float = 0

    _graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    _local_poses: list[Pose] = field(default_factory=list)
    _obstacle_list: list[Obstacle] = field(default_factory=list)

    _root_idx: int = 0

    @property
    def dimension(self) -> int:
        return self.pose.dimension

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_idx(self) -> int:
        return self._root_idx

    def get_gamma(self, position, in_global_frame: bool = True):
        if not in_global_frame:
            position = self.pose.transform_pose_from_relative(position)

        gammas = [
            obs.get_gamma(position, in_global_frame=True) for obs in self._obstacle_list
        ]
        return np.min(gammas)

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_component(self, idx_obs: int) -> Obstacle:
        return self._obstacle_list[idx_obs]

    def set_root(self, obstacle: Obstacle) -> None:
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self.pose.transform_pose_from_relative(self._local_poses[-1])

        self._obstacle_list.append(obstacle)
        self._root_idx = 0  # Obstacle ID
        self._graph.add_node(
            self._root_idx, references_children=[], indeces_children=[]
        )

    def add_component(
        self,
        obstacle: Obstacle,
        reference_position: npt.ArrayLike,
        parent_ind: int,
    ) -> None:
        """Create and add an obstacle container in the local frame of reference."""
        reference_position = np.array(reference_position)
        obstacle.set_reference_point(reference_position, in_global_frame=False)

        new_id = len(self._obstacle_list)
        # Put obstacle to 'global' frame, but store local pose
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self.pose.transform_pose_from_relative(self._local_poses[-1])
        self._obstacle_list.append(obstacle)

        self._graph.add_node(
            new_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )

        self._graph.nodes[parent_ind]["indeces_children"].append(new_id)
        self._graph.add_edge(parent_ind, new_id)

    def update_obstacles(self, delta_time):
        # Update all positions of the moved obstacles
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            obs.shape = self.pose.transform_pose_from_relative(pose)


def create_epfl_multi_container(scaling=1.0, margin_absolut=0.1):
    container = MultiObstacleContainer()
    container.append(create_obstacle_e(margin_absolut, scaling))
    container.append(create_obstacle_p(margin_absolut, scaling))
    container.append(create_obstacle_f(margin_absolut, scaling))
    container.append(create_obstacle_l(margin_absolut, scaling))
    return container


def visualize_avoidance(visualize=True):
    container = create_epfl_multi_container(scaling=1.0 / 50)
    attractor = np.array([13, 4.0])
    initial_dynamics = LinearSystem(attractor_position=attractor, maximum_velocity=1.0)

    multibstacle_avoider = MultiObstacleAvoider(
        obstacle_container=container,
        initial_dynamics=initial_dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(10, 5))

        x_lim = [-2, 14.5]
        y_lim = [-2, 6.0]
        n_grid = 20

        for multi_obs in container:
            plot_obstacles(
                obstacle_container=multi_obs._obstacle_list,
                ax=ax,
                x_lim=x_lim,
                y_lim=y_lim,
                noTicks=False,
                # reference_point_number=True,
                show_obstacle_number=True,
                # ** kwargs,
            )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                container.get_gamma(x, in_global_frame=True) <= 1
            ),
            dynamics=multibstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
        )


class AnimatorRotationAvoidanceEPFL(Animator):
    def setup(self, x_lim=[-2, 14.5], y_lim=[-2, 6.0]):
        self.attractor = np.array([6, -25.0])
        self.n_traj = 20

        self.fig, self.ax = plt.subplots(figsize=(12, 9 / 4 * 3))

        self.container = create_epfl_multi_container(
            scaling=1.0 / 50, margin_absolut=0.0
        )
        # self.attractor = np.array([13, 4.0])

        self.initial_dynamics = LinearSystem(
            attractor_position=self.attractor, maximum_velocity=1.0
        )

        self.avoider = MultiObstacleAvoider(
            obstacle_container=self.container,
            initial_dynamics=self.initial_dynamics,
            create_convergence_dynamics=True,
        )

        # self.start_positions = np.vstack(
        #     (
        #         np.ones(self.n_traj) * x_lim[0],
        #         np.linspace(y_lim[0], y_lim[1], self.n_traj),
        #     )
        # )
        self.start_positions = np.vstack(
            (
                np.linspace(x_lim[0], x_lim[1], self.n_traj),
                np.ones(self.n_traj) * y_lim[1],
            )
        )

        self.n_grid = 15
        self.attractor = np.array([8.0, 0])
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.trajectories = []
        for tt in range(self.n_traj):
            self.trajectories.append(np.zeros((self.dimension, self.it_max + 1)))
            self.trajectories[tt][:, 0] = self.start_positions[:, tt]

        self.x_lim = x_lim
        self.y_lim = y_lim

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

        # for obs in self.environment:
        #     obs.pose.position = (
        #         self.dt_simulation * obs.twist.linear + obs.pose.position
        #     )
        #     obs.pose.orientation = (
        #         self.dt_simulation * obs.twist.angular + obs.pose.orientation
        #     )

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

        for multi_obs in self.container:
            plot_obstacles(
                obstacle_container=multi_obs._obstacle_list,
                ax=self.ax,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                draw_reference=False,
                draw_center=False,
                noTicks=True,
                # reference_point_number=True,
                # show_obstacle_number=True,
                # ** kwargs,
            )

            root_obs = multi_obs._obstacle_list[multi_obs.root_idx]
            reference_point = root_obs.get_reference_point(in_global_frame=True)
            self.ax.plot(
                reference_point[0],
                reference_point[1],
                "k+",
                linewidth=12,
                markeredgewidth=2.4,
                markersize=8,
                zorder=3,
            )

        self.ax.scatter(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            marker="*",
            s=200,
            color="white",
            zorder=5,
        )

        # plot_obstacle_dynamics(
        #     obstacle_container=[],
        #     collision_check_functor=lambda x: (
        #         container.get_gamma(x, in_global_frame=True) <= 1
        #     ),
        #     dynamics=multibstacle_avoider.evaluate,
        #     x_lim=x_lim,
        #     y_lim=y_lim,
        #     ax=ax,
        #     do_quiver=True,
        #     # do_quiver=False,
        #     n_grid=n_grid,
        #     show_ticks=True,
        #     # vectorfield_color=vf_color,
        # )

        # Plot Trajectory


def animation_epfl(save_animation=False):
    animator = AnimatorRotationAvoidanceEPFL(
        dt_simulation=0.07,
        dt_sleep=0.001,
        it_max=160,
        animation_name="static_circle",
        file_type=".gif",
    )
    animator.setup()
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    plt.close("all")
    # epfl_logo_parser()

    # visualize_avoidance()
    animation_epfl(save_animation=True)
