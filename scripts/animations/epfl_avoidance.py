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


def create_epfl_multi_container(scaling=1.0):
    container = MultiObstacleContainer()
    container.append(create_obstacle_e(scaling=scaling))
    container.append(create_obstacle_p(scaling=scaling))
    container.append(create_obstacle_f(scaling=scaling))
    container.append(create_obstacle_l(scaling=scaling))
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

    velocity = np.array([-1.0, 0])
    linearized_velociy = np.array([-1.0, 0])

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
                draw_reference=True,
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


if (__name__) == "__main__":
    plt.close("all")
    # epfl_logo_parser()

    visualize_avoidance()
