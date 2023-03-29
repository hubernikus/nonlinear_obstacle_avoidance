"""
Create 'Arch'-Obstacle which might be often used
"""
import numpy as np

from vartools.states import Pose

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider


class BlockArchObstacle:
    def __init__(self, wall_width: float, axes_length: np.ndarray, pose: Pose):
        self.dimension = 0

        self.pose = pose
        self.axes_length = axes_length
        self._graph = nx.DiGraph()

        self._local_poses = []
        self._obstacle_list = []

        self._root_id = 0
        self.set_root(
            Cuboid(
                axes_length=[wall_width, axes_length[0]],
                pose=Pose(np.zeros(self.dimension), orientation=0.0),
            ),
        )

        delta_pos = (axes_length - wall_width) * 0.5
        self.add_component(
            Cuboid(
                axes_length=[wall_width, axes_length[1]],
                pose=Pose(delta_pos, orientation=0.0),
            ),
            reference_position=[0, -delta_pos],
            parent_id=0,
        )

        self.add_component(
            Cuboid(
                axes_length=[wall_width, axes_length[1]],
                pose=Pose(np.array([-delta_pos[0], delta_pos[1]]), orientation=0.0),
            ),
            reference_position=[0, -delta_pos],
            parent_id=0,
        )

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_id(self) -> int:
        return self._root_id

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_component(self, idx_obs: int) -> Obstacle:
        return self._obstacle_list[idx]

    def set_root(self, obstacle: Obstacle) -> None:
        self._obstacle_list.append(obstacle)
        obs_id = 0  # Obstacle ID
        self._graph.add_node(obs_id, references_children=[], indeces_children=[])

    def add_component(
        self,
        obstacle: Obstacle,
        reference_position: npt.ArrayLike,
        parent_id: int,
    ) -> None:
        reference_position = np.array(reference_position)
        obstacle.set_reference_point(reference_position, in_global_frame=False)

        new_id = len(self._obstacle_list) - 1
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
        # self._graph.nodes[parent_ind]["references_children"].append(
        #     self._local_poses[-1]
        # )
        self._graph.nodes[parent_ind]["indeces_children"].append(new_id)
        self._graph.add_edge(parent_ind, new_id)

    def update_obstacles(self, delta_time):
        # Update all positions of the moved obstacles
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            obs.shape = self.pose.transform_pose_from_relative(pose)


def create_arch_obstacle(
    wall_width: float, axes_length: np.ndarray, pose: Pose
) -> MultiBodyObstacle:
    multi_block = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
    )

    multi_block.set_root(
        Cuboid(axes_length=[1.0, 4.0], center_position=np.zeros(dimension)),
        name=0,
    )

    multi_block.add_component(
        Cuboid(axes_length=[4.0, 1.0], center_position=np.zeros(dimension)),
        name=1,
        parent_name=0,
        reference_position=[1.5, 0.0],
        parent_reference_position=[0.0, 1.5],
    )

    multi_block.add_component(
        Cuboid(axes_length=[4.0, 1.0], center_position=np.zeros(dimension)),
        name=1,
        parent_name=0,
        reference_position=[1.5, 0.0],
        parent_reference_position=[0.0, -1.5],
    )


def test_2d_blocky_arch(visualize=False):
    dimension: int = 2

    multi_block = BlockArchObstacle(
        wall_width=0.4,
        axes_length=np.array([3, 5]),
        pose=Pose(np.array([0, 0], orientation=0)),
    )

    multibstacle_avoider = MultiObstacleAvoider(obstacle=multi_block)

    velocity = np.array([1.0, 0])
    linearized_velociy = np.array([1.0, 0])

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))

        x_lim = [-7, 3.0]
        y_lim = [-5, 5.0]
        n_grid = 20

        plot_obstacles(
            obstacle_container=multi_block._obstacle_list,
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
                multi_block.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: multibstacle_avoider.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            # do_quiver=False,
            n_grid=n_grid,
            show_ticks=False,
            # vectorfield_color=vf_color,
        )

    # Test positions [which has been prone to a rounding error]
    position = np.array([-3.8, 1.6])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] > 0

    position = np.array([-2.0, -2.02])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert np.allclose(averaged_direction, [1, 0], atol=1e-2)
