from __future__ import annotations  # Self typing

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt
from numpy import linalg

from scipy.spatial.transform import Rotation

import networkx as nx

from vartools.state_filters import PositionFilter, SimpleOrientationFilter
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from roam.rigid_body import RigidBody
from roam.multi_obstacle_avoider import MultiObstacleAvoider
from roam.dynamics.circular_dynamics import SimpleCircularDynamics
from roam.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def plot_3d_cuboid(ax, cube: Cuboid, color="green"):
    # TODO: include orientation
    axis = cube.axes_length
    orientation = cube.orientation

    pos_ranges = np.array(
        [
            cube.center_position - axis / 2.0,
            cube.center_position + axis / 2.0,
        ]
    ).T
    posx = pos_ranges[0, :]
    posy = pos_ranges[1, :]
    posz = pos_ranges[2, :]

    # Define the vertices of the cube
    for ii in posx:
        for jj in posy:
            ax.plot([ii, ii], [jj, jj], posz, color=color, marker="o")

    for ii in posx:
        for jj in posz:
            ax.plot([ii, ii], posy, [jj, jj], color=color, marker="o")

    for ii in posy:
        for jj in posz:
            ax.plot(posx, [ii, ii], [jj, jj], color=color, marker="o")


def plot_3d_ellipsoid(ax, ellipse: Ellipse):
    # TODO: inclde orientation?

    # your ellispsoid and center in matrix form
    diag_axes = np.diag(ellipse.axes_length)
    # dimension = 3
    # A = np.eye(dimension)
    # for dd in range(dimension):
    #     A[dd, :] =
    A = diag_axes

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    # radii = 1.0 / np.sqrt(s)
    radii = ellipse.axes_length / 2.0

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], rotation) + ellipse.center_position
            )

    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color="b", alpha=0.2)


# class HumanTrackContainer(Obstacle):
class MultiBodyObstacle:
    dimension = 3

    def __init__(
        self,
        update_frequency: float = 100.0,
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    ):
        # super().__init__(center_position=np.zeros(3))

        self._obstacle_list: list[Obstacle] = []

        self._graph = nx.DiGraph()
        self.robot = robot

        # Pose updater (Optional) can be for example an OptitrackInterface
        self.pose_updater = pose_updater

        self._id_counter = 0

        self.position_filters = []
        self.orientation_filters = []
        self.update_frequency = update_frequency

        self.visualization_handler = visualization_handler

    def __getitem__(self, key) -> Obstacle:
        return self._obstacle_list[key]

    def __setitem__(self, key: int, value: Obstacle) -> None:
        self._obstacle_list[key] = value

    def get_obstacle_id_from_name(self, name: str) -> int:
        return [x for x, y in self._graph.nodes(data=True) if y["name"] == name][0]

    def get_obstacle_id_from_optitrackid(self, opt_id: int) -> int:
        return [
            x for x, y in self._graph.nodes(data=True) if y["optitrack_id"] == opt_id
        ][0]

    def get_component(self, idx: int) -> Obstacle:
        return self._obstacle_list[idx]

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    @property
    def root_id(self) -> int:
        return self.root_idx

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    def set_root(
        self,
        obstacle: Obstacle,
        name: str,
        update_id: Optional[int] = None,
    ):
        self._obstacle_list.append(obstacle)
        self._graph.add_node(
            self._id_counter,
            name=name,
            update_id=update_id,
            references_children=[],
            indeces_children=[],
        )

        self.create_filters(is_updating=(not update_id is None))

        self._id_counter += 1
        self.root_idx = 0

    def create_filters(self, is_updating: bool):
        if is_updating:
            self.position_filters.append(
                PositionFilter(
                    update_frequency=self.update_frequency,
                    initial_position=np.zeros(3),
                )
            )
            self.orientation_filters.append(
                SimpleOrientationFilter(
                    update_frequency=self.update_frequency,
                    initial_orientation=Rotation.from_euler("x", 0),
                )
            )
        else:
            self.position_filters.append(None)
            self.orientation_filters.append(None)

    def get_update_id(self, idx_node: int) -> int:
        return self._graph.nodes[idx_node]["update_id"]

    def add_component(
        self,
        obstacle: Obstacle,
        name: str,
        reference_position: npt.ArrayLike,
        parent_name: str,
        parent_reference_position: npt.ArrayLike,
        update_id: Optional[int] = None,
    ):
        reference_position = np.array(reference_position)
        obstacle.set_reference_point(reference_position, in_global_frame=False)
        self._obstacle_list.append(obstacle)
        parent_ind = self.get_obstacle_id_from_name(parent_name)

        self._graph.add_node(
            self._id_counter,
            name=name,
            update_id=update_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )
        self._graph.nodes[parent_ind]["references_children"].append(
            np.array(parent_reference_position)
        )
        self._graph.nodes[parent_ind]["indeces_children"].append(self._id_counter)

        self._graph.add_edge(parent_ind, self._id_counter)

        self.create_filters(is_updating=(not update_id is None))
        self._id_counter += 1

    @property
    def optitrack_indeces(self):
        indeces_tree_opti = [-1] * self.n_components
        for ii in range(self.n_components):
            if self._graph.nodes[ii]["update_id"] is not None:
                indeces_tree_opti[ii] = self._graph.nodes[ii]["update_id"]

        return indeces_tree_opti

    def update_using_optitrack(self, transform_to_robot_frame: bool = True) -> None:
        if self.pose_updater is not None:
            new_object_poses = self.pose_updater.get_messages()

        else:
            new_object_poses = []
        indeces_measures = [oo.obs_id for oo in new_object_poses]
        indeces_optitrack_tree = self.optitrack_indeces

        if self.robot is not None:
            try:
                index_franka_list = indeces_measures.index(self.robot.optitrack_id)

            except ValueError:
                # Element not in list
                pass

            else:
                # So far: no filter for the robot (!)
                franka_object = new_object_poses[index_franka_list]

                self.robot.pose.position = franka_object.position
                self.robot.pose.rotation = franka_object.rotation

                # print("Robot")
                # print(self.robot.position)
                # print("Euler[Robot", self.robot.rotation.as_euler("xyz"))

        # Update filters and put to poses
        filtered_poses = [None] * self.n_components
        for ii, idx_meas in enumerate(indeces_measures):
            try:
                idx_tree = indeces_optitrack_tree.index(idx_meas)

            except:
                pass
            else:
                self.position_filters[idx_tree].run_once(new_object_poses[ii].position)
                self.orientation_filters[idx_tree].run_once(
                    new_object_poses[ii].rotation
                )

                if self.position_filters[idx_tree] is None:
                    continue

                if transform_to_robot_frame and self.robot is not None:
                    filtered_poses[idx_tree] = ObjectPose(
                        position=self.robot.pose.transform_position_to_relative(
                            self.position_filters[idx_tree].position
                        ),
                        orientation=self.robot.pose.transform_orientation_to_relative(
                            self.orientation_filters[idx_tree].rotation
                        ),
                    )
                else:
                    filtered_poses[idx_tree] = ObjectPose(
                        position=self.position_filters[idx_tree].position,
                        orientation=self.orientation_filters[idx_tree].rotation,
                    )

        # # Transform to robot frame(?)
        # filtered_poses = [None] * self.n_components
        # if transform_to_robot_frame and self.robot is not None:
        #     for ii in range(self.n_components):
        #         if self.position_filters[ii] is None:
        #             continue
        #         print(f"trafo for {ii}")
        #         filtered_poses[ii] = ObjectPose(
        #             position=self.robot.pose.transform_position_to_relative(
        #                 self.position_filters[ii].position
        #             ),
        #             orientation=self.robot.pose.transform_orientation_to_relative(
        #                 self.orientation_filters[ii].rotation
        #             ),
        #         )
        # else:
        #     for ii in range(self.n_components):
        #         if self.position_filters[ii] is None:
        #             continue
        #         filtered_poses[ii] = ObjectPose(
        #             position=self.position_filters[ii].position,
        #             orientation=self.orientation_filters[ii].rotation,
        #         )
        if filtered_poses[self.root_idx] is not None:
            # print("Doing root")
            # self.update_dynamic_obstacle(self.root_idx, filtered_poses[self.root_idx])
            self[self.root_idx].pose = filtered_poses[self.root_idx]
            # breakpoint()

            # Assumption of rotation only in z (since it's a human in 2D)
            zyx_rot = self[self.root_idx].pose.orientation.as_euler("zyx")
            self[self.root_idx].pose.orientation = Rotation.from_euler("z", zyx_rot[0])

        obs_indeces = list(self._graph.successors(self.root_idx))
        it_node = 0
        while it_node < len(obs_indeces):
            idx_node = obs_indeces[it_node]
            obs_indeces = obs_indeces + list(self._graph.successors(idx_node))

            it_node += 1  # Iterate

            # idx_optitrack = self._graph.nodes[idx_node]["update_id"]
            # try:
            # idx_measure = indeces_measures.index(idx_optitrack)

            if filtered_poses[idx_node] is None:
                # Static opbstacle - no optitrack exists...
                # We assume orientation was constant?
                idx_parent = list(self._graph.predecessors(idx_node))[0]
                self[idx_node].orientation = self[idx_parent].orientation

                self.align_position_with_parent(idx_node)

            else:
                self[idx_node].pose = filtered_poses[idx_node]
                # self.update_dynamic_obstacle(idx_node, filtered_poses[idx_node])
                self.update_orientation_based_on_position(idx_node)
                self.align_position_with_parent(idx_node)
                # self.align_position_with_parent(idx_node)

                # Reset position filter
                # self.position_filters[idx_node]._position = self[idx_node].pose.position

    # def set_orientation(self, idx_obs: int, orientation: float | Rotation) -> None:
    #     self[idx_obs].orientation = orientation
    #     self.align_position_with_parent(idx_node)
    #     # TODO update all the children, too

    def update_dynamic_obstacle(self, idx_obs: int, obs_measure: RigidBody):
        # Update position
        # self.position_filters[idx_obs].run_once(obs_measure.position)
        # self.orientation_filters[idx_obs].run_once(obs_measure.rotation)

        self[idx_obs].pose.position = self.robot.pose.transform_position_to_relative(
            self.position_filters[idx_obs].position
        )
        self[
            idx_obs
        ].pose.orientation = self.robot.pose.transform_orientation_to_relative(
            self.orientation_filters[idx_obs].rotation
        )
        self[
            idx_obs
        ].linear_velocity = self.robot.pose.transform_linear_velocity_to_relative(
            self.position_filters[idx_obs].velocity
        )

    def update_orientation_based_on_position(
        self, idx_obs: int, position_trust: float = 1.0
    ) -> None:
        idx_parent = list(self._graph.predecessors(idx_obs))[0]
        idx_local_ref = self._graph.nodes[idx_parent]["indeces_children"].index(idx_obs)
        local_reference_parent = self._graph.nodes[idx_parent]["references_children"][
            idx_local_ref
        ]
        reference_parent = self[idx_parent].pose.transform_position_from_relative(
            local_reference_parent
        )

        axes_direction = self[idx_obs].position - reference_parent
        axes_direction = np.array([1.0, 0, -1.0])
        if not (axes_norm := np.linalg.norm(axes_direction)):
            # No information from position only
            return
        axes_direction = axes_direction / axes_norm

        rot_vec = np.cross([0.0, 0, 1.0], axes_direction)

        if rotvec_norm := np.linalg.norm(rot_vec):
            rot_vec = rot_vec / rotvec_norm
            theta = np.arcsin(rotvec_norm)
            quat = np.hstack((rot_vec * np.cos(theta / 2.0), [np.sin(theta / 2.0)]))

        else:
            quat = np.array([0, 0, 0, 1.0])

        self[idx_obs].pose.orientation = Rotation.from_quat(quat)

    def align_position_with_parent(self, idx_obs: int):
        """Update obstacle with respect to the movement of the body-parts (limbs)
        under the assumption of FULL-TRUST(!) to the orientation."""
        idx_parent = list(self._graph.predecessors(idx_obs))[0]
        reference_obstacle = self[idx_obs].pose.transform_position_from_relative(
            self._graph.nodes[idx_obs]["local_reference"]
        )

        idx_local_ref = self._graph.nodes[idx_parent]["indeces_children"].index(idx_obs)

        local_reference_parent = self._graph.nodes[idx_parent]["references_children"][
            idx_local_ref
        ]
        reference_parent = self[idx_parent].pose.transform_position_from_relative(
            local_reference_parent
        )

        delta_ref = reference_parent - reference_obstacle

        # Full believe in orientation (and parent)
        self[idx_obs].pose.position = self[idx_obs].pose.position + delta_ref

    def update(self):
        self.update_using_optitrack()

        if self.robot is not None:
            self.robot.publish_robot_transform()

        if self.visualization_handler is not None:
            self.visualization_handler.update(self._obstacle_list)

    def get_gamma(self, position: Vector, in_global_frame: bool = True) -> bool:
        # in_global_frame is not used but kept for compatibility

        # Get minimum gamma
        gammas = np.zeros(self.n_components)

        for ii in range(self.n_components):
            gammas[ii] = self._obstacle_list[ii].get_gamma(
                position, in_global_frame=True
            )

        return np.min(gammas)


def create_2d_human():
    upper_arm_axes = [0.5, 0.18]
    lower_arm_axes = [0.4, 0.14]
    head_dimension = [0.2, 0.3]

    dimension = 2

    new_human = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
    )

    distance_scaling = 3

    new_human.set_root(
        Cuboid(
            axes_length=[0.4, 0.7],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="body",
    )
    new_human[-1].set_reference_point(np.array([0, -0.3]), in_global_frame=False)

    new_human.add_component(
        Cuboid(
            axes_length=[0.12, 0.12],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="neck",
        parent_name="body",
        reference_position=[0.0, -0.05],
        parent_reference_position=[0.0, 0.30],
    )

    new_human.add_component(
        Ellipse(
            axes_length=[0.2, 0.3],
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="head",
        parent_name="neck",
        reference_position=[0.0, -0.12],
        parent_reference_position=[0.0, 0.05],
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="upperarm1",
        parent_name="body",
        reference_position=[-0.2, 0],
        parent_reference_position=[0.15, 0.3],
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="lowerarm1",
        parent_name="upperarm1",
        reference_position=[-0.18, 0],
        parent_reference_position=[0.2, 0],
    )

    new_human.add_component(
        Ellipse(
            axes_length=upper_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="upperarm2",
        parent_name="body",
        reference_position=[0.2, 0],
        parent_reference_position=[-0.15, 0.3],
    )

    new_human.add_component(
        Ellipse(
            axes_length=lower_arm_axes,
            center_position=np.zeros(dimension),
            distance_scaling=distance_scaling,
        ),
        name="lowerarm2",
        parent_name="upperarm2",
        reference_position=[0.18, 0],
        parent_reference_position=[-0.2, 0],
    )

    new_human.update()

    idx_obs = new_human.get_obstacle_id_from_name("lowerarm1")
    new_human[idx_obs].orientation = 70 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    idx_obs = new_human.get_obstacle_id_from_name("lowerarm2")
    new_human[idx_obs].orientation = 45 * np.pi / 180
    # new_human.set_orientation(idx_obs, orientation=)
    new_human.align_position_with_parent(idx_obs)

    return new_human


def test_2d_human_with_linear(visualize=False):
    # Set arm-orientation
    new_human = create_2d_human()
    multibstacle_avoider = MultiObstacleAvoider(obstacle=new_human)

    # First with (very) simple dyn
    velocity = np.array([1.0, 0.0])
    linearized_velociy = np.array([1.0, 0.0])

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_lim = [-1.3, 1.3]
        y_lim = [-0.25, 1.2]
        n_grid = 20

        plot_obstacles(
            obstacle_container=new_human._obstacle_list,
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
                new_human.get_gamma(x, in_global_frame=True) <= 1
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
    position = np.array([-0.201, -0.2])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )

    # assert (
    #     averaged_direction[1] > 0 and abs(averaged_direction[0]) < 0.1
    # ), "Not tangent to surface in front of obstacle."

    position = np.array([-0.3, -0.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] > 0

    position = np.array([-0.6, 1.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0

    position = np.array([0.4, 0.0])
    averaged_direction = multibstacle_avoider.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[1] < 0


def test_2d_human_with_circular(visualize=False, savefig=False):
    # Set arm-orientation
    new_human = create_2d_human()
    multibstacle_avoider = MultiObstacleAvoider(obstacle=new_human)

    # First with (very) simple dynanmic
    circular_ds = SimpleCircularDynamics(
        radius=1.0,
        pose=ObjectPose(position=np.array([1, 1]), orientation=30.0 / 180 * np.pi),
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - circular_ds.pose.position,
    )

    multibstacle_avoider = MultiObstacleAvoider(
        obstacle=new_human,
        initial_dynamics=circular_ds,
        convergence_dynamics=rotation_projector,
    )

    if visualize:
        fig, ax = plt.subplots(figsize=(7, 4))
        n_grid = 100

        x_lim = [-1.3, 1.3]
        y_lim = [-0.25, 1.2]

        plot_obstacles(
            obstacle_container=new_human._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            noTicks=False,
            # reference_point_number=True,
            # show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            collision_check_functor=lambda x: (
                new_human.get_gamma(x, in_global_frame=True) <= 1
            ),
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=multibstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # do_quiver=True,
            do_quiver=False,
            n_grid=n_grid,
            show_ticks=True,
            # vectorfield_color=vf_color,
            attractor_position=circular_ds.pose.position,
        )

        if savefig:
            figname = "multibody_human"
            plt.savefig(
                "figures/" + "rotated_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    position = np.array([-1.0, 0.0])
    velocity = multibstacle_avoider.evaluate(position)
    assert velocity[1] > 0, "Expected o be moving upwards."

    position = np.array([1.0, 1.0])
    velocity = multibstacle_avoider.evaluate(position)
    assert np.allclose(velocity, [0, 0]), "No velocity at attractor"
    # breakpoint()


def test_2d_blocky_arch(visualize=False):
    dimension = 2

    multi_block = MultiBodyObstacle(
        visualization_handler=None,
        pose_updater=None,
        robot=None,
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

    multi_block.update()

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


if (__name__) == "__main__":
    # figtype = ".png"
    figtype = ".pdf"

    import matplotlib.pyplot as plt

    from dynamic_obstacle_avoidance.visualization import plot_obstacles
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )

    # plt.close("all")
    plt.ion()

    # test_2d_blocky_arch(visualize=True)
    # test_2d_human_with_linear(visualize=True)
    test_2d_human_with_circular(visualize=True, savefig=True)

    print("[INFO] Done.")
