#!/usr/bin/env python3
from typing import Optional, Callable

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

from vartools.states import Pose
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.dynamics import ConstantValue, LinearSystem

# from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.dynamics.sequenced_dynamics import evaluate_dynamics_sequence
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.dynamics import SimpleCircularDynamics
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

from scripts.three_dimensional.visualizer3d import CubeVisualizer

Vector = np.ndarray


def plot_double_plot(
    trajectory,
    ranges,
    obstacle: Cuboid = None,
    attractor_position=None,
    ind_axes=[[0, 2], [1, 2]],
    ax=None,
):
    n_plots = len(ind_axes)
    if ax is None:
        _, axs = plt.subplots(1, n_plots, figsize=(10, 6))

    color = "black"

    for ii, idx in enumerate(ind_axes):
        ax = axs[ii]

        ax.plot(trajectory[idx[0], :], trajectory[idx[1], :])
        ax.plot(trajectory[idx[0], 0], trajectory[idx[1], 0], "x", color=color)
        ax.plot(trajectory[idx[0], -1], trajectory[idx[1], -1], "o", color=color)
        ax.set_xlabel(f"ax={idx[0]}")
        ax.set_ylabel(f"ax={idx[1]}")

        if obstacle is not None:
            tmp_obs = Cuboid(
                axes_length=obstacle.axes_length[idx],
                pose=Pose(obstacle.center_position[idx]),
                margin_absolut=obstacle.margin_absolut,
                distance_scaling=obstacle.distance_scaling,
            )

            plot_obstacles(
                ax=ax,
                x_lim=ranges[idx[0]],
                y_lim=ranges[idx[1]],
                obstacle_container=[tmp_obs],
            )

        if attractor_position is not None:
            ax.plot(
                attractor_position[idx[0]], attractor_position[idx[1]], "*", color=color
            )

        ax.grid()


def create_circular_conveyer_dynamics():
    center_pose = Pose(
        np.array([0.4, 0.0, 0.25]),
        orientation=Rotation.from_euler("x", 0.0),
    )
    return SimpleCircularDynamics(pose=center_pose, radius=0.1)


def create_conveyer_obstacles(margin_absolut=0.1, distance_scaling=10.0):
    print(f"Create environment with")
    print(f"margin={margin_absolut}")
    print(f"scaling={distance_scaling}")

    container = MultiObstacleContainer()

    # Conveyer Belt [static]
    conveyer_belt = MultiObstacle(Pose.create_trivial(dimension=3))
    conveyer_belt.set_root(
        Cuboid(
            center_position=np.array([0.5, 0.0, -0.25]),
            axes_length=np.array([0.5, 2.5, 0.8]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(conveyer_belt)

    box1 = MultiObstacle(Pose(np.array([0.5, 0.2, 0.22])))
    box1.set_root(
        Cuboid(
            center_position=np.array([0.0, 0, 0.0]),
            axes_length=np.array([0.16, 0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    box1[-1].set_reference_point(np.array([0.0, 0.0, -0.08]), in_global_frame=False)
    container.append(box1)

    box2 = MultiObstacle(
        Pose(np.array([0.5, -0.4, 0.38]), orientation=Rotation.from_euler("z", 0.2))
    )
    box2.set_root(
        Cuboid(
            center_position=np.array([0.0, 0, -0.12]),
            axes_length=np.array([0.26, 0.37, 0.24]),
            # axes_length=np.array([0.16, 0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    box2[-1].set_reference_point(np.array([0.0, 0.0, -0.12]), in_global_frame=False)
    container.append(box2)

    for obs in container:
        obs.update_pose(obs.pose)

    # container.visualization_handler = RvizHandler(container)
    return container


def integrate_trajectory(
    start: Vector,
    it_max: int,
    step_size: float,
    velocity_functor: Callable[[Vector], Vector],
) -> np.ndarray:
    positions = np.zeros((start.shape[0], it_max + 1))
    positions[:, 0] = start
    for ii in range(it_max):
        velocity = velocity_functor(positions[:, ii])
        positions[:, ii + 1] = positions[:, ii] + velocity * step_size

    return positions


class Visualization3D:
    dimension = 3
    obstacle_color = hex_to_rgba_float("724545ff")

    figsize = (800, 600)

    def __init__(self):
        self.engine = Engine()
        self.engine.start()
        # self.scene = self.engine.new_scene(
        #     size=self.figsize,
        #     bgcolor=(1, 1, 1),
        #     fgcolor=(0.5, 0.5, 0.5),
        # )
        # self.scene.scene.disable_render = False  # for speed
        # self.scene.background = (255, 255, 255)
        # self.scene.background = (1, 1, 1)

        # self.obstacle_color = np.array(self.obstacle_color)
        # self.obstacle_color[-1] = 0.5
        self.scene = mlab.figure(
            size=self.figsize, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5)
        )

    def plot_multi_obstacles(self, container):
        for tree in container:
            self.plot_obstacles(tree)

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            if isinstance(obs, Ellipse):
                source = ParametricSurface()
                source.function = "ellipsoid"
                self.engine.add_source(source)
                surface = Surface()
                source.add_module(surface)

                actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
                # actor.property.ambient = 1
                actor.property.opacity = self.obstacle_color[-1]
                actor.property.color = tuple(self.obstacle_color[:3])

                actor.mapper.scalar_visibility = False
                actor.property.backface_culling = True
                actor.property.specular = 0.1

                if obs.pose.orientation is not None:
                    orientation = obs.pose.orientation.as_euler("xyz")
                    orientation = orientation.reshape(3) * 180 / np.pi
                    actor.actor.orientation = orientation

                actor.actor.origin = np.zeros(self.dimension)
                actor.actor.position = obs.center_position
                actor.actor.scale = obs.axes_length * 0.5
                actor.enable_texture = True

            if isinstance(obs, Cuboid):
                visualizer = CubeVisualizer(obs)
                visualizer.draw_cube()


def test_linear_avoidance_cube(visualize=False, n_grid=5):
    distance_scaling = 10
    margin_absolut = 0.1

    container = create_conveyer_obstacles(
        margin_absolut=margin_absolut, distance_scaling=distance_scaling
    )
    dynamics = ConstantValue(np.array([0, 1, 0]))

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        x_range = [0.4, 0.6]
        y_value = -0.3
        z_range = [0.1, 0.4]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]
            it_max = 120
            step_size = 0.02
            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, ii]
            for ii in range(it_max):
                velocity = avoider.evaluate_sequence(positions[:, ii])
                np.set_printoptions(precision=17)

                if np.linalg.norm(velocity) < 1e-3:
                    breakpoint()
                positions[:, ii + 1] = positions[:, ii] + velocity * step_size

            trajectory = positions
            mlab.plot3d(
                trajectory[0, :],
                trajectory[1, :],
                trajectory[2, :],
                color=color,
                tube_radius=0.01,
            )

    position = np.array([0, 0, 0])
    velocity = avoider.evaluate_sequence(position)
    assert velocity[0] > 0 and velocity[2] > 0, "Avoid by going above."


def test_straight_avoidance_cube(visualize=False, n_grid=2):
    distance_scaling = 10
    margin_absolut = 0.1

    container = MultiObstacleContainer()
    box = MultiObstacle(
        Pose(
            np.array([0.5, 0.2, 0.38]),
        )
    )
    box.set_root(
        Cuboid(
            center_position=np.array([0.0, 0, -0.12]),
            axes_length=np.array([0.16, 0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    box[-1].set_reference_point(np.array([0.0, 0.0, -0.12]), in_global_frame=False)
    container.append(box)
    for obs in container:
        obs.update_pose(obs.pose)

    dynamics = LinearSystem(
        attractor_position=np.array([0.4, 0.0, 0.25]),
        maximum_velocity=1.0,
    )
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        it_max = 150
        step_size = 0.02

        x_range = [0.3, 0.7]
        y_value = 0.4
        z_range = [-0.05, 0.35]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )

        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        n_traj = start_positions.shape[1]
        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        attractor = dynamics.attractor_position
        mlab.points3d(attractor[0], attractor[1], attractor[2], scale_factor=0.03)

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]

            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, ii]

            for ii in range(it_max):
                velocity = avoider.evaluate_sequence(positions[:, ii])
                positions[:, ii + 1] = positions[:, ii] + velocity * step_size

            trajectory = positions
            mlab.plot3d(
                trajectory[0, :],
                trajectory[1, :],
                trajectory[2, :],
                color=color,
                tube_radius=0.01,
            )

        plot_double_plot(
            trajectory,
            ranges=[x_range, [0, 1], z_range],
            obstacle=container.get_obstacle_tree(0).get_component(0),
            attractor_position=dynamics.attractor_position,
        )

    # No singularity
    position = np.array([0, 1, 0.0])
    velocity = avoider.evaluate_sequence(position)
    assert np.any(np.isnan(velocity))


def test_circular_avoidance_cube(visualize=False, n_grid=3):
    distance_scaling = 5
    margin_absolut = 0.0

    dynamics = SimpleCircularDynamics(
        pose=Pose(
            np.array([0.0, 0.0, 0.0]),
            orientation=Rotation.from_euler("x", 0.0),
        ),
        radius=1.0,
    )

    container = MultiObstacleContainer()
    box = MultiObstacle(
        Pose(
            np.array([0.0, 0.0, 0.0]),
        )
    )
    box.set_root(
        Cuboid(
            center_position=np.array([-2, 0, 0.0]),
            axes_length=np.array([1.0, 1.0, 1.0]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    container.append(box)

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        it_max = 150
        step_size = 0.08

        x_value = -4
        y_range = [-1.0, 1.5]
        z_range = [-0.4, 0.4]

        xv = x_value * np.ones(n_grid * n_grid)
        yv, zv = np.meshgrid(
            np.linspace(y_range[0], y_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]

            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, ii]

            for ii in range(it_max):
                velocity = avoider.evaluate_sequence(positions[:, ii])
                positions[:, ii + 1] = positions[:, ii] + velocity * step_size

            trajectory = positions
            mlab.plot3d(
                trajectory[0, :],
                trajectory[1, :],
                trajectory[2, :],
                color=color,
                tube_radius=0.01,
            )

    position = np.array([-4, -1.0, 0])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([-4, -1.5, 0])
    velocity2 = avoider.evaluate_sequence(position)
    assert velocity1[1] < velocity1[2] < 0, "Splitting the dynamics."


def test_linear_avoidance_sphere(visualize=True, n_grid=3):
    distance_scaling = 5.0
    margin_absolut = 0.0

    container = MultiObstacleContainer()

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        it_max = 120
        step_size = 0.05

        x_range = [-0.4, 0.4]
        y_value = -4.0
        z_range = [-0.4, 0.4]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]

            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, ii]

            for ii in range(it_max):
                velocity = avoider.evaluate_sequence(positions[:, ii])
                positions[:, ii + 1] = positions[:, ii] + velocity * step_size

            trajectory = positions
            mlab.plot3d(
                trajectory[0, :],
                trajectory[1, :],
                trajectory[2, :],
                color=color,
                tube_radius=0.05,
            )

        attractor = dynamics.attractor_position
        mlab.points3d(attractor[0], attractor[1], attractor[2], scale_factor=0.1)

    position = np.array([0.2, -5, 0.2])
    velocity = avoider.evaluate_sequence(position)
    assert (
        velocity[0] > 0 and velocity[1] > 0 and velocity[2] > 0
    ), "Avoiding towards attractor"


def test_conveyer_setup(visualize=False, n_grid=2):
    distance_scaling = 20.0
    margin_absolut = 0.1

    dynamics = create_circular_conveyer_dynamics()
    container = create_conveyer_obstacles(
        distance_scaling=distance_scaling, margin_absolut=margin_absolut
    )
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        reference_dynamics=dynamics,
    )

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        it_max = 120
        step_size = 0.01

        x_range = [-0.0, 0.8]
        y_range = [-0.4, 0.4]
        z_value = 0.4

        zv = z_value * np.ones(n_grid * n_grid)
        xv, yv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(y_range[0], y_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        # start_positions = np.array([start_positions[:, 1]]).T

        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for jj, position in enumerate(start_positions.T):
            color = color_list[jj][:3]

            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, jj]

            # For debugging only
            velocity_old = avoider.evaluate_sequence(positions[:, jj])
            for ii in range(it_max):
                velocity = avoider.evaluate_sequence(positions[:, ii])
                # if velocity[2] > 0:
                #     positions[:, ii]
                #     breakpoint()

                positions[:, ii + 1] = positions[:, ii] + velocity * step_size

                # if not np.allclose(velocity, velocity_old, atol=0.6):
                #     breakpoint()
                if np.allclose(
                    positions[:, ii + 1],
                    np.array([0.498195, 0.010943, 0.3428145]),
                    atol=1e-2,
                ):
                    print(positions[:, ii + 1])

                velocity_old = velocity
            trajectory = positions
            mlab.plot3d(
                trajectory[0, :],
                trajectory[1, :],
                trajectory[2, :],
                color=color,
                tube_radius=0.01,
            )

            plot_double_plot(positions, ranges=[x_range, y_range, [0, 0.8]])

        attractor = dynamics.attractor_position
        mlab.points3d(attractor[0], attractor[1], attractor[2], scale_factor=0.02)

    # print()
    # position = np.array([0.6717153338730932, -0.0920069684374953, 0.3734692658417906])
    # velocity1 = avoider.evaluate_sequence(position)
    # print("velocity1", velocity1)

    # print()
    # position = np.array([0.6777531783374638, -0.0999618734235742, 0.3739829981519747])
    # velocity0 = avoider.evaluate_sequence(position)
    # print("velocity0", velocity0)

    # breakpoint()

    # Two close positoins
    # position = np.array([0.4981972483529464, 0.004332362438969479, 0.3428155262137731])
    # initial_sequence = evaluate_dynamics_sequence(position, avoider.initial_dynamics)
    # convergence1 = avoider.compute_convergence_sequence(position, initial_sequence)
    # velocity1 = avoider.evaluate_sequence(position)

    # position = np.array([0.4941097112959491, 0.01210789275673355, 0.34470485376277726])
    # initial_sequence = evaluate_dynamics_sequence(position, avoider.initial_dynamics)
    # convergence2 = avoider.compute_convergence_sequence(position, initial_sequence)
    # velocity2 = avoider.evaluate_sequence(position)

    # breakpoint()
    # assert np.allclose(
    #     convergence1.get_end_vector(), convergence2.get_end_vector(), atol=0.26
    # )
    # assert np.allclose(velocity1, velocity2, atol=0.25)


if (__name__) == "__main__":
    figtype = ".jpeg"
    # np.set_printoptions(precision=3)
    np.set_printoptions(precision=16)

    # mlab.close(all=True)
    # plt.close("all")

    # test_straight_avoidance_cube(visualize=True)
    # test_linear_avoidance_sphere(visualize=True)
    # test_circular_avoidance_cube(visualize=True)

    test_conveyer_setup(visualize=True)
