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

    optitrack_obstacles = MultiObstacleContainer()

    # Conveyer Belt [static]
    # conveyer_belt = MultiObstacle(Pose.create_trivial(dimension=3))
    # conveyer_belt.set_root(
    #     Cuboid(
    #         center_position=np.array([0.5, 0.0, -0.25]),
    #         axes_length=np.array([0.5, 2.5, 0.8]),
    #         margin_absolut=margin_absolut,
    #         distance_scaling=distance_scaling,
    #     )
    # )
    # optitrack_obstacles.append(conveyer_belt)

    # box1 = MultiObstacle(Pose(np.array([0.5, 0.5, 0.3])))
    # box1.set_root(
    #     Cuboid(
    #         center_position=np.array([0.0, 0, 0.0]),
    #         axes_length=np.array([0.16, 0.16, 0.16]),
    #         margin_absolut=margin_absolut,
    #         distance_scaling=distance_scaling,
    #     )
    # )
    # box1[-1].set_reference_point(np.array([0.0, 0.0, -0.08]), in_global_frame=False)
    # optitrack_obstacles.append(box1)

    box2 = MultiObstacle(
        Pose(
            np.array([0.5, 0.2, 0.38]),
            # orientation=Rotation.from_euler("z", 0.2)
        )
    )
    box2.set_root(
        Cuboid(
            center_position=np.array([0.0, 0, -0.12]),
            # axes_length=np.array([0.26, 0.37, 0.24]),
            axes_length=np.array([0.16, 0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    box2[-1].set_reference_point(np.array([0.0, 0.0, -0.12]), in_global_frame=False)
    optitrack_obstacles.append(box2)

    for obs in optitrack_obstacles:
        obs.update_pose(obs.pose)

    # optitrack_obstacles.visualization_handler = RvizHandler(optitrack_obstacles)
    return optitrack_obstacles


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


def test_straight_avoidance_cube(visualize=False, n_grid=3):
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

    # conatiner = create_conveyer_obstacles(margin_absolut, distance_scaling)

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

        x_range = [0.0, 0.6]
        y_value = 0.4
        # y_value = -0.3
        z_range = [-0.2, 0.4]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        start_positions = np.array([[0.6, 0.4, 0.1]]).T

        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        attractor = dynamics.attractor_position
        mlab.points3d(attractor[0], attractor[1], attractor[2], scale_factor=0.03)

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]
            it_max = 120
            step_size = 0.02
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
                ranges=[[0, 1], [0, 1], [-0.2, 0.8]],
                obstacle=container.get_obstacle_tree(0).get_component(0),
                attractor_position=dynamics.attractor_position,
            )

    position = np.array([0.33845, 0.0564, 0.44093])
    initial = evaluate_dynamics_sequence(position, avoider.initial_dynamics)
    convergence1 = avoider.compute_convergence_sequence(position, initial)
    velocity1 = avoider.evaluate_sequence(position)
    print("convergence1", convergence1.get_end_vector())
    print("velocity1", velocity1)
    print()

    position = np.array([0.33931, 0.05435, 0.43740])
    initial = evaluate_dynamics_sequence(position, avoider.initial_dynamics)
    convergence2 = avoider.compute_convergence_sequence(position, initial)
    velocity2 = avoider.evaluate_sequence(position)
    print("convergence2", convergence2.get_end_vector())
    print("velocity2", velocity2)
    breakpoint()

    position = np.array([0.3347327127745454, 0.012918052763852035, 0.42139429339490314])
    velocity1 = avoider.evaluate_sequence(position)

    position = np.array([0.3335963058382737, 0.013751535953591689, 0.4248657827749872])
    velocity2 = avoider.evaluate_sequence(position)
    position = np.array([0.33245048743770905, 0.014642767379258714, 0.4284016248533758])
    velocity3 = avoider.evaluate_sequence(position)
    breakpoint()

    position = np.array([0.3362342841663879, 0.043198437634528145, 0.25370974868181606])
    velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(velocity)), "No singularity."

    position = np.array([0.4153644675841204, -0.134944819439844, 0.25385367192686137])
    velocity = avoider.evaluate_sequence(position)
    assert not np.any(np.isnan(velocity)), "Singularity detected"


def test_circular_avoidance_cube(visualize=False, n_grid=5):
    distance_scaling = 10
    margin_absolut = 0.1

    container = create_conveyer_obstacles(
        margin_absolut=margin_absolut, distance_scaling=distance_scaling
    )
    # dynamics = create_circular_conveyer_dynamics()
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

        x_range = [0.4, 0.6]
        y_value = -0.3
        z_range = [0.1, 0.4]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        # start_positions = np.array([[0.5, -0.2, 0.4]]).T
        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]
            it_max = 120
            # it_max = 2
            step_size = 0.02
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
    breakpoint()

    position = np.array([0.508838938173121, 0.0132225929006104, 0.34378821844303004])
    velocity2 = avoider.evaluate_sequence(position)
    initial2 = evaluate_dynamics_sequence(position, avoider.initial_dynamics)
    convergence_sequence = avoider.compute_convergence_sequence(
        position, initial_sequence
    )
    # covnergence2 = avoider.(position2)
    print(velocity2)
    breakpoint()

    # assert velocity2[2] < 0, "Should be going down(!)"

    position1 = np.array([0.4901681004606718, 0.01320262067375966, 0.33676500195122705])
    velocity1 = avoider.evaluate_sequence(position1)
    print(velocity1)

    breakpoint()
    # assert np.allclose(position1, position2, atol=1e-1), "Sanity check of input."
    np.allclose(velocity1, velocity2, atol=1e-1)

    position3 = np.array([0.4901179016891956, 0.01322259290061041, 0.33675097516405117])
    velocity3 = avoider.evaluate_sequence(position3)
    print(velocity3)

    position4 = np.array([0.5087878780934926, 0.01324247218810984, 0.3437762320797586])
    velocity4 = avoider.evaluate_sequence(position4)
    print(velocity4)

    position5 = np.array([0.4900675921896201, 0.01324247218810985, 0.3367369923710065])
    velocity5 = avoider.evaluate_sequence(position5)
    print(velocity5)
    breakpoint()


if (__name__) == "__main__":
    figtype = ".jpeg"
    np.set_printoptions(precision=18)
    mlab.close(all=True)
    plt.close("all")

    # test_circular_avoidance_cube(visualize=True)
    test_straight_avoidance_cube(visualize=True)
    # test_circualar_avoidance_cube(visualize=False, n_grid=1)
    # test_visualize_2d(visualize=True)
