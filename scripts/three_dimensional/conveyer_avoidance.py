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

# from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_container import MultiObstacleContainer
from nonlinear_avoidance.dynamics import SimpleCircularDynamics

from scripts.three_dimensional.visualizer3d import CubeVisualizer

Vector = np.ndarray


def create_circular_conveyer_dynamics():
    center_pose = Pose(
        np.array([0.5, 0.0, 0.3]),
        orientation=Rotation.from_euler("x", 0.0),
    )
    return SimpleCircularDynamics(pose=center_pose, radius=0.1)


def create_conveyer_obstacles(margin_absolut=0.1, distance_scaling=10.0):
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

    box1 = MultiObstacle(Pose(np.array([0.5, -0.2, 0.30])))
    box1.set_root(
        Cuboid(
            center_position=np.array([0.0, 0, -0.06]),
            axes_length=np.array([0.16, 0.16, 0.16]),
            margin_absolut=margin_absolut,
            distance_scaling=distance_scaling,
        )
    )
    # box1[-1].set_reference_point(np.array([0.0, 0.0, -0.08]), in_global_frame=False)
    optitrack_obstacles.append(box1)

    # box2 = MultiObstacle(Pose(np.array([0.5, 0.8, 0.35])))
    # box2.set_root(
    #     Cuboid(
    #         center_position=np.array([0.0, 0, -0.12]),
    #         axes_length=np.array([0.26, 0.37, 0.24]),
    #         margin_absolut=margin_absolut,
    #         distance_scaling=distance_scaling,
    #     )
    # )
    # box2[-1].set_reference_point(np.array([0.0, 0.0, -0.12]), in_global_frame=False)
    # optitrack_obstacles.append(box2)

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


def test_circualar_avoidance_cube(visualize=False, n_grid=1):
    container = create_conveyer_obstacles(margin_absolut=0, distance_scaling=1)
    dynamics = create_circular_conveyer_dynamics()

    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True,
    )

    if visualize:
        visualizer = Visualization3D()
        visualizer.plot_multi_obstacles(container)

        x_range = [-0.9, 0.9]
        y_value = -2
        z_range = [-0.2, 0.9]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        # start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        start_positions = np.array([[0.5, 0.1, 0.4]]).T
        n_traj = start_positions.shape[1]

        cm = plt.get_cmap("gist_rainbow")
        color_list = [cm(1.0 * cc / n_traj) for cc in range(n_traj)]

        for ii, position in enumerate(start_positions.T):
            color = color_list[ii][:3]
            # trajectory = integrate_trajectory(
            #     np.array(position),
            #     it_max=120,
            #     step_size=0.02,
            #     velocity_functor=avoider.evaluate,
            # )
            # positions = np.zeros((start.shape[0], it_max + 1))
            it_max = 120
            step_size = 0.02
            positions = np.zeros((start_positions.shape[0], it_max + 1))
            positions[:, 0] = start_positions[:, 0]
            for ii in range(it_max):
                velocity = avoider.evaluate(positions[:, ii])

                # print("vel-mag", np.linalg.norm(velocity))
                print("pos", positions[:, ii])
                print("vel", velocity)

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

    # Similar position - similar velocity?
    position1 = [0.49545138, 0.05724965, 0.3041467]
    velocity1 = avoider.evaluate(np.array(position1))
    print(velocity1)

    print()
    print("pos 2")
    position2 = [0.49103304, 0.05732273, 0.32365242]
    velocity2 = avoider.evaluate(np.array(position2))
    print(velocity2)

    breakpoint()
    assert np.allclose(velocity1, velocity2, atol=0.1)


if (__name__) == "__main__":
    figtype = ".jpeg"
    mlab.close(all=True)

    # test_circualar_avoidance_cube(visualize=True, n_grid=1)
    test_circualar_avoidance_cube(visualize=False, n_grid=1)
