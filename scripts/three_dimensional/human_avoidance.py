from __future__ import annotations  # To be removed in future python versions
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

from vartools.states import Pose
from vartools.dynamics import ConstantValue
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.multi_body_human import create_3d_human
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics

Vector = np.ndarray


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

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            if isinstance(obs, Ellipse):
                # plot_ellipse_3d(
                #     scene=scene, center=obs.center_position, axes_length=obs.axes_length
                # )

                source = ParametricSurface()
                source.function = "ellipsoid"
                self.engine.add_source(source)
                surface = Surface()
                source.add_module(surface)

                actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
                # defaults to 0 for some reason, ah don't need it, turn off scalar visibility instead
                # actor.property.ambient = 1
                actor.property.opacity = self.obstacle_color[-1]
                actor.property.color = tuple(self.obstacle_color[:3])

                # Colour ellipses by their scalar indices into colour map
                actor.mapper.scalar_visibility = False

                # gets rid of weird rendering artifact when opacity is < 1
                actor.property.backface_culling = True
                actor.property.specular = 0.1

                # actor.property.frontface_culling = True
                if obs.pose.orientation is not None:
                    orientation = obs.pose.orientation.as_euler("xyz")
                    orientation = orientation.reshape(3) * 180 / np.pi
                    # ind_negativ = orientation < 0
                    # if np.sum(ind_negativ):
                    #     orientation[ind_negativ] = 360 - orientation[ind_negativ]

                    actor.actor.orientation = orientation
                    # breakpoint()

                # actor.actor.origin = obs.center_position
                # actor.actor.position = np.zeros(self.dimension)
                actor.actor.origin = np.zeros(self.dimension)
                actor.actor.position = obs.center_position
                actor.actor.scale = obs.axes_length * 0.5
                actor.enable_texture = True

            if isinstance(obs, Cuboid):
                visualizer = CubeVisualizer(obs)
                visualizer.draw_cube()


def plot_reference_points(obstacles):
    obstacle_color = hex_to_rgba_float("724545ff")

    for ii, obs in enumerate(obstacles):
        point = obs.get_reference_point(in_global_frame=True)
        mlab.points3d(
            point[0], point[1], point[2], scale_factor=0.1, color=obstacle_color[:3]
        )


@dataclass
class CubeVisualizer:
    n_grid = 2
    obstacle: Obstacle
    faces: list = field(default_factory=lambda: [])

    obstacle_color: np.ndarray = hex_to_rgba("724545ff")

    def __post_init__(self):
        self.faces = self.compute_cube_faces(self.obstacle.axes_length)

        # self.obstacle_color = np.array(self.obstacle_color)
        # self.obstacle_color[-1] = 120

    def compute_cube_faces(self, axes_length):
        xmin, ymin, zmin = -axes_length * 0.5
        xmax, ymax, zmax = axes_length * 0.5

        nn = self.n_grid * 1j

        faces = []
        x, y = np.mgrid[xmin:xmax:nn, ymin:ymax:nn]
        z = np.ones(y.shape) * zmin
        faces.append((x, y, z))

        x, y = np.mgrid[xmin:xmax:nn, ymin:ymax:nn]
        z = np.ones(y.shape) * zmax
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:nn, zmin:zmax:nn]
        y = np.ones(z.shape) * ymin
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:nn, zmin:zmax:nn]
        y = np.ones(z.shape) * ymax
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:nn, zmin:zmax:nn]
        x = np.ones(z.shape) * xmin
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:nn, zmin:zmax:nn]
        x = np.ones(z.shape) * xmax
        faces.append((x, y, z))

        return faces

    def transform_faces_from_relative(self, pose: Pose) -> list:
        global_faces: list = []
        for grid in self.faces:
            xx, yy, zz = grid
            pos = pose.transform_positions_from_relative(
                np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
            )
            global_faces.append(
                (
                    pos[0, :].reshape(self.n_grid, self.n_grid),
                    pos[1, :].reshape(self.n_grid, self.n_grid),
                    pos[2, :].reshape(self.n_grid, self.n_grid),
                )
            )

        return global_faces

    def draw_cube(self) -> None:
        global_faces = self.transform_faces_from_relative(self.obstacle.pose)

        for grid in global_faces:
            x, y, z = grid
            mesh = mlab.mesh(x, y, z, opacity=1.0)
            # The lut is a 255x4 array, with the columns representing RGBA
            # (red, green, blue, alpha) coded with integers going from 0 to 255.
            mesh.module_manager.scalar_lut_manager.lut.number_of_colors = 2
            mesh.module_manager.scalar_lut_manager.lut.table = np.tile(
                self.obstacle_color, (2, 1)
            )


def get_perpendicular_vector(initial: Vector, nominal: Vector) -> Vector:
    perp_vector = initial - (initial @ nominal) * nominal

    if not (perp_norm := np.linalg.norm(perp_vector)):
        return np.zeros_like(initial)

    return perp_vector / perp_norm


@dataclass
class SpiralingDynamics3D:
    """Dynamics consisting of 2D rotating and linear direction.

    The dynamics are spiraling around the center in the y-z plane, while mainting"""

    pose: Pose
    direction: Vector
    circular_dynamics: SimpleCircularDynamics = SimpleCircularDynamics(
        pose=Pose.create_trivial(2)
    )
    speed: float = 1.0

    @classmethod
    def create_from_direction(
        cls,
        center: Vector,
        direction: Vector,
        radius: float = 1.0,
        speed: float = 1.0,
    ) -> Self:
        direction = direction
        basis = get_orthogonal_basis(direction)
        rotation = Rotation.from_matrix(basis)

        circular = SimpleCircularDynamics(pose=Pose.create_trivial(2), radius=radius)

        # rotation = Rotation.from_matrix(basis.T)
        return cls(Pose(center, rotation), direction, circular, speed)

    def evaluate(self, position: Vector) -> Vector:
        local_position = self.pose.transform_position_to_relative(position)
        rotating_vel2d = self.circular_dynamics.evaluate(local_position[1:])

        rotating_velocity = np.hstack((0.0, rotating_vel2d))
        rotating_velocity = self.pose.transform_position_from_relative(
            rotating_velocity
        )

        combined_velocity = rotating_velocity + self.direction
        combined_velocity = combined_velocity / np.linalg.norm(combined_velocity)
        combined_velocity = combined_velocity * self.speed
        return combined_velocity

    def evaluate_convergence_around_obstacle(
        self, position: Vector, obstacle: Obstacle
    ) -> Vector:
        return self.direction


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


def integrate_trajectory_with_differences(
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


def plot_axes(lensoffset=0.0):
    xx = yy = zz = np.arange(-1.0, 1.0, 0.1)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
    mlab.plot3d(yx, yy + lensoffset, yz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(zx, zy + lensoffset, zz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(xx, xy + lensoffset, xz, line_width=0.01, tube_radius=0.01)


def set_view():
    mlab.view(
        -150.15633889829527,
        68.76031172885509,
        4.135728793575641,
        (-0.16062227, -0.1689306, -0.00697224)
        # distance=5.004231840226419,
        # focalpoint=(-0.32913308, 0.38534346, -0.14484502),
    )
    mlab.background = (255, 255, 255)


def main(savefig=False, n_grid=6):
    human_obstacle = create_3d_human()

    visualizer = Visualization3D()
    visualizer.plot_obstacles(human_obstacle)

    # plot_multi_obstacle_3d(, obstacle=human_obstacle)
    # plot_reference_points(human_obstacle._obstacle_list)
    # plot_axes()

    nominal = np.array([0.0, 1, 0.0])
    convergence_dynamics = ConstantValue(nominal)

    dynamics = SpiralingDynamics3D.create_from_direction(
        center=np.array([-0.2, 0, 0.0]),
        direction=nominal,
        radius=0.1,
        speed=1.0,
    )

    avoider = MultiObstacleAvoider(
        obstacle=human_obstacle,
        initial_dynamics=dynamics,
        convergence_dynamics=dynamics,
    )

    x_range = [-0.9, 0.9]
    y_value = -2
    z_range = [-0.2, 0.9]

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
        trajecotry = integrate_trajectory(
            np.array(position),
            it_max=120,
            step_size=0.05,
            velocity_functor=avoider.evaluate,
        )
        # trajecotry = integrate_trajectory(
        #     np.array(position),
        #     it_max=100,
        #     step_size=0.1,
        #     velocity_functor=dynamics.evaluate,
        # )
        mlab.plot3d(
            trajecotry[0, :],
            trajecotry[1, :],
            trajecotry[2, :],
            color=color,
            tube_radius=0.01,
        )

    set_view()
    if savefig:
        mlab.savefig(
            str(Path("figures") / ("human_avoidance_3d_avoidance" + figtype)),
            magnification=2,
        )

    # Initial dynamics
    visualizer = Visualization3D()

    for ii, position in enumerate(start_positions.T):
        color = color_list[ii][:3]
        trajecotry = integrate_trajectory(
            np.array(position),
            it_max=120,
            step_size=0.05,
            velocity_functor=dynamics.evaluate,
        )
        # trajecotry = integrate_trajectory(
        #     np.array(position),
        #     it_max=100,
        #     step_size=0.1,
        #     velocity_functor=dynamics.evaluate,
        # )
        mlab.plot3d(
            trajecotry[0, :],
            trajecotry[1, :],
            trajecotry[2, :],
            color=color,
            tube_radius=0.01,
        )

    set_view()
    if savefig:
        mlab.savefig(
            str(Path("figures") / ("human_avoidance_3d_initial" + figtype)),
            magnification=2,
        )


if (__name__) == "__main__":
    figtype = ".jpeg"
    mlab.close(all=True)
    main(savefig=True, n_grid=4)
