from __future__ import annotations  # To be removed in future python versions
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

from vartools.states import Pose
from vartools.colors import hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.multi_body_human import create_3d_human

Vector = np.ndarray


class Visualization3D:
    obstacle_color = hex_to_rgba_float("724545ff")

    def __init__(self):
        self.engine = Engine()
        self.engine.start()
        self.scene = self.engine.new_scene(
            bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600)
        )
        self.scene.scene.disable_render = True  # for speed

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

                actor.property.opacity = 0.7

                # tuple(np.random.rand(3))
                # breakpoint()
                actor.property.color = self.obstacle_color[:3]

                # Colour ellipses by their scalar indices into colour map
                actor.mapper.scalar_visibility = False
                # breakpoint()

                # gets rid of weird rendering artifact when opacity is < 1
                actor.property.backface_culling = True
                actor.property.specular = 0.1

                # actor.property.frontface_culling = True
                if obs.pose.orientation is not None:
                    actor.actor.orientation = obs.pose.orientation.as_euler("xyz")
                actor.actor.origin = obs.center_position
                actor.actor.position = obs.center_position
                actor.actor.scale = obs.axes_length
                actor.enable_texture = True

            # if isinstance(obs, Cuboid):
            #     plot_ellipse_3d(
            #         scene=scene, center=obs.center_position, axes_length=obs.axes_length
            #     )


def cube_faces(center_position, axes_length):
    xmin, ymin, zmin = center_position - axes_length
    faces = []

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmin
    faces.append((x, y, z))

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmax
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymin
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymax
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmin
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmax
    faces.append((x, y, z))

    return faces


def draw_cube(center_position, axes_length):
    faces = cube_faces(center_position, axes_length)
    for grid in faces:
        x, y, z = grid
        mlab.mesh(x, y, z, opacity=0.4)


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

    @classmethod
    def create_from_direction(cls, center: Vector, direction: Vector) -> Self:
        direction = direction
        basis = get_orthogonal_basis(direction)
        rotation = Rotation.from_matrix(basis)
        # rotation = Rotation.from_matrix(basis.T)
        return cls(Pose(center, rotation), direction)

    def evaluate(self, position: Vector) -> Vector:
        local_position = self.pose.transform_position_to_relative(position)
        rotating_vel2d = self.circular_dynamics.evaluate(local_position[1:])

        rotating_velocity = np.hstack((0.0, rotating_vel2d))
        rotating_velocity = self.pose.transform_position_from_relative(
            rotating_velocity
        )

        combined_velocity = rotating_velocity + self.direction
        return combined_velocity


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


def main():
    visualizer = Visualization3D()

    human_obstacle = create_3d_human()

    visualizer.plot_obstacles(human_obstacle)

    breakpoint()
    # fig = mlab.figure(size=(800, 600))
    # plot_multi_obstacle_3d(, obstacle=human_obstacle)

    dynamics = SpiralingDynamics3D.create_from_direction(
        np.array([0, 0, 0.0]),
        np.array([1, 0, 0.0]),
    )
    start_positions = [
        [-2.0, -1, 0.0],
        [-1.0, -3, 0],
        [-3.0, -1, 0],
        [-5.0, -2, 0],
        [-3.0, -2, 0],
    ]
    for ii, position in enumerate(start_positions):
        trajecotry = integrate_trajectory(
            np.array(position),
            it_max=100,
            step_size=0.1,
            velocity_functor=dynamics.evaluate,
        )
        mlab.plot3d(trajecotry[0, :], trajecotry[1, :], trajecotry[2, :])

    breakpoint()


if (__name__) == "__main__":
    main()
