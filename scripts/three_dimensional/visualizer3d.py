from dataclasses import dataclass, field

import numpy as np

from mayavi import mlab

from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.states import Pose

from dynamic_obstacle_avoidance.obstacles import Obstacle


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
