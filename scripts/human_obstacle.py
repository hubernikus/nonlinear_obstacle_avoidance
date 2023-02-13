import numpy as np
from numpy import linalg

from dataclasses import dataclass, field

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from scipy.spatial.transform import Rotation


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


@dataclass
class Limb:
    name: int
    parent_ind: int
    # (!) Reference position is in frame of the parent
    _reference_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    children_ind: list[int] = field(default_factory=list)
    references_children: list[np.ndarray] = field(default_factory=list)


class HumanMultiLimb:
    dimension = 3

    def __init__(self):
        self.body = Cuboid(
            center_position=np.zeros(self.dimension),
            axes_length=np.array([0.2, 0.4, 0.5]),
        )
        self.limb_names = ["body"]

        self.limbs_descriptions: list[Limb] = []

        self.shoulder_width = 0.4
        self.upperarm_length = 0.35
        self.forearm_length = 0.3

        self.direction_uperarm0_ = np.ones(self.dimension) / (1.0 * self.dimension)
        self.direction_uperarm1_ = np.ones(self.dimension) / (1.0 * self.dimension)

        self.direction_forearm0_ = np.ones(self.dimension) / (1.0 * self.dimension)
        self.direction_forearm1_ = np.ones(self.dimension) / (1.0 * self.dimension)

        self.upperarm0 = Ellipse(
            center_position=np.array([1, 2, 1]),
            axes_length=np.array([0.2, 0.4, 0.5]),
        )

        self.upperarm1 = Ellipse(
            center_position=np.zeros(self.dimension),
            axes_length=np.array([0.2, 0.4, 0.5]),
        )

    def add_limb(
        self, obstacle, name, reference_position, parent_name, parent_reference_position
    ):
        parent_ind = [ll.name for ll in self.limb_descriptions].index(parent_name)
        if not len(parent_id):
            breakpoint()

        # Add the element to the limb-list
        self.limbs_descriptions[parent_ind].children_ind.append(
            len(self.limbs_descriptions)
        )
        self.limbs_descriptions[parent_ind].children_ind.append(
            parent_reference_position
        )

        self.limbs_descriptions.append(
            name,
            parent_ind,
            reference_position,
        )

    def update_obstacles(self):
        pass

    def update_direction(self):
        pass


def plot_human_obstalce():
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    human_with_limbs = HumanMultiLimb()
    plot_3d_cuboid(ax, human_with_limbs.body)
    # plot_3d_ellipsoid(ax, human_with_limbs.body)
    plot_3d_ellipsoid(ax, human_with_limbs.upperarm0)
    ax.axis("equal")


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()

    plot_human_obstalce()
