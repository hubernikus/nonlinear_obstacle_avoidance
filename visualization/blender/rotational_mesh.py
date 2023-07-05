import bpy

import math
from typing import Optional
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from vartools.math import get_intersection_with_circle
from nonlinear_avoidance.vector_rotation import directional_vector_addition

from blender_math import get_quat_from_direction, deg_to_euler
from materials import create_color, hex_to_rgba

Vertice = tuple[float]


def is_even(value: int) -> bool:
    return not (value % 2)


def is_odd(value: int) -> bool:
    return bool(value % 2)


def get_circle_point(
    point,
    start_point,
    radius: float = math.pi / 2.0,
    weight: float = 1.0,
    center=[1, 0, 0],
    scene=None,
):
    if scene is not None:
        radius = scene.half_circle.radius
        center = scene.normal_point.location

    if hasattr(point, "location"):
        point = np.array(point.location)

    if hasattr(start_point, "location"):
        start_point = np.array(start_point.location)

    intersection = get_intersection_with_circle(
        point - np.array(center),
        direction=point - start_point,
        radius=radius,
    )

    return weight * (intersection + center) + (1 - weight) * point


class RotationalMesh:
    def __init__(self, n_lattitude, n_longitude, it_max: int = 10) -> None:
        self.radius = 1.0

        self.it_max = it_max
        if is_odd(n_lattitude):
            raise ValueError("Even number required.")

        self.n_lattitude = n_lattitude
        self.n_longitude = n_longitude

        self.dimension = 3

        self.vertices = [
            (0, 0.0, 0.0) for _ in range(self.n_lattitude * self.n_longitude + 1)
        ]
        self.edges = []
        self.faces = []

        self.vector_init = [
            (0, 0.0, 0.0) for _ in range(self.n_lattitude * self.n_longitude + 1)
        ]
        self.vector_final = np.zeros(
            (self.dimension, self.n_lattitude * self.n_longitude + 1)
        )

        self.create_vertices()
        self.create_edges_and_faces()
        self.evaluate_vector_final()

        # self.update_vertices(ii=0)

        self.mesh = bpy.data.meshes.new("circle_space_mesh")
        self.mesh.from_pydata(self.vertices, self.edges, self.faces)
        self.mesh.update()

        # self.create_time_series()

        # Make object from mesh
        self.object = bpy.data.objects.new("circle_space_object", self.mesh)
        polygons = self.object.data.polygons
        polygons.foreach_set("use_smooth", [True] * len(polygons))

        # Make collection
        self.collection = bpy.data.collections.new("circle_space_collection")
        bpy.context.scene.collection.children.link(self.collection)
        self.collection.objects.link(self.object)

        self.set_colors()

        # Create animation
        # self.object.position
        print("Circle space done.")

    def set_colors(self) -> None:
        self.material_names = ["StereographicGreen", "SteorographicRed"]
        create_color(
            color=(75.0 / 255, 102.0 / 255, 20.0 / 255, 1),
            name=self.material_names[0],
            obj=self.object,
        )
        create_color(
            color=(164.0 / 255, 26.0 / 255, 38.0 / 255, 1),
            name=self.material_names[1],
            obj=self.object,
        )

        for poly in self.mesh.polygons:
            keys = []
            for kk in poly.edge_keys:
                keys += kk

            # Not that this min-lattitude key only works
            # due to the way they are structured
            min_key = min(keys)
            _, it_latt = self.get_iterators(min_key)

            if it_latt >= self.n_lattitude / 2.0 - 1:
                poly.material_index = 0
            else:
                poly.material_index = 1

    def change_transparency(self, frames, values) -> None:
        # Somehow this was needed...
        for ii in [0, 1]:
            self.object.active_material_index = ii
            self.object.active_material.blend_method = "BLEND"
            self.object.active_material.show_transparent_back = False

        # mat.blend_method = 'OPAQUE'
        for jj, color in enumerate(self.material_names):
            mat = bpy.data.materials[color]
            mat.blend_method = "BLEND"

            alpha = mat.node_tree.nodes["Principled BSDF"].inputs[21]
            for frame, value in zip(frames, values):
                # Set at start
                alpha.default_value = value
                int_frame = int(frame)
                alpha.keyframe_insert("default_value", frame=int_frame)

                if jj == 0:
                    print(f"Transparency: value={value} @ frame={int_frame}")

    def move_object(self):
        self.object.keyframe_insert(data_path="location", frame=1)

        self.object.location = (3.0, 0.0, 0.0)
        self.object.keyframe_insert(data_path="location", frame=10)

        self.object.location = (3.0, 0.0, 3.0)
        self.object.keyframe_insert(data_path="location", frame=20)

    # def unfold_at_position(self, start: int, stop: int, step: int) -> None:

    def make_unfold(self, start: int, stop: int, step: int) -> None:
        n_it = self.it_max

        # Make sure last frame is in frames
        frames = np.arange(start, stop + step, step)
        frames[-1] = stop
        n_it = frames.shape[0]
        for ii, frame in enumerate(frames):
            self.update_vertices((frame - start) / (stop - start))

            for jj, vert in enumerate(self.mesh.vertices):
                vert.co = self.vertices[jj]
                vert.keyframe_insert("co", frame=frame)

    def make_fold(self, start: int, stop: int, step: int) -> None:
        n_it = self.it_max

        # Make sure last frame is in frames
        frames = np.arange(start, stop + step, step)
        frames[-1] = stop

        n_it = frames.shape[0]
        for ii, frame in enumerate(frames):
            self.update_vertices((stop - frame) / (stop - start))

            for jj, vert in enumerate(self.mesh.vertices):
                vert.co = self.vertices[jj]
                vert.keyframe_insert("co", frame=frame)

    def get_index(self, it_long: int, it_latt: int) -> int:
        if it_latt < 0:
            return 0
        return 1 + it_long + it_latt * self.n_longitude

    def get_iterators(self, it: int) -> tuple[int]:
        it_latt = int((it - 1) / self.n_longitude)
        it_long = (it - 1) % self.n_longitude
        return (it_long, it_latt)

    def update_vertices(self, frac_weight: float) -> None:
        for lo in range(self.n_longitude):
            position_parent = np.array(self.vertices[self.get_index(-1, -1)])

            for la in range(self.n_lattitude):
                index = self.get_index(lo, la)

                vector = directional_vector_addition(
                    self.vector_init[index], self.vector_final[:, index], frac_weight
                )
                position_parent = position_parent + vector
                self.vertices[index] = tuple(position_parent)

    def evaluate_vector_final(self):
        dxy = math.pi / self.n_lattitude
        delta_long = 2 * math.pi / self.n_longitude

        for lo in range(self.n_longitude):
            vect = (
                0,
                dxy * math.sin(lo * delta_long),
                dxy * math.cos(lo * delta_long),
            )
            for la in range(self.n_lattitude):
                self.vector_final[:, self.get_index(lo, la)] = vect

    def create_edges_and_faces(self):
        # + vector init
        for ii in range(self.n_longitude):
            ii_mod = (ii + 1) % self.n_longitude
            self.edges.append((self.get_index(-1, -1), self.get_index(ii, 0)))

            self.vector_init[self.get_index(ii, 0)] = np.array(
                self.vertices[self.get_index(ii, 0)]
            ) - np.array(self.vertices[self.get_index(-1, -1)])

            self.faces.append(
                (
                    self.get_index(-1, -1),
                    self.get_index(ii, 0),
                    self.get_index(ii_mod, 0),
                )
            )
            for jj in range(1, self.n_lattitude):
                self.edges.append((self.get_index(ii, jj - 1), self.get_index(ii, jj)))
                self.edges.append((self.get_index(ii, jj), self.get_index(ii_mod, jj)))

                self.vector_init[self.get_index(ii, jj)] = np.array(
                    self.vertices[self.get_index(ii, jj)]
                ) - np.array(self.vertices[self.get_index(ii, jj - 1)])

                self.faces.append(
                    (
                        self.get_index(ii, jj),
                        self.get_index(ii_mod, jj),
                        self.get_index(ii_mod, jj - 1),
                        self.get_index(ii, jj - 1),
                    )
                )

    def create_vertices(self):
        delta_long = 2 * math.pi / self.n_longitude
        delta_latt = math.pi / self.n_lattitude

        self.vertices[self.get_index(-1, -1)] = (1.0, 0.0, 0.0)

        for la in range(self.n_lattitude):
            lattitude = (1 + la) * delta_latt
            sin_ = math.sin(lattitude)
            xx = math.cos(lattitude)

            for lo in range(self.n_longitude):
                longitude = lo * delta_long
                zz = math.cos(longitude) * sin_
                yy = math.sin(longitude) * sin_
                self.vertices[self.get_index(lo, la)] = (xx, yy, zz)


class SeparatingPlane:
    def __init__(self, scale: float = 2):
        bpy.ops.mesh.primitive_plane_add(
            location=(0, 0, 0),
            rotation=deg_to_euler([0, 90, 0]),
            scale=(scale, scale, scale),
        )
        # self.object.rotation_mode = "XYZ"

        self.object = bpy.context.object
        self.object.scale = (scale, scale, scale)

        create_color(hex_to_rgba("0b50f0ff"), "brown", obj=self.object)


class SeparatingCircle:
    def __init__(self, radius: float, color=""):
        # self.object.rotation_mode = "XYZ"
        # bpy.ops.mesh.primitive_circle_add(
        #     location=(1, 0, 0),
        #     rotation=deg_to_euler([0, 90, 0]),
        #     scale=(radius, radius, radius),
        # )
        self.initial_radius = radius
        self.radius = radius

        dx = 0.05
        head = bpy.ops.mesh.primitive_cone_add(
            radius1=radius - dx * 0.5,
            radius2=radius + dx * 0.5,
            depth=0.0,
            # location=tuple(root),
            location=(1 + 0.05, 0, 0),  # Put a bit to the front to show
            rotation=deg_to_euler([0, 90, 0]),
            end_fill_type="NOTHING",
        )

        self.object = bpy.context.object
        polygons = self.object.data.polygons
        polygons.foreach_set("use_smooth", [True] * len(polygons))

        create_color(hex_to_rgba("0b50f0ff"), "brown", obj=self.object)

    def scale(self, scale: float, frame1: int, frame2: Optional[int] = None):
        if frame2 is None:
            frame2 = frame1
        else:
            self.object.keyframe_insert("scale", frame=frame1)

        # Update radius
        self.radius = self.initial_radius * scale

        # Scale and save
        self.object.scale = (scale, scale, scale)
        self.object.keyframe_insert("scale", frame=frame2)
