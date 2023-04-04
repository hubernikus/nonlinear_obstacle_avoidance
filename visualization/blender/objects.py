import numpy as np
from typing import Optional
from materials import create_color, hex_to_rgba

import bpy
import bmesh

from blender_math import get_quat_from_direction, deg_to_euler


class Line3D:
    def __init__(self, point1: float, point2: float, color=""):
        dx = 0.01
        length = np.linalg.norm(point2 - point1)
        bpy.ops.mesh.primitive_cube_add(
            location=0.5 * (point1 + point2), scale=(dx, dx, length * 0.5)
        )

        self.object = bpy.context.object
        self.object.rotation_mode = "QUATERNION"
        self.object.rotation_quaternion = get_quat_from_direction(point2 - point1)
        try:
            create_color(hex_to_rgba(color), "brown", obj=self.object)
        except:
            breakpoint()


class CubeObstacle:
    def __init__(self, position):
        bpy.ops.mesh.primitive_cube_add(
            location=tuple(position), align="WORLD", scale=(1, 1, 1)
        )
        self.object = bpy.context.object

        # Make object from mesh
        # cube_object = bpy.data.objects.new("Cube", cube_mesh)
        create_color(hex_to_rgba("b07c7cff"), "brown", obj=self.object)


class MovingSphere:
    def __init__(self, position: np.ndarray, radius: float = 0.2, color="000000"):
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=tuple(position),
            align="WORLD",
            scale=(radius, radius, radius),
        )
        self.object = bpy.context.object
        self.hex_color = color

        # Make object from mesh
        # cube_object = bpy.data.objects.new("Cube", cube_mesh)
        create_color(hex_to_rgba(color), "", obj=self.object)

        # ?! why is this the wro
        polygons = self.object.data.polygons
        polygons.foreach_set("use_smooth", [True] * len(polygons))

    @property
    def location(self):
        return np.array(self.object.location)

    def go_to(self, position, start: Optional[int], stop: int) -> None:
        if start is not None:
            self.object.keyframe_insert(data_path="location", frame=start)

        self.object.location = tuple(position)
        self.object.keyframe_insert(data_path="location", frame=stop)

    def follow_path(self, start: int, step: int, path: np.ndarray) -> None:
        for ii in range(path.shape[1]):
            self.object.location = tuple(path[:, ii])
            self.object.keyframe_insert(data_path="location", frame=start + ii * step)


class ArrowBlender:
    def __init__(self, root, direction, name="", color: Optional[str] = None):

        # Store direction for later conversion
        self.direction = direction

        ratio_shaft_length = 0.6
        ratio_radius_shaft = 0.07
        ratio_radius_head = 0.2
        length = np.linalg.norm(direction)

        bm = bmesh.new()

        # Add cylinder
        shaft_depth = length * ratio_shaft_length
        shaft = bpy.ops.mesh.primitive_cylinder_add(
            radius=length * ratio_radius_shaft,
            depth=shaft_depth,
            location=(0.0, 0.0, shaft_depth * 0.5),
        )  # location will be set
        shaft_obj = bpy.context.object

        # obj = bpy.context.object
        # bm.from_mesh(obj.to_mesh())

        # Add cone
        head_depth = length * (1 - ratio_shaft_length)
        head = bpy.ops.mesh.primitive_cone_add(
            radius1=ratio_radius_head * length,
            radius2=0.0,
            depth=head_depth,
            # location=tuple(root),
            location=(0, 0, shaft_depth + head_depth * 0.5),
        )
        head_obj = bpy.context.object

        # Merge obstacles
        bpy.ops.object.select_all(action="DESELECT")
        shaft_obj.select_set(True)
        head_obj.select_set(True)
        bpy.ops.object.join()
        self.object = bpy.context.object
        # Reset origin
        saved_location = bpy.context.scene.cursor.location.xyz  # returns a vector
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        bpy.context.scene.cursor.location.xyz = saved_location
        bpy.ops.object.select_all(action="DESELECT")

        # Set center
        self.object.location = tuple(root)

        # ?! why is this the wro
        self.object.rotation_mode = "QUATERNION"
        quat = get_quat_from_direction(direction, null_vector=[0, 0, 1])
        self.object.rotation_quaternion = quat

        polygons = self.object.data.polygons
        polygons.foreach_set("use_smooth", [True] * len(polygons))

        if color is not None:
            self.hex_color = color
            create_color(
                hex_to_rgba(self.hex_color), name="ArrowColor", obj=self.object
            )
        else:
            self.hex_color = None

    def get_tip_position(self):
        return self.direction + np.array(self.object.location)

    @classmethod
    def from_root_and_tip(cls, root, tip, name=""):
        return cls(root, tip - root)
