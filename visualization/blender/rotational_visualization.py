import math
import shutil
from pathlib import Path
from dataclasses import dataclass

import bpy
import bmesh

import numpy as np

from nonlinear_avoidance.vector_rotation import directional_vector_addition

from rotational_mesh import RotationalMesh
from materials import create_color, hex_to_rgba

# from create_arrow import create_arrow

Vertice = tuple[float]


def show_render(obj, start, stop):
    obj.hide_render = True
    obj.keyframe_insert("hide_render", frame=start)
    obj.hide_render = False
    obj.keyframe_insert("hide_render", frame=stop)


def show_viewport(obj, start, stop):
    obj.hide_viewport = True
    obj.keyframe_insert("hide_viewport", frame=start)
    obj.hide_viewport = False
    obj.keyframe_insert("hide_viewport", frame=stop)


def hide_render(obj, start, stop):
    # # Make whole object disappear
    obj.hide_render = False
    obj.keyframe_insert("hide_render", frame=start)
    obj.hide_render = True
    obj.keyframe_insert("hide_render", frame=stop)


def hide_viewpoert(obj, start: int, stop: int):
    obj.hide_viewport = False
    obj.keyframe_insert("hide_viewport", frame=start)
    obj.hide_viewport = True
    obj.keyframe_insert("hide_viewport", frame=stop)


def make_object_disappear(obj, start: int, stop: int):
    for mat in obj.data.materials:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Alpha"].default_value = 1.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=start)
        bsdf.inputs["Base Color"].default_value[3] = 1.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

        bsdf.inputs["Alpha"].default_value = 0.0
        bsdf.inputs["Alpha"].keyframe_insert(data_path="default_value", frame=stop)
        bsdf.inputs["Base Color"].default_value[3] = 0.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

    hide_render(obj, stop, stop + 1)


def make_object_appear(obj, start: int, stop: int, alpha: float = 1.0):
    for mat in obj.data.materials:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Alpha"].default_value = 0.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=start)
        bsdf.inputs["Base Color"].default_value[3] = 0.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Alpha"].keyframe_insert(data_path="default_value", frame=stop)
        bsdf.inputs["Base Color"].default_value[3] = alpha
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

    # Hide oject one key frame after
    show_render(obj, start - 1, start)


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
    def __init__(self, start_position: np.ndarray, radius: float = 0.2):
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=tuple(start_position),
            align="WORLD",
            scale=(radius, radius, radius),
        )
        self.object = bpy.context.object

        # Make object from mesh
        # cube_object = bpy.data.objects.new("Cube", cube_mesh)
        create_color(hex_to_rgba("b07c7cff"), "", obj=self.object)

    def go_to(self, position, start: int, stop: int) -> None:
        self.object.keyframe_insert(data_path="location", frame=start)
        self.object.location = tuple(position)
        self.object.keyframe_insert(data_path="location", frame=stop)

    def follow_path(self, start: int, step: int, path: np.ndarray) -> None:
        for ii in range(path.shape[1]):
            self.object.location = tuple(path[:, ii])
            self.object.keyframe_insert(data_path="location", frame=start + ii * step)


def create_point_movement(start_position, end_position):
    return bpy.ops.mesh.primitive_cube_add(
        location=tupel(1.5, 0, 0),
    )


class ArrowBlender:
    def __init__(self, root, direction, name=""):
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
        obj = bpy.context.object
        bm.from_mesh(obj.to_mesh())

        # Add cone
        head_depth = length * (1 - ratio_shaft_length)
        head = bpy.ops.mesh.primitive_cone_add(
            radius1=ratio_radius_head * length,
            radius2=0.0,
            depth=head_depth,
            # location=tuple(root),
            location=(0, 0, shaft_depth + head_depth * 0.5),
        )
        obj = bpy.context.object
        bm.from_mesh(obj.to_mesh())

        breakpoint()
        # mesh_copy = bpy.data.meshes.new("meshCopy")
        self.object = bpy.data.objects.new("arrow" + name, bm.to_mesh())

        # mesh_copy.to_mesh(self.object.data)
        # mesh_copy.free()

        # self.object = bpy.data.objects.new("arrow" + name, bm)

    @classmethod
    def from_root_to_tip(cls, root, tip, name=""):
        return cls(root, tip - root)


def print_meshes(filepath: Path):
    # load all meshes
    with bpy.data.libraries.load(str(filepath)) as (data_from, data_to):
        data_to.meshes = data_from.meshes

        # for mesh in data_to.meshes:
        #     if mesh is not None:
        #         breakpoint()
        #         print(mesh)
    try:
        bpy.data.objects["Cube"].select_set(True)
    except KeyError:
        print("No Cube in the scene.")
    else:
        bpy.ops.object.delete()

    # now operate directly on the loaded data
    for mesh in data_to.meshes:
        if mesh is not None:
            print(mesh.name)


def render_video():
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"


def main():
    filepath = Path.home() / "Videos" / "rotational_visualization.blend"
    print_meshes(filepath)

    # # Import 'layer' / scene
    if "Main" not in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers.new("Main")
    # bpy.ops.file.pack_all()

    # bpy.ops.scene.new(type="NEW")

    new_context = bpy.context.scene
    bpy.context.scene.name = "Main"
    bpy.context.scene.frame_end = 100

    print(f"newScene: {new_context}")
    # bpy.data.scenes[len(bpy.data.scenes) - 1].name = "Main"

    cube_obstacle = CubeObstacle([3.0, 0, 0])
    # cube_obstacle.make_appear(150, 200)
    make_object_disappear(cube_obstacle.object, 50, 60)
    make_object_appear(cube_obstacle.object, 80, 140)

    ### Rotational Mesh
    rotational = RotationalMesh(12, 12)
    rotational.make_unfold(50, 100, 10)
    make_object_appear(rotational.object, 20, 30)
    # rotational.make_fold(150, 200, 10)

    agent_start = [-12.0, 1, 0]
    agent_stop = [0, 0, 0.0]
    agent = MovingSphere(agent_start)
    agent.go_to(agent_stop, 0, 30)

    velocity_init = np.array(agent_stop) - np.array(agent_start)
    velocity_init = velocity_init / np.linalg.norm(velocity_init)
    velocity_arrow = ArrowBlender(agent_start, velocity_init)

    # write all meshes starting with a capital letter and
    # set them with fake-user enabled so they aren't lost on re-saving
    # data_blocks = {mesh for mesh in bpy.data.meshes}

    # bpy.data.libraries.write(str(filepath), {rotational.mesh}, fake_user=True)
    bpy.data.libraries.write(str(filepath), {new_context})

    # data_blocks = {
    #     *bpy.context.selected_objects,
    #     *bpy.data.images,
    #     *bpy.data.materials,
    #     *bpy.data.textures,
    #     *bpy.data.node_groups,
    # }
    # # bpy.data.libraries.write(str(filepath), data_blocks, compress=True)
    # bpy.ops.file.unpack_all()  # Will unpack ALL files in the current file.


def test_check_blender():
    blender_bin = shutil.which("blender")

    if blender_bin:
        print("Found:", blender_bin)
        bpy.app.binary_path = blender_bin
    else:
        print("Unable to find blender!")


if (__name__) == "__main__":
    # test_check_blender()

    main()

    print("Done")
