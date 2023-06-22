import math
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import bpy
import bmesh

import mathutils

import numpy as np

from vartools.math import get_intersection_with_circle

from nonlinear_avoidance.vector_rotation import directional_vector_addition

# (Local) blender-tools
from rotational_mesh import RotationalMesh, SeparatingCircle, SeparatingPlane
from materials import create_color, hex_to_rgba
from creators import delete_all_meshes
from blender_math import get_quat_from_direction, deg_to_euler
from blender_math import DirectionalSpaceTransformer

from motion import make_disappear, make_appear, move_to

from objects import ArrowBlender, MovingSphere, CubeObstacle, Line3D

# from create_arrow import create_arrow


class CameraPoser:
    object = bpy.data.objects["Camera"]

    def store_camera_keyframe(self, frame):
        self.object.keyframe_insert("rotation_euler", frame=frame)
        self.object.keyframe_insert("location", frame=frame)
        self.object.data.keyframe_insert("lens", frame=frame)

    def set_mid_point(self, frame1, frame2, fraction=1.0 / 2):
        frame = int(fraction * (frame2 - frame1) + frame1)
        self.object.location = [6, 12, 3.0]
        self.object.keyframe_insert("location", frame=frame)
        # self.object.rotation_mode = "XYZ"
        # self.object.rotation_euler = deg_to_euler([80, 0, 132])
        # self.object.data.lens = 55
        # self.store_camera_keyframe(frame)

    def to_global_view(self, start: int, stop: Optional[int] = None):
        if stop is None:
            stop = start
        else:
            self.store_camera_keyframe(start)
            # self.set_mid_point(start, stop)

        self.object.location = [-12, 16, 6.0]
        # self.object.rotation_quqaternion = [-0.2, -0.16, 0.6, 0.8]
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([75, 0, 360 - 150])
        self.object.data.lens = 55
        self.store_camera_keyframe(stop)

    def to_midpoint(self, frame1: int, frame2: Optional[int] = None):
        if frame2 is None:
            frame2 = frame1
        else:
            self.store_camera_keyframe(frame1)

        self.object.location = [6, 12, 3.0]
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([82.5, 0, 150])
        self.object.data.lens = 40
        self.store_camera_keyframe(frame2)

    def to_direction_space(self, start: int, stop: Optional[int] = None):
        if stop is None:
            stop = start
        else:
            self.store_camera_keyframe(start)
            # self.set_mid_point(start, stop)

        self.object.location = [10.0, 0, 0]
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([90, 0, 90])
        self.object.data.lens = 25

        self.store_camera_keyframe(stop)

    def to_final_move(self, frame1, frame2):
        self.store_camera_keyframe(frame1)

        self.object.location = [-12, 16, 6.0]
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([75, 0, 232])
        self.object.data.lens = 30

        self.store_camera_keyframe(frame2)


def get_circle_point(point, start_point, radius, weight: float = 1.0, center=[1, 0, 0]):
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


def render_video():
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"


def create_lights():
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_1", type="POINT")
    light_data.energy = 500
    # light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, -5, 5)

    # create new object with our light datablock
    light_data = bpy.data.lights.new(name="light_2", type="POINT")
    light_data.energy = 500
    # light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="light_2", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, -2, -5)


def main(render_scene=False):
    dir_point_radius = 0.1
    # Filepath and clean-up
    filepath = Path.home() / "Videos" / "rotational_visualization.blend"
    delete_all_meshes(filepath)

    # # Import 'layer' / scene
    if "Main" not in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers.new("Main")

    # bpy.context.scene.render.film_transparent = True
    bpy.context.scene.view_settings.view_transform = "Standard"

    new_context = bpy.context.scene
    bpy.context.scene.name = "Main"
    print(f"newScene: {new_context}")

    create_lights()
    # Setup initial elements
    frame = 0
    df = 300
    camera = CameraPoser()
    camera.to_global_view(frame)

    ### Movement of Agent
    cube_obstacle = CubeObstacle([3.0, 0, 0])

    # Setup Agent
    agent_start = [-12.0, -6, 0]
    agent_stop = [0, 0, 0.0]
    agent = MovingSphere(agent_start)
    agent.go_to(agent_stop, frame, frame + df)

    # Setup velocity arrow
    velocity_init = np.array(agent_stop) - np.array(agent_start)
    velocity_init = velocity_init / np.linalg.norm(velocity_init)
    velocity_arrow = ArrowBlender(agent_start, velocity_init, color="0000ffff")
    move_to(velocity_arrow.object, agent_stop, frame, frame + df)

    ### Pause
    frame = frame + 10

    ### Create Obstacle Normal
    frame = frame + df
    df = 60

    normal_vector = np.array([-1, 0, 0])
    dir_transformer = DirectionalSpaceTransformer.from_vector(
        (-1) * normal_vector, center=[1, 0.0, 0]
    )
    normal_start = [2, 0, 0]
    normal_arrow = ArrowBlender(normal_start, normal_vector, color="6d1119ff")
    make_appear(normal_arrow, frame, frame + df)

    # Rotation-Mesh [include blend]
    frame = frame + df
    df = 30
    rotational = RotationalMesh(32, 32)
    make_appear(rotational.object, frame, frame + df)
    camera.to_midpoint(frame, frame + df)
    rotational.change_transparency(frames=[360, 500], values=[0.0, 1.0])

    ### Convert vectors to points
    frame = frame + df
    df = 30
    normal_point = MovingSphere(
        -1 * normal_arrow.direction,
        radius=dir_point_radius,
        color=normal_arrow.hex_color,
    )
    make_appear(normal_point, frame, frame + df)

    velocity_point = MovingSphere(
        velocity_arrow.direction,
        radius=dir_point_radius,
        color=velocity_arrow.hex_color,
    )
    make_appear(velocity_point, frame, frame + df)
    make_disappear(normal_arrow, frame, frame + df)
    make_disappear(velocity_arrow, frame, frame + df)

    ### Pause
    frame = frame + df
    df = 30

    ### Plane
    frame = frame + df
    df = 30
    half_plane = SeparatingPlane()
    make_appear(half_plane.object, frame, frame + df, alpha=0.6)

    ### Pause
    frame = frame + df
    df = 30

    ### Unfold
    frame = frame + df
    df = 150
    make_disappear(half_plane, frame, frame + df * 0.5)  # Get out fast
    make_disappear(cube_obstacle, frame, frame + df * 0.5)  # Get out fast
    half_circle = SeparatingCircle(radius=math.pi * 0.5)
    make_appear(half_circle, frame + df - 5, frame + df)

    rotational.make_unfold(frame, frame + df, 10)
    camera.to_direction_space(frame, frame + df)

    dir_nor = dir_transformer.transform_to_direction_space(
        (-1) * normal_arrow.direction
    )
    move_to(normal_point, dir_nor, frame, frame + df)
    dir_vel = dir_transformer.transform_to_direction_space(velocity_arrow.direction)
    move_to(velocity_point, dir_vel, frame, frame + df)
    # rotational.make_fold(150, 200, 10)
    # Todo -> update vectors

    ### Create surface line
    frame = frame + df
    df = 30
    surface_point = get_circle_point(velocity_point, normal_point, half_circle.radius)
    surf_line = Line3D(normal_point.location, surface_point, color="000000")
    make_appear(surf_line, frame + df - 1, frame + df)

    ### Move Vector ()
    frame = frame + df
    df = 70
    move_to(velocity_point, surface_point, frame, frame + df)

    ### Pause
    frame = frame + df
    df = 30
    make_disappear(surf_line, frame, frame + 1)

    ### Fold
    frame = frame + df
    df = 100
    camera.to_midpoint(frame, frame + df)
    rotational.make_fold(frame, frame + df, 10)
    make_appear(cube_obstacle, frame + df * 0.3, frame + df * 0.4)

    vel_vector = dir_transformer.transform_from_direction_space(velocity_point.location)
    move_to(velocity_point, vel_vector, frame, frame + df)
    make_disappear(normal_point, frame, frame + df * 0.5)
    make_disappear(half_circle, frame, frame + 5)

    ### Points to vector
    frame = frame + df
    df = 30
    modulated_arrow = ArrowBlender(
        agent.location, vel_vector, color=velocity_point.hex_color
    )
    make_appear(modulated_arrow, frame, frame + df)
    make_disappear(velocity_point, frame, frame + df)
    # make_disappear(rotational, frame, frame + df)
    rotational.change_transparency(frames=[890, 948], values=[1.0, 0.0])

    ### Pause
    frame = frame + 10

    ### Move out of the scene
    frame = frame + df
    df = 100
    # camera.to_global_view(frame, frame + df * 0.6)
    camera.to_final_move(frame - 10, frame + df * 0.5)
    final_position = agent.location + modulated_arrow.direction * 20
    print(f"Final position {np.round(final_position,2 )}")
    move_to(modulated_arrow, final_position, frame, frame + df)
    move_to(agent, final_position, frame, frame + df)

    # half_circle.scale(2.0, 30, 40)
    bpy.context.scene.frame_end = int(frame + df)
    background = bpy.data.worlds["World"].node_tree.nodes["Background"]
    background.inputs["Strength"].default_value = 5.0
    background.inputs["Color"].default_value = (0.00719348, 0.00719348, 0.00719348, 1)

    # Ouput settings
    # bpy.context.space_data.context = "OUTPUT"
    # filepath = Path.home() / "Videos" / "direction_space.mp4"
    videopath = Path.home() / "Videos" / "direction_space.mp4"
    bpy.context.scene.render.filepath = str(videopath)
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"

    # Write to file (!)
    bpy.data.libraries.write(str(filepath), {new_context})

    if render_scene:
        # outpath = Path.home() / "Videos" / "direction_space.mp4"
        # image_name = bpy.path.basename(str(outpath))
        # bpy.context.scene.render.filepath += "-" + image_name
        # Render still image, automatically write to output path
        bpy.ops.render.render(write_still=False)
        print(f"Rendered to: {filepath}")


def test_check_blender():
    blender_bin = shutil.which("blender")

    if blender_bin:
        print("Found:", blender_bin)
        bpy.app.binary_path = blender_bin
    else:
        print("Unable to find blender!")


if (__name__) == "__main__":
    # test_check_blender()

    main(render_scene=True)
    # scene_introduction()

    print("Done")
