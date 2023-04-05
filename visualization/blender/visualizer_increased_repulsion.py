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

from motion import make_disappear, make_appear, move_to, align_with

from objects import ArrowBlender, MovingSphere, CubeObstacle, Line3D

# from create_arrow import create_arrow


class CameraPoser:
    object = bpy.data.objects["Camera"]

    def __init(sefl):
        self.view_it = []

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

    def to_global0(self, start: int, stop=None):
        if stop is None:
            stop = start
        else:
            self.store_camera_keyframe(start)

        self.object.rotation_mode = "XYZ"

        self.object.location = [-13, 19, 6.0]
        # self.object.rotation_quqaternion = [-0.2, -0.16, 0.6, 0.8]
        self.object.rotation_euler = deg_to_euler([75, 0, -145])
        self.object.data.lens = 55
        self.store_camera_keyframe(stop)

    def to_global1(self, start, stop):
        self.store_camera_keyframe(start)
        self.object.rotation_mode = "XYZ"

        self.object.location = [-15, 12, 8.0]
        # self.object.rotation_quqaternion = [-0.2, -0.16, 0.6, 0.8]
        self.object.rotation_euler = deg_to_euler([65, 0, -125])
        self.object.data.lens = 55
        self.store_camera_keyframe(stop)

    def to_midpoint(
        self,
        frame1: int,
        frame2: Optional[int] = None,
    ):
        if frame2 is None:
            frame2 = frame1
        else:
            self.store_camera_keyframe(frame1)

        self.object.location = [6, 12, 3.0]
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([82.5, 0, 150])
        self.object.data.lens = 40
        self.store_camera_keyframe(frame2)

    def to_direction_space(
        self, frame1: int, frame2: Optional[int] = None, center=[1, 0, 0]
    ):
        if frame2 is None:
            fram2 = frame1
        else:
            self.store_camera_keyframe(frame1)

        self.object.location = np.array(center) + np.array([-10, 0, 0])
        self.object.rotation_mode = "XYZ"
        self.object.rotation_euler = deg_to_euler([90, 0, -90])
        self.object.data.lens = 25

        self.store_camera_keyframe(frame2)

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
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, -5, 5)

    # create new object with our light datablock
    light_data = bpy.data.lights.new(name="light_2", type="POINT")
    light_data.energy = 500
    light_object = bpy.data.objects.new(name="light_2", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, -2, -5)


def create_sphere_from_arrow(arrow, agent, frame1, frame2, dir_point_radius=0.1):
    frame = frame1
    df = frame2 - frame1

    sphere = MovingSphere(
        arrow.direction + agent.location,
        radius=dir_point_radius,
        color=arrow.hex_color,
    )
    make_disappear(arrow, frame, frame + df)
    make_appear(sphere, frame, frame + df)
    return sphere


def create_arrow_from_sphere(sphere, agent, frame1, frame2):
    frame = frame1
    df = frame2 - frame1

    arrow = ArrowBlender(
        agent.location, sphere.location - agent.location, color=sphere.hex_color
    )
    make_appear(arrow, frame, frame + df)
    make_disappear(sphere, frame, frame + df)
    return arrow


def do_scene_unfolding(scene, agent, frame1, frame2, create_normal: bool = True):
    frame = frame1
    df = frame2 - frame1

    make_appear(scene.rotational, frame, frame + 1)
    make_appear(scene.half_circle, frame + df - 5, frame + df)

    scene.rotational.make_unfold(frame, frame + df, 10)
    scene.camera.to_direction_space(frame, frame + df, center=agent.location)

    if create_normal:
        scene.normal_point = MovingSphere(
            -1 * scene.normal_direction + agent.location,
            radius=scene.dir_point_radius,
            color="6d1119ff",
        )
        make_appear(scene.normal_point, frame, frame + 1)


def do_scene_folding(scene, frame1, frame2):
    frame = frame1
    df = frame2 - frame1

    # scene.camera.to_midpoint(frame, frame + df)
    scene.rotational.make_fold(frame, frame + df, 10)

    make_disappear(scene.half_circle, frame, frame + 5)
    make_disappear(scene.normal_point, frame, frame + 1)
    make_disappear(scene.rotational, frame + df - 1, frame + df)


class SceneStorer:
    def __init__(self):
        self.rel_rot_center = np.array([1.0, 0, 0])

        self.dir_point_radius = 0.1
        self.camera = None
        self.rotational = None
        self.normal_point = None
        self.normal_direction = np.array([-1, 0, 0.0])
        self.transformer = DirectionalSpaceTransformer.from_vector(
            (-1) * self.normal_direction, center=self.rel_rot_center
        )
        self.half_circle = SeparatingCircle(radius=math.pi * 0.5)
        self.velocity_point = None


def main(render_scene=False):
    # Filepath and clean-up
    filepath = Path.home() / "Videos" / "rotational_incremental_radiuses.blend"
    delete_all_meshes(filepath)

    # # Import 'layer' / scene
    if "Main" not in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers.new("Main")

    new_context = bpy.context.scene
    bpy.context.scene.name = "Main"
    print(f"newScene: {new_context}")

    create_lights()

    scene = SceneStorer()
    scene.camera = CameraPoser()
    scene.camera.to_global0(0)  # Start at global

    cube_obstacle = CubeObstacle([3.0, 0, 0], scale=(0.5, 15, 5))

    ### Movement of Agent
    agent = MovingSphere([0, -6, 0])
    velocity_arrow = ArrowBlender(agent.location, direction=[0, 1, 0], color="0000ffff")
    # scene.rotational = RotationalMesh(32, 32)
    scene.rotational = RotationalMesh(16, 16)

    # Setup initial elements
    frame = 0
    df = 30
    end_position = agent.location + velocity_arrow.direction * 3
    move_to(agent, end_position, frame, frame + df)
    move_to(velocity_arrow.object, end_position, frame, frame + df)

    # for radius in [math.pi * 3 / 4, math.pi]:
    camera_poses = [scene.camera.to_global1, scene.camera.to_global0]
    for radius, camera_global in zip([math.pi * 5 / 8, math.pi], camera_poses):
        # Velocity step
        end_position = agent.location + velocity_arrow.direction * 4
        move_to(agent, end_position, frame, frame + df)
        move_to(velocity_arrow.object, end_position, frame, frame + df)

        # Update transformer
        scene.transformer.center = agent.location
        move_to(scene.rotational, agent.location, frame, frame + df)
        scene.transformer.center = agent.location + scene.rel_rot_center
        move_to(scene.half_circle, scene.transformer.center, frame, frame + df)

        ### Create Rotation-Mesh
        frame = frame + df
        df = 30
        make_appear(scene.rotational.object, frame, frame + df)
        # scene.camera.to_midpoint(frame, frame + df)

        ### Convert vectors to points
        frame = frame + df
        df = 30
        velocity_point = create_sphere_from_arrow(
            velocity_arrow, agent, frame, frame + df
        )

        # ### Plane
        # frame = frame + df
        # df = 30
        # half_plane = SeparatingPlane()
        # make_appear(half_plane.object, frame, frame + 1, alpha=0.6)

        ### Unfold
        frame = frame + df
        df = 110
        make_disappear(cube_obstacle, frame, frame + df * 0.5)  # Get out fast
        do_scene_unfolding(scene, agent, frame, frame + df)
        # Move velocity point
        dir_vel = scene.transformer.transform_to_direction_space(
            velocity_point.location
        )
        move_to(velocity_point, dir_vel, frame, frame + df)

        ### Break
        frame = frame + df
        df = 10

        ### Move Vector ()
        frame = frame + df
        df = 40
        scene.half_circle.scale(
            radius / scene.half_circle.initial_radius, frame, frame + df
        )
        surface_point = get_circle_point(
            velocity_point,
            scene.normal_point,
            scene.half_circle.radius,
            center=scene.normal_point.location,
        )
        move_to(velocity_point, surface_point, frame, frame + df)

        ### Break
        frame = frame + df
        df = 10

        ### Fold
        frame = frame + df
        df = 60
        do_scene_folding(scene, frame, frame + df)
        vec = scene.transformer.transform_from_direction_space(velocity_point.location)
        move_to(velocity_point, agent.location + vec, frame, frame + df)
        make_appear(cube_obstacle, frame + df * 0.3, frame + df * 0.4)
        camera_global(frame, frame + df)
        make_appear(velocity_arrow, frame, frame + df * 0.5)  # Get out fast

        ### Points to vector
        frame = frame + df
        df = 30
        align_with(velocity_arrow, vec, frame, frame + df)
        make_disappear(velocity_point, frame + df * 0.5, frame)  # Get out fast
        make_appear(velocity_arrow, frame, frame + df * 0.5)  # Get out fast

        velocity_arrow.direction = vec
        # move_to(velocity_arrow, vec, frame, frame + df)

        ### Pause
        frame = frame
        df = 10

    ### Pause
    frame = frame + df
    df = 50
    end_position = agent.location + velocity_arrow.direction * 7
    move_to(agent, end_position, frame, frame + df)
    move_to(velocity_arrow.object, end_position, frame, frame + df)

    # half_circle.scale(2.0, 30, 40)
    bpy.context.scene.frame_end = int(frame + df)
    background = bpy.data.worlds["World"].node_tree.nodes["Background"]
    background.inputs["Strength"].default_value = 5.0
    background.inputs["Color"].default_value = (0.00719348, 0.00719348, 0.00719348, 1)

    # Ouput settings
    # bpy.context.space_data.context = "OUTPUT"
    # filepath = Path.home() / "Videos" / "direction_space.mp4"
    videopath = Path.home() / "Videos" / "direction_space_incremental_radiuses"
    bpy.context.scene.render.filepath = str(videopath)
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"

    # Write to file (!)
    bpy.data.libraries.write(str(filepath), {new_context})

    if render_scene:
        bpy.ops.render.render(write_still=False)
        print(f"Rendered to: {filepath}")


if (__name__) == "__main__":
    main(render_scene=False)

    print("Done")
