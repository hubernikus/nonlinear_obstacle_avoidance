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
from rotational_mesh import get_circle_point
from materials import create_color, hex_to_rgba
from creators import delete_all_meshes
from blender_math import get_quat_from_direction, deg_to_euler
from blender_math import DirectionalSpaceTransformer
from motion import make_disappear, make_appear, move_to, align_with
from objects import ArrowBlender, MovingSphere, ObjectAssembly, CubeObstacle, Line3D
from objects import BlenderAttractor


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
        self.object.rotation_euler = deg_to_euler([75, 0, -140])
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


def create_lights_and_background():
    background = bpy.data.worlds["World"].node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.00719348, 0.00719348, 0.00719348, 1)
    background.inputs["Strength"].default_value = 0.0
    # background.inputs["Strength"].default_value = 5.0

    # bpy.context.scene.view_settings.exposure = 2
    bpy.context.scene.view_settings.exposure = 1.0

    # orphan_lights = [c for c in bpy.data.lights if not c.users]
    orphan_lights = [c for c in bpy.data.lights]
    while orphan_lights:
        bpy.data.lights.remove(orphan_lights.pop())

    # for light in bpy.data.lights:
    #     breakpoint()
    #     light.outliner.item_activate(deselect_all=True)
    #     bpy.ops.object.delete(use_global=False, confirm=False)

    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_1", type="POINT")
    light_data.energy = 1000
    light_data.diffuse_factor = 1.0
    light_data.specular_factor = 1.0
    light_data.volume_factor = 0.48
    light_data.shadow_soft_size = 3
    light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    # light_object.location = (-20, 20, 0)
    light_object.location = (-10, 10, 0)
    light_object.rotation_euler = deg_to_euler([90, 0, -135])

    # create new object with our light datablock
    light_data = bpy.data.lights.new(name="light_2", type="POINT")
    light_data.energy = 800
    light_data.diffuse_factor = 2.5
    light_data.specular_factor = 0
    light_data.volume_factor = 0.0
    light_data.shadow_soft_size = 3
    light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="light_2", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    # light_object.location = (-20, -20, 0)
    light_object.location = (-10, -10, 0)
    light_object.rotation_euler = deg_to_euler([90, 0, -45])

    # create new object with our light datablock
    light_data = bpy.data.lights.new(name="light_3", type="POINT")
    light_data.energy = 100
    light_data.diffuse_factor = 10
    light_data.specular_factor = 0
    light_data.volume_factor = 1.0
    light_data.shadow_soft_size = 7
    light_data.use_shadow = False
    light_object = bpy.data.objects.new(name="light_3", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (10, 0, 20)
    light_object.rotation_euler = deg_to_euler([0, 0, 0])


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

    make_disappear(scene.agent, frame, frame + 1)
    # make_appear(scene.rotational, frame, frame + 1)
    make_appear(scene.half_circle, frame + df - 5, frame + df)

    scene.rotational.make_unfold(frame, frame + df, 10)
    scene.camera.to_direction_space(frame, frame + df, center=agent.location)

    make_appear(scene.normal_point, frame, frame + 1)
    move_to(
        scene.normal_point,
        -1 * scene.normal_direction + scene.agent.location,
        frame,
        frame + 1,
    )


def do_scene_folding(scene, frame1, frame2):
    frame = frame1
    df = frame2 - frame1

    # scene.camera.to_midpoint(frame, frame + df)
    scene.rotational.make_fold(frame, frame + df, 10)

    make_disappear(scene.half_circle, frame, frame + 5)
    make_disappear(scene.normal_point, frame, frame + 1)
    # make_disappear(scene.rotational, frame + df - 1, frame + df)

    make_appear(scene.agent, frame + df - 1, frame + df)


def to_direction_space(point, scene, frame1, frame2):
    frame = frame1
    df = frame2 - frame1

    dir_vel = scene.transformer.transform_to_direction_space(point.location)
    move_to(point, dir_vel, frame, frame + df)


class SceneStorer:
    def __init__(self):
        self.rel_rot_center = np.array([1.0, 0, 0])

        self.dir_point_radius = 0.1
        self.camera = None
        self.rotational = None
        self.normal_direction = np.array([-1, 0, 0.0])

        self.normal_point = MovingSphere(
            -1 * self.normal_direction,
            radius=self.dir_point_radius,
            color="#36080C",
        )

        self.transformer = DirectionalSpaceTransformer.from_vector(
            (-1) * self.normal_direction, center=self.rel_rot_center
        )
        self.half_circle = SeparatingCircle(radius=math.pi * 0.5)
        self.velocity_point = None


def main(render_scene=False):
    # Filepath and clean-up
    filepath = Path.home() / "Videos" / "rotation_complex.blend"
    delete_all_meshes(filepath)

    fps = 24
    convergence_color = "0b8095ff"

    # # Import 'layer' / scene
    if "Main" not in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers.new("Main")

    new_context = bpy.context.scene
    bpy.context.scene.name = "Main"
    print(f"newScene: {new_context}")

    # bpy.context.scene.render.film_transparent = True
    bpy.context.scene.view_settings.view_transform = "Standard"

    create_lights_and_background()

    scene = SceneStorer()
    scene.camera = CameraPoser()
    scene.camera.to_global0(0)  # Start at global

    # Attractor
    # attractor_position = np.array([4, 0, 2.5])
    attractor_position = np.array([6, -4, 2])
    BlenderAttractor(
        attractor_position,
        rotation=(45.0, 45.0, 0),
        scale=(0.2, 0.2, 0.2),
        hexcolor=convergence_color,
    )

    obstacle_assembly = []
    main_obstacle = ObjectAssembly()
    object_center = [4.0, 3.0, 0]
    # rotation = (0.0, math.pi * 1 / 5, math.pi * 1 / 3)
    rotation = (0.0, 0.0, 0.0)
    ax0 = 4.0
    ax1 = 1.0
    main_obstacle.append(CubeObstacle(object_center, scale=(ax0, ax1, ax1)))
    main_obstacle.append(CubeObstacle(object_center, scale=(ax1, ax0, ax1)))
    main_obstacle.append(CubeObstacle(object_center, scale=(ax1, ax1, ax0)))

    ### Movement of Agent
    start_pos = np.array([-6, -2, 4])
    scene.agent = MovingSphere(start_pos, radius=0.2, color="#191919")
    velocity = (-1) * start_pos / np.linalg.norm(start_pos)
    velocity_arrow = ArrowBlender(
        scene.agent.location, direction=velocity, color="0000ffff"
    )
    # scene.rotational = RotationalMesh(32, 32)
    scene.rotational = RotationalMesh(16, 16)

    # Setup initial elements
    frame = 0
    df = fps * 6
    end_position = scene.agent.location + velocity_arrow.direction * 3
    move_to(scene.agent, end_position, frame, frame + df)
    move_to(velocity_arrow.object, end_position, frame, frame + df)

    # for radius in [math.pi * 3 / 4, math.pi]:
    camera_poses = [scene.camera.to_global1, scene.camera.to_global0]

    # Velocity step
    end_position = scene.agent.location + velocity_arrow.direction * 4
    move_to(scene.agent, end_position, frame, frame + df)
    move_to(velocity_arrow.object, end_position, frame, frame + df)

    # Update transformer
    scene.transformer.center = scene.agent.location
    move_to(scene.rotational, scene.agent.location, frame, frame + df)
    scene.transformer.center = scene.agent.location + scene.rel_rot_center
    delta_circle = (-0.1, 0, 0)
    move_to(
        scene.half_circle, scene.transformer.center + delta_circle, frame, frame + df
    )

    # Create relevant arrows
    frame = frame + df
    df = 3 * fps
    ref_dir = object_center - scene.agent.location
    ref_dir = ref_dir / np.linalg.norm(ref_dir)
    ref_arrow = ArrowBlender(scene.agent.location, ref_dir, color="3b1e78ff")
    make_appear(ref_arrow, frame, frame + df)

    frame = frame + df
    df = 3 * fps
    conv_dir = attractor_position - scene.agent.location
    # conv_dir = np.array([1, 0, 0.5])
    conv_dir = conv_dir / np.linalg.norm(conv_dir)
    conv_arrow = ArrowBlender(scene.agent.location, conv_dir, color=convergence_color)
    make_appear(conv_arrow, frame, frame + df)

    ### Create Rotation-Mesh
    frame = frame + df
    df = 2 * fps
    make_appear(scene.rotational.object, frame - 1, frame)
    scene.rotational.change_transparency(
        frames=[frame, frame + df * 1.5], values=[0.0, 1.0]
    )
    velocity_point = create_sphere_from_arrow(
        velocity_arrow, scene.agent, frame, frame + df
    )
    conv_point = create_sphere_from_arrow(conv_arrow, scene.agent, frame, frame + df)
    ref_point = create_sphere_from_arrow(ref_arrow, scene.agent, frame, frame + df)

    ### Unfold
    frame = frame + df
    df = 3.0 * fps
    main_obstacle.make_disappear(frame, frame + df * 0.5)  # Get out fast
    do_scene_unfolding(scene, scene.agent, frame, frame + df)
    # scene.rotational.change_transparency(
    #     frames=[frame + df * 0.8, frame + 1.1 * df], values=[0.0, 1.0]
    # )
    # Move velocity point
    to_direction_space(velocity_point, scene, frame, frame + df)
    to_direction_space(conv_point, scene, frame, frame + df)
    to_direction_space(ref_point, scene, frame, frame + df)

    ### Break
    frame = frame + df
    df = int(0.1 * fps)

    ### Create Tangent
    frame = frame + df
    df = int(0.5 * fps)
    surface_point = get_circle_point(conv_point, ref_point, scene=scene)
    conf_ref_line = Line3D(ref_point, surface_point, color="000000")
    make_appear(conf_ref_line, frame + df - 1, frame + df)

    # Move to tangent
    frame = frame + df
    df = 2 * fps
    move_to(conv_point, surface_point, frame, frame + df)
    make_disappear(ref_point, frame + df - 1, frame + df)
    make_disappear(conf_ref_line, frame + df - 1, frame + df)

    tang_point = conv_point

    ### Create line
    frame = frame + df
    df = int(0.5 * fps)
    ww = 0.8
    vel_tang_line = Line3D(velocity_point, tang_point, color="000000")
    make_appear(vel_tang_line, frame + df - 1, frame + df)

    ### Move velocity
    frame = frame + df
    df = 2 * fps
    mid_point = ww * tang_point.location + (1 - ww) * velocity_point.location
    move_to(velocity_point, mid_point, frame, frame + df)
    make_disappear(vel_tang_line, frame + df - 1, frame + df)

    # Pause [look at velocity]
    frame = frame + df
    df = 1 * fps

    ### Fold
    frame = frame + df
    df = 2 * fps
    do_scene_folding(scene, frame, frame + df)
    scene.rotational.change_transparency(
        frames=[frame + df * 0.5, frame + 1.5 * df], values=[1.0, 0.0]
    )

    # Restore vector
    vec = scene.transformer.transform_from_direction_space(velocity_point.location)
    move_to(velocity_point, scene.agent.location + vec, frame, frame + df)
    main_obstacle.make_appear(frame + df * 0.3, frame + df * 0.4)
    scene.camera.to_global0(frame, frame + df)
    make_appear(velocity_arrow, frame + df - 1, frame)  # Get out fast

    velocity_arrow.align_with(vec, frame, frame + df)
    velocity_arrow.direction = vec
    make_disappear(tang_point, frame, frame + 1)

    # Break
    frame = frame + df
    df = int(0.5 * fps)
    make_disappear(velocity_point, frame + df * 0.5, frame)  # Get out fast

    ### Pause
    frame = frame
    df = 0.5 * fps

    ### Move out
    frame = frame + df
    df = 2 * fps
    end_position = scene.agent.location + velocity_arrow.direction * 7
    move_to(scene.agent, end_position, frame, frame + df)
    move_to(velocity_arrow.object, end_position, frame, frame + df)

    # half_circle.scale(2.0, 30, 40)
    bpy.context.scene.frame_end = int(frame + df)

    # Ouput settings
    videopath = Path.home() / "Videos" / "direction_space_complex"
    bpy.context.scene.render.filepath = str(videopath)
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    # Write to file (!)
    bpy.data.libraries.write(str(filepath), {new_context})

    if render_scene:
        bpy.ops.render.render(write_still=False)
        print(f"Rendered to: {filepath}")


if (__name__) == "__main__":
    # print(hex_to_rgba("#191919"))
    main(render_scene=False)

    print("Done")
