import math
import shutil
from pathlib import Path
from dataclasses import dataclass

import bpy

import numpy as np

from nonlinear_avoidance.vector_rotation import directional_vector_addition

Vertice = tuple[float]


def is_even(value: int) -> bool:
    return not (value % 2)


def is_odd(value: int) -> bool:
    return value % 2


def hex_to_rgb(hex_value: "str") -> tuple[float]:
    if hex_value[0] == "#":
        hex_value = hex_value[1:]
    return tuple(int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def hex_to_rgba(hex_value: "str", a_value: float = 1.0) -> tuple[float]:
    if hex_value[0] == "#":
        hex_value = hex_value[1:]
    return tuple([int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [a_value])


def create_color(
    color: tuple[float], name: str, obj, use_transparency: bool = False
) -> None:
    obj.color = color
    # Create a material
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    # mat.use_transparency = True
    # mat.transparency_method = "Z_TRANSPARENCY"

    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = color

    # principled.inputs["Base Color"].use_transparency = True
    # mat.transparency_method = "Z_TRANSPARENCY"

    obj.data.materials.append(mat)


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

        # Make collection
        self.collection = bpy.data.collections.new("circle_space_collection")
        bpy.context.scene.collection.children.link(self.collection)
        self.collection.objects.link(self.object)

        self.set_colors()

        # Create animation
        # self.object.position
        print("Circle space done.")

    def set_colors(self) -> None:
        create_color(
            color=(75.0 / 255, 102.0 / 255, 20.0 / 255, 1),
            name="Green",
            obj=self.object,
        )
        create_color(
            color=(164.0 / 255, 26.0 / 255, 38.0 / 255, 1),
            name="Red",
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

    def move_object(self):
        self.object.keyframe_insert(data_path="location", frame=1)

        self.object.location = (3.0, 0.0, 0.0)
        self.object.keyframe_insert(data_path="location", frame=10)

        self.object.location = (3.0, 0.0, 3.0)
        self.object.keyframe_insert(data_path="location", frame=20)

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


class CubeObstacle:
    def __init__(self, position):
        bpy.ops.mesh.primitive_cube_add(
            location=tuple(position), align="WORLD", scale=(1, 1, 1)
        )
        self.object = bpy.context.object

        # Make object from mesh
        # cube_object = bpy.data.objects.new("Cube", cube_mesh)
        create_color(hex_to_rgba("b07c7cff"), "brown", obj=self.object)

    def make_appear(self, start: int, stop: int):
        self.object.data.materials[0].diffuse_color[3] = 0.0
        self.object.data.materials[0].keyframe_insert(
            data_path="diffuse_color",
            frame=start,
        )

        self.object.data.materials[0].diffuse_color[3] = 1.0
        self.object.data.materials[0].keyframe_insert(
            data_path="diffuse_color", frame=stop
        )

    def make_disappear(self, start: int, stop: int):
        self.object.data.materials[0].diffuse_color[3] = 1.0
        self.object.data.materials[0].keyframe_insert(
            data_path="diffuse_color",
            frame=start,
        )

        self.object.data.materials[0].diffuse_color[3] = 0.0
        self.object.data.materials[0].keyframe_insert(
            data_path="diffuse_color", frame=stop
        )

        pass


def create_point_movement(start_position, end_position):
    return bpy.ops.mesh.primitive_cube_add(
        location=tupel(1.5, 0, 0),
    )


@dataclass
class VectorBlender:
    shaft_lenght: float
    head_length: float
    shaft_width: float
    head_widht: float

    def __post_init__(self):
        pass


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

    print(f"newScene: {new_context}")
    # bpy.data.scenes[len(bpy.data.scenes) - 1].name = "Main"

    cube_obstacle = CubeObstacle([3.0, 0, 0])
    cube_obstacle.make_disappear(10, 20)

    # Rotational Mesh
    rotational = RotationalMesh(12, 12)
    rotational.make_unfold(50, 100, 10)
    rotational.make_fold(150, 200, 10)

    bpy.context.scene.frame_end = 300

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
