import bpy

import shutil

import numpy as np
import math

from pathlib import Path

from nonlinear_avoidance.vector_rotation import directional_vector_addition

Vertice = tuple[float]


def is_even(value: int) -> bool:
    return not (value % 2)


def is_odd(value: int) -> bool:
    return value % 2


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

        self.create_time_series()

        # Make object from mesh
        self.object = bpy.data.objects.new("circle_space_object", self.mesh)

        # Make collection
        self.collection = bpy.data.collections.new("circle_space_collection")
        bpy.context.scene.collection.children.link(self.collection)
        self.collection.objects.link(self.object)

        # Create animation
        # self.object.position
        print("Circle space done.")

    def move_object(self):
        self.object.keyframe_insert(data_path="location", frame=1)

        self.object.location = (3.0, 0.0, 0.0)
        self.object.keyframe_insert(data_path="location", frame=10)

        self.object.location = (3.0, 0.0, 3.0)
        self.object.keyframe_insert(data_path="location", frame=20)

    def create_time_series(self):
        delta_frame = 5
        for ii in range(self.it_max + 1):
            self.update_vertices(ii)

            for jj, vert in enumerate(self.mesh.vertices):
                vert.co = self.vertices[jj]
                vert.keyframe_insert("co", frame=ii * delta_frame)

    def get_index(self, it_long: int, it_latt: int) -> int:
        if it_latt < 0:
            return 0
        return 1 + it_long + it_latt * self.n_lattitude

    def update_vertices(self, ii: int) -> None:
        frac_weight = ii / self.it_max
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


def create_new_mesh():
    # make mesh
    vertices = [(0, 0, 0)]
    edges = []
    faces = []
    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()

    # make object from mesh
    new_object = bpy.data.objects.new("new_object", new_mesh)

    # make collection
    new_collection = bpy.data.collections.new("new_collection")
    bpy.context.scene.collection.children.link(new_collection)

    # add object to scene collection
    new_collection.objects.link(self.object)


def print_meshes(filepath: Path):
    # load all meshes
    with bpy.data.libraries.load(str(filepath)) as (data_from, data_to):
        data_to.meshes = data_from.meshes

        # for mesh in data_to.meshes:
        #     if mesh is not None:
        #         breakpoint()
        #         print(mesh)

    bpy.data.objects["Cube"].select_set(True)
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
    # bpy.context.scene.name = "Main"
    new_context = bpy.context.scene

    print("newScene?: %s" % (new_context))
    # bpy.data.scenes[len(bpy.data.scenes) - 1].name = "Main"

    # create_new_mesh()
    # rotational = RotationalMesh(32, 32)
    rotational = RotationalMesh(12, 12)

    bpy.context.scene.frame_end = 200

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


# bpy.ops.object.data_instance_add(
#     # session_uuid=1042,
#     type="MESH",
#     align="WORLD",
#     location=(0.0, 0.0, 0.0),
#     scale=(1, 1, 1),
#     drop_x=710,
#     drop_y=641,
# )


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
