import bpy

import shutil

import numpy as np
import math

from pathlib import Path

Vertice = tuple[float]


class RotationalMesh:
    def __init__(self, n_lattitude, n_longitude, it_max: int = 100) -> None:
        self.radius = 1.0

        self.it_max = it_max
        self.n_lattitude = n_lattitude
        self.n_longitude = n_longitude

        self.vertices = [
            (0, 0.0, 0.0) for _ in range(self.n_lattitude * self.n_longitude + 1)
        ]
        self.edges = []
        self.faces = []

        self.vector_init = [
            (0, 0.0, 0.0) for _ in range(self.n_lattitude * self.n_longitude + 1)
        ]
        self.vector_final = [
            (0, 0.0, 0.0) for _ in range(self.n_lattitude * self.n_longitude + 1)
        ]

        self.create_vertices()
        self.create_edges_and_faces()

        self.mesh = bpy.data.meshes.new("circle_space_mesh")
        self.mesh.from_pydata(self.vertices, self.edges, self.faces)
        self.mesh.update()

        # Make object from mesh
        self.object = bpy.data.objects.new("circle_space_object", self.mesh)

        # Make collection
        self.collection = bpy.data.collections.new("circle_space_collection")
        bpy.context.scene.collection.children.link(self.collection)
        self.collection.objects.link(self.object)

        print("Circle space done.")

    def get_index(self, it_long: int, it_latt: int) -> int:
        if it_latt < 0:
            return 0
        return 1 + it_long + it_latt * self.n_lattitude

    def update_positions(self, ii):
        frac = ii / self.it_max
        for lo in range(self.n_longitude):
            position_parent = np.array(self.mesh[self.get_index(-1, -1)])

            for la in range(self.n_lattitude):
                index = self.get_index(lo, la)
                vector = (1 - frac) * self.vector_init[
                    index
                ] + frac * self.vector_final[index]

                position_parent = position_parent + vector
                self.mesh[index] = tupple(position_parent)

    def evaluate_vector_final(self):
        dxy = math.pi / self.n_latt
        delta_long = 2 * math.pi / self.n_longitude

        for lo in range(self.n_longitude):
            vect = (dxy * math.cos(lo * delta_long), math.sin(lo * delta_long), 0)
            for la in range(self.n_lattitude):
                self.vector_final[self.get_index(lo, la)] = vect

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

        self.vertices[self.get_index(-1, -1)] = (0.0, 0.0, 1.0)

        for lo in range(self.n_longitude):
            longitude = (1 + lo) * delta_long
            sin_ = math.sin(longitude)
            zz = math.cos(longitude)

            for la in range(self.n_lattitude):
                lattitude = la * delta_latt
                xx = math.sin(lattitude) * sin_
                yy = math.cos(lattitude) * sin_
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
    rotational = RotationalMesh(6, 6)

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
