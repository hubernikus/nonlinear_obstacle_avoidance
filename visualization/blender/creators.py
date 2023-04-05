import bpy

from pathlib import Path


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


def delete_all_meshes(filepath: Path):
    with bpy.data.libraries.load(str(filepath)) as (data_from, data_to):
        data_to.meshes = data_from.meshes

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
