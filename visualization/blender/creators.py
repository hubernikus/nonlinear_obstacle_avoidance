import bpy


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
