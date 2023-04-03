import bpy
from math import *
from mathutils import Vector


def create_arrow(
    tubeRadius,
    arrowLength,
    headLength,
    tubeLength,
    tubeAcross,
    tubeAround,
    headAround,
    headRadius,
    arrows,
):

    verts = []
    edges = []
    faces = []
    # append tube verticies 0 thru (tubeAcross x tubeAround - 1)
    for i in range(tubeAcross):
        v1 = tubeLength * (i / (tubeAcross - 1))  # - length/2
        for j in range(tubeAround):
            angle = 2 * pi * j / tubeAround
            v2 = tubeRadius * cos(angle)
            v3 = tubeRadius * sin(angle)

            verts.append([v1, v2, v3])

            index1 = tubeAround * i + j
            index2 = tubeAround * i + j + 1
            if index2 == (i + 1) * tubeAround:
                index2 = i * tubeAround
            edge = [index1, index2]
            edges.append(edge)

    # append cone verts (tubeAcross x tubeAround) thru (tubeAcross x tubeAround + coneAround + 1) (exctra 1 for the tip)
    for i in range(headAround):
        angle = 2 * pi * i / headAround
        v1 = tubeLength
        v2 = headRadius * cos(angle)
        v3 = headRadius * sin(angle)
        verts.append([v1, v2, v3])

    verts.append([arrowLength, 0, 0])

    # append tube faces:
    # for i in range of number of faces (added 1 for the circle face)
    for i in range(tubeAround * (tubeAcross - 1) + 1):
        face = []
        if i == 0:
            # make face out of first slicesAround edges:
            for j in range(tubeAround):
                face.append(edges[j][0])
        else:
            face = [
                edges[i - 1][0],
                edges[i - 1][1],
                edges[i - 1 + tubeAround][1],
                edges[i - 1 + tubeAround][0],
            ]

        faces.append(face)

    # append face for back of cone:
    face = []
    for i in range(headAround):
        face.append(tubeAround * tubeAcross + i)
    faces.append(face)

    # append cone faces:
    for i in range(headAround):
        index1 = tubeAcross * tubeAround + i
        index2 = tubeAcross * tubeAround + i + 1
        index3 = tubeAcross * tubeAround + headAround

        if index2 == index3:
            index2 = index2 - headAround

        face = [index1, index2, index3]
        faces.append(face)

    mesh = bpy.data.meshes.new("Arrow")
    arrow = bpy.data.objects.new("Arrow", mesh)
    # col = bpy.data.collections.get("Collection")
    # col.objects.link(bendableArrow)
    arrows.objects.link(arrow)
    mesh.from_pydata(verts, [], faces)
