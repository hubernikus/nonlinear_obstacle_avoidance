import bpy


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
    obj.show_transparent = True
    # Create a material
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    # if use_transparency:
    # mat.blend_method = "BLEND"

    # mat.transparency_method = "Z_TRANSPARENCY"
    # mat.use_transparency = True

    # mat.use_transparency = True
    # mat.transparency_method = "Z_TRANSPARENCY"

    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = color

    # principled.inputs["Base Color"].use_transparency = True
    # mat.transparency_method = "Z_TRANSPARENCY"

    obj.data.materials.append(mat)
