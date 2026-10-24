"""Triangle sheets between neighboring strings of one muscle.

A sheet joins each pair of consecutive strings (by atlas index) of a muscle and side
with a triangle strip. Its vertices are the strings' attachment vertices, hooked to
their bones, so the sheet stretches with the strings. Original and Rigid sheets are
built from twin strings in the same order, so polygon i of one sheet is polygon i of
its twin, and each triangle can be colored by its area change.
"""

import json

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np
from mathutils import Matrix  # pyright: ignore[reportMissingImports]

from .muscle_atlas import MIDLINE
from .muscle_strings import (
    _group_material,
    attachment_world,
    cached_object,
    parse_cached,
    _strings_by_key,
    _tag_view_redraw,
    delta_bin,
    delta_bin_materials,
    muscle_collections,
    unpaired_material,
)
from .correspondence import _millimeters_per_unit


SHEET_PROPERTY = "biomechanics_muscle_sheet"
# Material slots: 0 is the muscle group color, then the delta bins, then unpaired.
GROUP_SLOT = 0
BIN_SLOT_OFFSET = 1


def sheet_info(obj):
    return parse_cached(obj.get(SHEET_PROPERTY) if obj is not None else None)


def _sheets_by_key(collection):
    sheets = {}
    for obj in collection.objects:
        info = sheet_info(obj)
        if info is not None:
            sheets[(info["muscle"], info["side"])] = (obj, info)
    return sheets


def sheets_exist(scene):
    return any(_sheets_by_key(collection) for collection in muscle_collections(scene).values())


def _sheet_topology(strings):
    """strings: [info] sorted by index. Returns (attachments, faces) or None."""
    attachments = []
    lookup = {}

    def vertex(attachment):
        key = tuple(attachment)
        if key not in lookup:
            lookup[key] = len(attachments)
            attachments.append(list(attachment))
        return lookup[key]

    faces = []
    for first, second in zip(strings, strings[1:]):
        a = [vertex(point) for point in first["attachments"]]
        b = [vertex(point) for point in second["attachments"]]
        if len(a) != len(b):
            continue
        for j in range(len(a) - 1):
            for triangle in ((a[j], a[j + 1], b[j + 1]), (a[j], b[j + 1], b[j])):
                if len(set(triangle)) == 3:
                    faces.append(list(triangle))
    if not faces:
        return None
    return attachments, faces


def _create_sheet(collection, bone_collection_name, first_info, attachments, faces):
    bones = [cached_object(bone) for bone, _index in attachments]
    if any(bone is None or bone.type != "MESH" for bone in bones):
        return None
    positions = [bone.matrix_world @ bone.data.vertices[index].co for bone, (_name, index) in zip(bones, attachments)]

    side = "" if first_info["side"] == MIDLINE else f" {first_info['side']}"
    name = f"{first_info['label']}{side} sheet ({bone_collection_name})"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(position) for position in positions], [], [tuple(face) for face in faces])
    mesh.update()
    mesh.materials.append(_group_material(first_info["group"]))
    for material in delta_bin_materials():
        mesh.materials.append(material)
    mesh.materials.append(unpaired_material())

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.matrix_world = Matrix.Identity(4)
    obj.color = mesh.materials[GROUP_SLOT].diffuse_color

    by_bone = {}
    for vertex_index, bone in enumerate(bones):
        by_bone.setdefault(bone.name, (bone, []))[1].append(vertex_index)
    for bone, vertex_indices in by_bone.values():
        hook = obj.modifiers.new(f"Hook {bone.name}", "HOOK")
        hook.object = bone
        hook.vertex_indices_set(vertex_indices)
        hook.matrix_inverse = bone.matrix_world.inverted()

    obj[SHEET_PROPERTY] = json.dumps(
        {
            "muscle": first_info["muscle"],
            "label": first_info["label"],
            "group": first_info["group"],
            "side": first_info["side"],
            "attachments": attachments,
            "faces": faces,
        }
    )
    return obj


def remove_sheets(scene):
    for collection in muscle_collections(scene).values():
        for obj, _info in _sheets_by_key(collection).values():
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def build_sheets(scene):
    """Rebuild sheets in every muscle collection. Returns {bone collection: sheet count}."""
    remove_sheets(scene)
    counts = {}
    for bones, collection in muscle_collections(scene).items():
        by_muscle = {}
        for (muscle, side, _index), (_obj, info) in _strings_by_key(collection).items():
            by_muscle.setdefault((muscle, side), []).append(info)
        count = 0
        for strings in by_muscle.values():
            strings.sort(key=lambda info: info["index"])
            topology = _sheet_topology(strings)
            if topology is None:
                continue
            if _create_sheet(collection, bones, strings[0], *topology) is not None:
                count += 1
        counts[bones] = count
    return counts


def triangle_areas_mm2(info, scene):
    """Return per-face world areas in square millimeters, or None if a bone is missing."""
    points = []
    for bone_name, index in info["attachments"]:
        point = attachment_world(bone_name, index)
        if point is None:
            return None
        points.append(tuple(point))
    points = np.array(points, dtype=np.float64)
    faces = np.array(info["faces"], dtype=np.int64)
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return areas * _millimeters_per_unit(scene) ** 2


def area_change_percent(source_areas, target_areas):
    """Per-face Rigid/Original area change in percent; NaN where the Original face is degenerate."""
    with np.errstate(divide="ignore", invalid="ignore"):
        change = 100.0 * (target_areas - source_areas) / source_areas
    change[~np.isfinite(change) | (source_areas <= 1e-9)] = np.nan
    return change


def _set_face_slots(obj, slots):
    mesh = obj.data
    current = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", current)
    if not np.array_equal(current, slots):
        mesh.polygons.foreach_set("material_index", slots)
        mesh.update()


def apply_sheet_colors(scene):
    collections = muscle_collections(scene)
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    source_sheets = _sheets_by_key(collections[source.name]) if source and source.name in collections else {}
    target_sheets = _sheets_by_key(collections[target.name]) if target and target.name in collections else {}
    limit = scene.biomechanics_muscle_sheet_range
    bin_materials = delta_bin_materials()
    unpaired_color = unpaired_material().diffuse_color
    unpaired_slot = BIN_SLOT_OFFSET + len(bin_materials)

    for key in set(source_sheets) | set(target_sheets):
        source_entry = source_sheets.get(key)
        target_entry = target_sheets.get(key)
        entries = [entry for entry in (source_entry, target_entry) if entry is not None]
        slots = None
        if source_entry and target_entry and source_entry[1]["faces"] == target_entry[1]["faces"]:
            source_areas = triangle_areas_mm2(source_entry[1], scene)
            target_areas = triangle_areas_mm2(target_entry[1], scene)
            if source_areas is not None and target_areas is not None:
                change = area_change_percent(source_areas, target_areas)
                slots = np.array(
                    [unpaired_slot if np.isnan(value) else BIN_SLOT_OFFSET + delta_bin(value, limit) for value in change],
                    dtype=np.int32,
                )
        # Object color mode (used by transparent Comparison Display) shows one color per
        # object, so each sheet also carries the color of its total area change.
        overall = unpaired_color
        if slots is not None:
            total_change = area_change_percent(np.array([source_areas.sum()]), np.array([target_areas.sum()]))[0]
            if not np.isnan(total_change):
                overall = bin_materials[delta_bin(total_change, limit)].diffuse_color
        for obj, info in entries:
            face_slots = slots if slots is not None else np.full(len(info["faces"]), unpaired_slot, dtype=np.int32)
            _set_face_slots(obj, face_slots)
            if tuple(obj.color) != tuple(overall):
                obj.color = overall


def restore_sheet_colors():
    for obj in bpy.data.objects:
        if sheet_info(obj) is not None:
            _set_face_slots(obj, np.full(len(obj.data.polygons), GROUP_SLOT, dtype=np.int32))
            obj.color = obj.data.materials[GROUP_SLOT].diffuse_color


def sheet_twin(obj, info, scene):
    collections = muscle_collections(scene)
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    if source is None or target is None:
        return None
    owner = next((name for name, collection in collections.items() if collection.objects.get(obj.name) is obj), None)
    other_name = target.name if owner == source.name else source.name if owner == target.name else None
    other = collections.get(other_name)
    if other is None:
        return None
    return _sheets_by_key(other).get((info["muscle"], info["side"]), (None, None))[0]


class BIOMECHANICS_OT_toggle_muscle_sheets(bpy.types.Operator):
    """Fill triangles between neighboring strings of each muscle, on Original and Rigid"""

    bl_idname = "biomechanics.toggle_muscle_sheets"
    bl_label = "Muscle Sheets"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(muscle_collections(context.scene))

    def execute(self, context):
        from .muscle_strings import apply_delta_colors, is_delta_active

        scene = context.scene
        if sheets_exist(scene):
            remove_sheets(scene)
            _tag_view_redraw()
            return {"FINISHED"}
        counts = build_sheets(scene)
        if is_delta_active():
            apply_delta_colors(scene)
        _tag_view_redraw()
        self.report({"INFO"}, ", ".join(f"{name}: {count} sheets" for name, count in counts.items()))
        return {"FINISHED"}
