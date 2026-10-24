"""Muscle strings between bones of one collection, with live length measurement.

Landmarks are detected once on the Original bones and stored as vertex indices.
Paired Rigid meshes share vertex ordering, so the same indices place the Rigid
strings on the same anatomical points. Every string connects bones of a single
collection; Original and Rigid strings are matched only by muscle, side, and
string index when their lengths are compared.

Strings are poly curves whose control points are hooked to their bones, so they
stretch as bones move. Lengths are always recomputed from the attachment vertices
in world space, never from the curve.
"""

import csv
import json

import blf  # pyright: ignore[reportMissingImports]
import bpy  # pyright: ignore[reportMissingImports]
import gpu  # pyright: ignore[reportMissingImports]
import numpy as np
from bpy_extras import view3d_utils  # pyright: ignore[reportMissingImports]
from bpy_extras.io_utils import ExportHelper  # pyright: ignore[reportMissingImports]
from gpu_extras.batch import batch_for_shader  # pyright: ignore[reportMissingImports]
from mathutils import Matrix, Vector  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, world_vertices
from .correspondence import (
    _draw_label,
    _draw_triangles,
    _line_triangles,
    _millimeters_per_unit,
    _tag_view_redraw,
    _uniform_shader,
    _view_region_under_mouse,
)
from .muscle_atlas import GROUPS, MIDLINE, landmark_spec, muscle_strings, string_key


STRING_RADIUS_MM = 0.6
HOVER_PIXELS = 8.0
HOVER_COLOR = (1.0, 1.0, 1.0, 1.0)
HOVER_LINE_WIDTH = 4.0
BUILT_FROM_PROPERTY = "biomechanics_muscle_bones"
STRING_PROPERTY = "biomechanics_muscle_string"

DELTA_BINS = 21
DELTA_SHORTER_COLOR = (0.1, 0.35, 1.0)
DELTA_NEUTRAL_COLOR = (0.82, 0.82, 0.82)
DELTA_LONGER_COLOR = (1.0, 0.15, 0.1)
DELTA_UNPAIRED_COLOR = (0.25, 0.25, 0.25, 1.0)
DELTA_REFRESH_INTERVAL = 0.25

_hover_running = False
_hover_sample = None
_draw_handler = None
_delta_active = False
_delta_signature = None
# Lookups by name in bpy.data.objects are linear in the scene's object count, so bone
# objects, their attachment vertices, and parsed string data are cached.
_bone_cache = {}
_local_co_cache = {}
_info_cache = {}


def _mesh_by_base_name(collection):
    meshes = {}
    for obj in collection.objects:
        if obj.type == "MESH":
            meshes.setdefault(base_object_name(obj.name), []).append(obj)
    return meshes


def _single(meshes, name, collection):
    matches = meshes.get(name, [])
    if len(matches) != 1:
        raise ValueError(f"{collection.name} needs exactly one mesh named {name} (found {len(matches)})")
    return matches[0]


def _body_frame(meshes, collection):
    """Return unit (anterior, left, superior) axes from the skeleton's own geometry."""

    def centroid(name):
        return world_vertices(_single(meshes, name, collection)).mean(axis=0)

    superior = centroid("T01") - centroid("sacrum")
    superior /= np.linalg.norm(superior)
    left = centroid("femur L") - centroid("femur R")
    left -= left.dot(superior) * superior
    left /= np.linalg.norm(left)
    anterior = np.cross(left, superior)
    return anterior, left, superior


class _LandmarkResolver:
    def __init__(self, meshes, collection):
        self.meshes = meshes
        self.collection = collection
        self.anterior, self.left, self.superior = _body_frame(meshes, collection)
        self._vertices = {}
        self._indices = {}

    def _bone_coordinates(self, bone):
        if bone not in self._vertices:
            vertices = world_vertices(_single(self.meshes, bone, self.collection))
            self._vertices[bone] = (vertices @ self.anterior, vertices @ self.left, vertices @ self.superior)
        return self._vertices[bone]

    def vertex_index(self, bone, landmark, side):
        key = (bone, landmark, side)
        if key in self._indices:
            return self._indices[key]

        (weight_a, weight_l, weight_s), bands = landmark_spec(bone, landmark)
        anterior, left, superior = self._bone_coordinates(bone)
        lateral_sign = 0.0 if side == MIDLINE else (1.0 if side == "L" else -1.0)
        lateral = left * lateral_sign
        axes = {"A": anterior, "S": superior, "L": lateral}

        mask = np.ones(len(anterior), dtype=bool)
        for axis, (low, high) in bands.items():
            values = axes[axis]
            span = values.max() - values.min()
            normalized = (values - values.min()) / span if span > 0.0 else np.zeros_like(values)
            mask &= (normalized >= low) & (normalized <= high)
        if not mask.any():
            raise ValueError(f"Landmark {landmark} band is empty on {bone}")

        score = weight_a * anterior + weight_l * lateral + weight_s * superior
        score = np.where(mask, score, -np.inf)
        index = int(np.argmax(score))
        self._indices[key] = index
        return index


def _muscle_collection_name(bone_collection):
    return f"{bone_collection.name} muscles"


def muscle_collections(scene):
    """Return {bone collection name: muscle collection} for built string sets."""
    found = {}
    for collection in bpy.data.collections:
        bones = collection.get(BUILT_FROM_PROPERTY)
        if bones:
            found[bones] = collection
    return found


def _remove_muscle_collection(collection):
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            else:
                bpy.data.curves.remove(data)
    bpy.data.collections.remove(collection)


def _group_material(group):
    label, color = GROUPS[group]
    name = f"Biomechanics Muscle {label}"
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _create_string(collection, bone_collection, string, attachments, scene):
    """attachments: [(bone object, vertex index)] along the string."""
    side = "" if string["side"] == MIDLINE else f" {string['side']}"
    name = f"{string['label']}{side} {string['index'] + 1:02d} ({bone_collection.name})"
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = STRING_RADIUS_MM / _millimeters_per_unit(scene)
    curve.bevel_resolution = 1
    spline = curve.splines.new("POLY")
    spline.points.add(len(attachments) - 1)
    for point, (bone, index) in zip(spline.points, attachments):
        point.co = (*(bone.matrix_world @ bone.data.vertices[index].co), 1.0)

    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    material = _group_material(string["group"])
    curve.materials.append(material)
    obj.color = material.diffuse_color
    obj.matrix_world = Matrix.Identity(4)

    for point_index, (bone, _index) in enumerate(attachments):
        hook = obj.modifiers.new(f"Hook {bone.name}", "HOOK")
        hook.object = bone
        hook.vertex_indices_set([point_index])
        hook.matrix_inverse = bone.matrix_world.inverted()

    obj[STRING_PROPERTY] = json.dumps(
        {
            "muscle": string["muscle"],
            "label": string["label"],
            "group": string["group"],
            "side": string["side"],
            "index": string["index"],
            "count": string["count"],
            "name": string["name"],
            "attachments": [[bone.name, index] for bone, index in attachments],
        }
    )
    return obj


def build_muscle_strings(scene):
    """Rebuild strings on Original and Rigid. Returns (counts per collection, skipped messages)."""
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    if source is None or target is None:
        raise ValueError("Choose the Original and Rigid collections")
    if source == target:
        raise ValueError("Original and Rigid collections must be different")

    clear_caches()
    source_meshes = _mesh_by_base_name(source)
    resolver = _LandmarkResolver(source_meshes, source)
    strings = muscle_strings()

    resolved = []
    skipped = []
    for string in strings:
        try:
            refs = [(bone, resolver.vertex_index(bone, landmark, string["side"])) for bone, landmark in string["points"]]
        except (KeyError, ValueError) as error:
            skipped.append(f"{string['label']} {string['side']} {string['name']}: {error}")
            continue
        resolved.append((string, refs))

    existing = muscle_collections(scene)
    counts = {}
    for bone_collection in (source, target):
        meshes = source_meshes if bone_collection == source else _mesh_by_base_name(bone_collection)
        old = existing.get(bone_collection.name)
        if old is not None:
            _remove_muscle_collection(old)
        collection = bpy.data.collections.new(_muscle_collection_name(bone_collection))
        collection[BUILT_FROM_PROPERTY] = bone_collection.name
        scene.collection.children.link(collection)

        count = 0
        for string, refs in resolved:
            try:
                attachments = [(_single(meshes, bone, bone_collection), index) for bone, index in refs]
            except ValueError as error:
                skipped.append(str(error))
                continue
            if any(index >= len(bone.data.vertices) for bone, index in attachments):
                skipped.append(f"{bone_collection.name}: vertex order differs for {string['label']}")
                continue
            _create_string(collection, bone_collection, string, attachments, scene)
            count += 1
        counts[bone_collection.name] = count
    return counts, skipped


def parse_cached(raw):
    if not raw:
        return None
    info = _info_cache.get(raw)
    if info is None:
        info = _info_cache[raw] = json.loads(raw)
    return info


def string_info(obj):
    return parse_cached(obj.get(STRING_PROPERTY) if obj is not None else None)


def clear_caches():
    _bone_cache.clear()
    _local_co_cache.clear()
    _info_cache.clear()


def cached_object(name):
    obj = _bone_cache.get(name)
    if obj is not None:
        try:
            if obj.name == name:
                return obj
        except ReferenceError:
            pass
    obj = bpy.data.objects.get(name)
    if obj is None:
        _bone_cache.pop(name, None)
    else:
        _bone_cache[name] = obj
    return obj


def attachment_world(bone_name, index):
    """World position of a bone's attachment vertex, or None if it is missing."""
    bone = cached_object(bone_name)
    if bone is None or bone.type != "MESH":
        return None
    key = (bone_name, index)
    co = _local_co_cache.get(key)
    if co is None:
        if index >= len(bone.data.vertices):
            return None
        co = _local_co_cache[key] = bone.data.vertices[index].co.copy()
    return bone.matrix_world @ co


def _attachment_points(info):
    points = []
    for bone_name, index in info["attachments"]:
        point = attachment_world(bone_name, index)
        if point is None:
            return None
        points.append(point)
    return points


def string_length_mm(info, scene):
    points = _attachment_points(info)
    if points is None:
        return None
    length = sum((b - a).length for a, b in zip(points, points[1:]))
    return length * _millimeters_per_unit(scene)


def _strings_by_key(collection):
    strings = {}
    for obj in collection.objects:
        info = string_info(obj)
        if info is not None:
            strings[(info["muscle"], info["side"], info["index"])] = (obj, info)
    return strings


def length_rows(scene):
    """Return comparison rows for every built string, Original first."""
    collections = muscle_collections(scene)
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    source_strings = _strings_by_key(collections[source.name]) if source and source.name in collections else {}
    target_strings = _strings_by_key(collections[target.name]) if target and target.name in collections else {}
    rows = []
    for key in sorted(set(source_strings) | set(target_strings)):
        source_entry = source_strings.get(key)
        target_entry = target_strings.get(key)
        info = (source_entry or target_entry)[1]
        source_length = string_length_mm(source_entry[1], scene) if source_entry else None
        target_length = string_length_mm(target_entry[1], scene) if target_entry else None
        rows.append(
            {
                "muscle": info["label"],
                "side": info["side"],
                "string": info["index"] + 1,
                "attachment": info["name"],
                "bones": " -> ".join(bone for bone, _index in info["attachments"]),
                "source_mm": source_length,
                "target_mm": target_length,
            }
        )
    return rows


def _twin(obj, info, scene):
    """Return the matching string object in the other muscle collection."""
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
    return _strings_by_key(other).get(string_key(info), (None, None))[0]


def length_delta(source_length, target_length, unit):
    """Rigid minus Original, in percent of the Original length or in millimeters."""
    if source_length is None or target_length is None:
        return None
    delta = target_length - source_length
    if unit == "PERCENT":
        return 100.0 * delta / source_length if source_length > 0.0 else None
    return delta


def _delta_color(t):
    """Diverging ramp for t in [-1, 1]: blue shorter, gray unchanged, red longer."""
    end = DELTA_LONGER_COLOR if t > 0.0 else DELTA_SHORTER_COLOR
    weight = min(abs(t), 1.0)
    return tuple(n + (e - n) * weight for n, e in zip(DELTA_NEUTRAL_COLOR, end)) + (1.0,)


def _delta_bin_material(index):
    name = f"Biomechanics Muscle Delta {index:02d}"
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    _set_diffuse(material, _delta_color(2.0 * index / (DELTA_BINS - 1) - 1.0))
    return material


def _set_diffuse(material, color):
    # Writing a material color tags a depsgraph update even when unchanged.
    if any(abs(current - wanted) > 1e-4 for current, wanted in zip(material.diffuse_color, color)):
        material.diffuse_color = color


def delta_bin_materials():
    return [_delta_bin_material(index) for index in range(DELTA_BINS)]


def unpaired_material():
    material = bpy.data.materials.get("Biomechanics Muscle Delta Unpaired")
    if material is None:
        material = bpy.data.materials.new("Biomechanics Muscle Delta Unpaired")
    _set_diffuse(material, DELTA_UNPAIRED_COLOR)
    return material


def delta_bin(value, limit):
    t = max(-1.0, min(1.0, value / max(limit, 1e-6)))
    return round((t + 1.0) / 2.0 * (DELTA_BINS - 1))


def _set_string_color(obj, material):
    slot = obj.material_slots[0] if obj.material_slots else None
    if slot is None:
        return
    if slot.link != "OBJECT":
        slot.link = "OBJECT"
    if slot.material != material:
        slot.material = material
    if tuple(obj.color) != tuple(material.diffuse_color):
        obj.color = material.diffuse_color


def _restore_string_color(obj):
    slot = obj.material_slots[0] if obj.material_slots else None
    if slot is None or slot.link == "DATA":
        return
    slot.material = None
    slot.link = "DATA"
    if obj.data.materials:
        obj.color = obj.data.materials[0].diffuse_color


def apply_delta_colors(scene):
    """Color each Original/Rigid twin pair by its length change."""
    collections = muscle_collections(scene)
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    source_strings = _strings_by_key(collections[source.name]) if source and source.name in collections else {}
    target_strings = _strings_by_key(collections[target.name]) if target and target.name in collections else {}
    unit = scene.biomechanics_muscle_delta_unit
    limit = scene.biomechanics_muscle_delta_range
    materials = delta_bin_materials()
    unpaired = unpaired_material()

    for key in set(source_strings) | set(target_strings):
        source_entry = source_strings.get(key)
        target_entry = target_strings.get(key)
        delta = None
        if source_entry and target_entry:
            delta = length_delta(
                string_length_mm(source_entry[1], scene), string_length_mm(target_entry[1], scene), unit
            )
        material = unpaired if delta is None else materials[delta_bin(delta, limit)]
        for entry in (source_entry, target_entry):
            if entry is not None:
                _set_string_color(entry[0], material)

    from .muscle_sheets import apply_sheet_colors

    apply_sheet_colors(scene)


def restore_delta_colors():
    for collection in bpy.data.collections:
        if collection.get(BUILT_FROM_PROPERTY):
            for obj in collection.objects:
                if string_info(obj) is not None:
                    _restore_string_color(obj)

    from .muscle_sheets import restore_sheet_colors

    restore_sheet_colors()


def _bones_signature(scene):
    """World matrices of every Original and Rigid mesh; changes when any bone moves."""
    values = []
    for collection in (scene.biomechanics_source_collection, scene.biomechanics_target_collection):
        if collection is None:
            continue
        for obj in collection.objects:
            if obj.type == "MESH":
                values.append(obj.name)
                values.extend(value for row in obj.matrix_world for value in row)
    return tuple(values)


def _apply_delta_colors_now(scene):
    global _delta_signature
    apply_delta_colors(scene)
    _delta_signature = _bones_signature(scene)


def _refresh_delta_colors():
    if not _delta_active:
        return None
    try:
        scene = bpy.context.scene
        if _bones_signature(scene) != _delta_signature:
            _apply_delta_colors_now(scene)
    except (AttributeError, ReferenceError):
        pass
    return DELTA_REFRESH_INTERVAL


def enable_delta_colors(scene):
    global _delta_active
    _delta_active = True
    _apply_delta_colors_now(scene)
    if not bpy.app.timers.is_registered(_refresh_delta_colors):
        bpy.app.timers.register(_refresh_delta_colors, first_interval=DELTA_REFRESH_INTERVAL)
    _tag_view_redraw()


def disable_delta_colors():
    global _delta_active
    _delta_active = False
    if bpy.app.timers.is_registered(_refresh_delta_colors):
        bpy.app.timers.unregister(_refresh_delta_colors)
    restore_delta_colors()
    _tag_view_redraw()


def update_delta_scale(scene, _context):
    if _delta_active:
        _apply_delta_colors_now(scene)
        _tag_view_redraw()


def is_delta_active():
    return _delta_active


def _bone_collection_name(obj):
    collection = obj.users_collection[0]
    return collection.get(BUILT_FROM_PROPERTY, collection.name)


def _source_first(scene, obj, value, twin_value):
    """Order a hovered value and its twin's as (Original, Rigid)."""
    target = scene.biomechanics_target_collection
    if target is not None and _bone_collection_name(obj) == target.name:
        return twin_value, value
    return value, twin_value


def _change_label(scene):
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    return f"change {target.name if target else 'Rigid'} vs {source.name if source else 'Original'}"


def _segment_distance(point, start, end):
    segment = end - start
    length_squared = segment.length_squared
    if length_squared == 0.0:
        return (point - start).length
    t = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
    return (point - (start + t * segment)).length


def _layer_collection(layer_collection, collection):
    if layer_collection.collection == collection:
        return layer_collection
    for child in layer_collection.children:
        found = _layer_collection(child, collection)
        if found is not None:
            return found
    return None


def _collection_shown(view_layer, collection):
    # Object.visible_get() reports False for valid strings when called from a 3D-view
    # context in Blender 5.x, so visibility is checked explicitly.
    layer_collection = _layer_collection(view_layer.layer_collection, collection)
    return (
        layer_collection is not None
        and not layer_collection.exclude
        and not layer_collection.hide_viewport
        and not collection.hide_viewport
    )


def _string_under_mouse(context, region, region_data, mouse):
    mouse = Vector(mouse)
    view_layer = context.view_layer
    best = None
    best_distance = HOVER_PIXELS * context.preferences.system.ui_scale
    for collection in muscle_collections(context.scene).values():
        if not _collection_shown(view_layer, collection):
            continue
        for obj in collection.objects:
            info = string_info(obj)
            if info is None or obj.hide_viewport or obj.hide_get(view_layer=view_layer):
                continue
            points = _attachment_points(info)
            if points is None:
                continue
            projected = [view3d_utils.location_3d_to_region_2d(region, region_data, point) for point in points]
            for start, end in zip(projected, projected[1:]):
                if start is None or end is None:
                    continue
                distance = _segment_distance(mouse, start, end)
                if distance < best_distance:
                    best, best_distance = obj.name, distance
    return best


def _smooth_color_shader():
    try:
        return gpu.shader.from_builtin("SMOOTH_COLOR")
    except ValueError:
        return gpu.shader.from_builtin("2D_SMOOTH_COLOR")


def _sheet_under_mouse(context, region, region_data, mouse):
    """Return ("sheet", object name, face index) for a muscle sheet under the mouse."""
    from .muscle_sheets import sheet_info

    origin = view3d_utils.region_2d_to_origin_3d(region, region_data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region_data, mouse)
    hit, _location, _normal, face_index, hit_object, _matrix = context.scene.ray_cast(
        context.evaluated_depsgraph_get(), origin, direction
    )
    if not hit or hit_object is None:
        return None
    obj = getattr(hit_object, "original", hit_object)
    if sheet_info(obj) is None:
        return None
    return ("sheet", obj.name, face_index)


def _draw_ramp_legend(bottom, title, labels):
    context = bpy.context
    region = context.region
    ui_scale = context.preferences.system.ui_scale
    left = 20.0 * ui_scale
    bottom *= ui_scale
    width = min(220.0 * ui_scale, region.width - 2.0 * left)
    height = 10.0 * ui_scale
    if width <= 0.0:
        return

    steps = 16
    positions = []
    colors = []
    for index in range(steps):
        x0 = left + width * index / steps
        x1 = left + width * (index + 1) / steps
        c0 = _delta_color(2.0 * index / steps - 1.0)
        c1 = _delta_color(2.0 * (index + 1) / steps - 1.0)
        positions.extend(
            ((x0, bottom), (x1, bottom), (x1, bottom + height), (x0, bottom), (x1, bottom + height), (x0, bottom + height))
        )
        colors.extend((c0, c1, c1, c0, c1, c0))
    shader = _smooth_color_shader()
    gpu.state.blend_set("ALPHA")
    batch_for_shader(shader, "TRIS", {"pos": positions, "color": colors}).draw(shader)
    gpu.state.blend_set("NONE")

    font_id = 0
    blf.size(font_id, 12.0 * ui_scale)
    blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.9)
    blf.position(font_id, left, bottom + height + 6.0 * ui_scale, 0.0)
    blf.draw(font_id, title)
    for fraction, text in labels:
        text_width, _ = blf.dimensions(font_id, text)
        x = left + width * fraction - text_width * fraction
        blf.position(font_id, x, bottom - 16.0 * ui_scale, 0.0)
        blf.draw(font_id, text)
    blf.disable(font_id, blf.SHADOW)


def _draw_delta_legend():
    from .muscle_sheets import sheets_exist

    context = bpy.context
    if context.region is None:
        return
    scene = context.scene
    source = scene.biomechanics_source_collection
    target = scene.biomechanics_target_collection
    versus = f"{target.name if target else 'Rigid'} vs {source.name if source else 'Original'}"
    limit = scene.biomechanics_muscle_delta_range
    suffix = "%" if scene.biomechanics_muscle_delta_unit == "PERCENT" else " mm"
    # Sits above the distance heatmap legend, which uses the bottom-left corner.
    _draw_ramp_legend(
        90.0,
        f"Muscle length: {versus}",
        ((0.0, f"-{limit:g}{suffix} shorter"), (0.5, "0"), (1.0, f"+{limit:g}{suffix} longer")),
    )
    if sheets_exist(scene):
        sheet_limit = scene.biomechanics_muscle_sheet_range
        _draw_ramp_legend(
            150.0,
            f"Sheet area: {versus}",
            ((0.0, f"-{sheet_limit:g}% smaller"), (0.5, "0"), (1.0, f"+{sheet_limit:g}% larger")),
        )


def _draw_overlay():
    if _delta_active:
        _draw_delta_legend()
    if isinstance(_hover_sample, tuple):
        _draw_sheet_hover()
    elif _hover_sample is not None:
        _draw_hover()


def _draw_sheet_hover():
    from .muscle_sheets import area_change_percent, sheet_info, sheet_twin, triangle_areas_mm2

    context = bpy.context
    region = context.region
    region_data = context.region_data
    _kind, name, face_index = _hover_sample
    obj = bpy.data.objects.get(name)
    info = sheet_info(obj)
    if region is None or region_data is None or info is None or face_index >= len(info["faces"]):
        return
    if len(obj.users_collection) == 0:
        return
    scene = context.scene
    areas = triangle_areas_mm2(info, scene)
    points = _attachment_points({"attachments": [info["attachments"][i] for i in info["faces"][face_index]]})
    if areas is None or points is None:
        return
    projected = [view3d_utils.location_3d_to_region_2d(region, region_data, point) for point in points]
    if any(point is None for point in projected):
        return

    side = "" if info["side"] == MIDLINE else f" {info['side']}"
    area = float(areas[face_index])
    lines = [
        f"{info['label']}{side}  sheet triangle {face_index + 1}/{len(info['faces'])}",
        f"{_bone_collection_name(obj)}: {area:.1f} mm²",
    ]
    twin = sheet_twin(obj, info, scene)
    twin_info = sheet_info(twin)
    if twin_info is not None and twin_info["faces"] == info["faces"]:
        twin_areas = triangle_areas_mm2(twin_info, scene)
        if twin_areas is not None:
            twin_area = float(twin_areas[face_index])
            lines.append(f"{_bone_collection_name(twin)}: {twin_area:.1f} mm²")
            source_area, target_area = _source_first(scene, obj, area, twin_area)
            change = area_change_percent(np.array([source_area]), np.array([target_area]))[0]
            percent = "" if np.isnan(change) else f" ({change:+.1f}%)"
            lines.append(f"{_change_label(scene)}: {target_area - source_area:+.1f} mm²{percent}")

    ui_scale = context.preferences.system.ui_scale
    shader = _uniform_shader()
    gpu.state.blend_set("ALPHA")
    try:
        for start, end in zip(projected, projected[1:] + projected[:1]):
            _draw_triangles(shader, _line_triangles(start, end, HOVER_LINE_WIDTH * ui_scale), HOVER_COLOR)
        anchor = (projected[0] + projected[1] + projected[2]) / 3.0
        line_height = 16.0 * ui_scale
        for offset, text in enumerate(lines):
            _draw_label(anchor - Vector((0.0, offset * line_height)), text, HOVER_COLOR)
    finally:
        gpu.state.blend_set("NONE")


def _draw_hover():
    context = bpy.context
    region = context.region
    region_data = context.region_data
    obj = bpy.data.objects.get(_hover_sample)
    info = string_info(obj)
    if region is None or region_data is None or info is None:
        return
    points = _attachment_points(info)
    if points is None or len(obj.users_collection) == 0:
        return
    projected = [view3d_utils.location_3d_to_region_2d(region, region_data, point) for point in points]
    if any(point is None for point in projected):
        return

    scene = context.scene
    ui_scale = context.preferences.system.ui_scale
    length = string_length_mm(info, scene)
    side = "" if info["side"] == MIDLINE else f" {info['side']}"
    lines = [
        f"{info['label']}{side}  string {info['index'] + 1}/{info['count']} ({info['name']})",
        f"{_bone_collection_name(obj)}: {length:.2f} mm",
    ]
    twin = _twin(obj, info, scene)
    twin_info = string_info(twin)
    twin_length = string_length_mm(twin_info, scene) if twin_info else None
    if twin_length is not None:
        lines.append(f"{_bone_collection_name(twin)}: {twin_length:.2f} mm")
        source_length, target_length = _source_first(scene, obj, length, twin_length)
        delta = target_length - source_length
        percent = f" ({100.0 * delta / source_length:+.1f}%)" if source_length > 0.0 else ""
        lines.append(f"{_change_label(scene)}: {delta:+.2f} mm{percent}")

    shader = _uniform_shader()
    gpu.state.blend_set("ALPHA")
    try:
        for start, end in zip(projected, projected[1:]):
            _draw_triangles(shader, _line_triangles(start, end, HOVER_LINE_WIDTH * ui_scale), HOVER_COLOR)
        anchor = (projected[0] + projected[-1]) / 2.0
        line_height = 16.0 * ui_scale
        for offset, text in enumerate(lines):
            _draw_label(anchor - Vector((0.0, offset * line_height)), text, HOVER_COLOR)
    finally:
        gpu.state.blend_set("NONE")


def is_hover_active():
    return _hover_running


class BIOMECHANICS_OT_build_muscle_strings(bpy.types.Operator):
    """Attach one string per muscle attachment between bones of Original, and the same on Rigid"""

    bl_idname = "biomechanics.build_muscle_strings"
    bl_label = "Build Muscle Strings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .muscle_sheets import build_sheets, sheets_exist

        had_sheets = sheets_exist(context.scene)
        try:
            counts, skipped = build_muscle_strings(context.scene)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if had_sheets:
            build_sheets(context.scene)
        if _delta_active:
            _apply_delta_colors_now(context.scene)
        message = ", ".join(f"{name}: {count} strings" for name, count in counts.items())
        if skipped:
            message += f"; skipped {len(skipped)}: {skipped[0]}"
        self.report({"WARNING"} if skipped else {"INFO"}, message)
        return {"FINISHED"}


class BIOMECHANICS_OT_remove_muscle_strings(bpy.types.Operator):
    """Delete the generated muscle string collections"""

    bl_idname = "biomechanics.remove_muscle_strings"
    bl_label = "Remove Muscle Strings"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(muscle_collections(context.scene))

    def execute(self, context):
        global _hover_sample
        disable_delta_colors()
        for collection in muscle_collections(context.scene).values():
            _remove_muscle_collection(collection)
        _hover_sample = None
        _tag_view_redraw()
        return {"FINISHED"}


class BIOMECHANICS_OT_toggle_muscle_hover(bpy.types.Operator):
    """Toggle hover display of muscle string lengths and their Original/Rigid difference"""

    bl_idname = "biomechanics.toggle_muscle_hover"
    bl_label = "Hover Muscle Lengths"

    def invoke(self, context, _event):
        global _hover_running, _hover_sample
        if _hover_running:
            _hover_running = False
            _hover_sample = None
            _tag_view_redraw()
            return {"FINISHED"}

        _hover_running = True
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self.invoke(context, None)

    def modal(self, context, event):
        global _hover_sample
        if not _hover_running:
            return {"CANCELLED"}
        if event.type != "MOUSEMOVE":
            return {"PASS_THROUGH"}

        region, region_data = _view_region_under_mouse(context, event)
        previous = _hover_sample
        if region is None or region_data is None:
            _hover_sample = None
        else:
            mouse = (event.mouse_x - region.x, event.mouse_y - region.y)
            _hover_sample = _string_under_mouse(context, region, region_data, mouse)
            if _hover_sample is None:
                _hover_sample = _sheet_under_mouse(context, region, region_data, mouse)
        if _hover_sample != previous:
            _tag_view_redraw()
        return {"PASS_THROUGH"}


class BIOMECHANICS_OT_toggle_muscle_delta_colors(bpy.types.Operator):
    """Color each muscle string and its twin by length change: blue shorter, red longer"""

    bl_idname = "biomechanics.toggle_muscle_delta_colors"
    bl_label = "Length Change Colors"

    @classmethod
    def poll(cls, context):
        return _delta_active or bool(muscle_collections(context.scene))

    def execute(self, context):
        if _delta_active:
            disable_delta_colors()
        else:
            enable_delta_colors(context.scene)
        return {"FINISHED"}


class BIOMECHANICS_OT_export_muscle_lengths(bpy.types.Operator, ExportHelper):
    """Write every muscle string length on Original and Rigid, and their difference, to CSV"""

    bl_idname = "biomechanics.export_muscle_lengths"
    bl_label = "Export Muscle Lengths"
    filename_ext = ".csv"

    filter_glob: bpy.props.StringProperty(default="*.csv", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bool(muscle_collections(context.scene))

    def execute(self, context):
        scene = context.scene
        rows = length_rows(scene)
        source_name = scene.biomechanics_source_collection.name
        target_name = scene.biomechanics_target_collection.name
        with open(self.filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["muscle", "side", "string", "attachment", "bones", f"{source_name}_mm", f"{target_name}_mm", "delta_mm", "delta_percent"]
            )
            for row in rows:
                source_length, target_length = row["source_mm"], row["target_mm"]
                delta = target_length - source_length if source_length is not None and target_length is not None else None
                percent = 100.0 * delta / source_length if delta is not None and source_length else None
                writer.writerow(
                    [
                        row["muscle"],
                        row["side"],
                        row["string"],
                        row["attachment"],
                        row["bones"],
                        *(f"{value:.3f}" if value is not None else "" for value in (source_length, target_length, delta, percent)),
                    ]
                )
        self.report({"INFO"}, f"Wrote {len(rows)} muscle strings to {self.filepath}")
        return {"FINISHED"}


@bpy.app.handlers.persistent
def _reset_on_load(_dummy):
    global _hover_running, _hover_sample
    _hover_running = False
    _hover_sample = None
    # Delta coloring is session state; a loaded file starts with group colors.
    disable_delta_colors()
    clear_caches()


def _remove_load_handler():
    for handler in tuple(bpy.app.handlers.load_post):
        if getattr(handler, "__name__", "") == _reset_on_load.__name__ and getattr(
            handler, "__module__", ""
        ) == __name__:
            bpy.app.handlers.load_post.remove(handler)


def register_muscle_strings():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), "WINDOW", "POST_PIXEL")
    _remove_load_handler()
    bpy.app.handlers.load_post.append(_reset_on_load)


def unregister_muscle_strings():
    global _draw_handler, _hover_running, _hover_sample
    _hover_running = False
    _hover_sample = None
    disable_delta_colors()
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    _remove_load_handler()
