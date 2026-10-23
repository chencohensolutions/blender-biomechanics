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

_hover_running = False
_hover_sample = None
_draw_handler = None


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


def string_info(obj):
    raw = obj.get(STRING_PROPERTY) if obj is not None else None
    return json.loads(raw) if raw else None


def _attachment_points(info):
    points = []
    for bone_name, index in info["attachments"]:
        bone = bpy.data.objects.get(bone_name)
        if bone is None or bone.type != "MESH" or index >= len(bone.data.vertices):
            return None
        points.append(bone.matrix_world @ bone.data.vertices[index].co)
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


def _bone_collection_name(obj):
    collection = obj.users_collection[0]
    return collection.get(BUILT_FROM_PROPERTY, collection.name)


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


def _draw_overlay():
    if _hover_sample is None:
        return
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
        delta = twin_length - length
        percent = f" ({100.0 * delta / length:+.1f}%)" if length > 0.0 else ""
        lines.append(f"{_bone_collection_name(twin)}: {twin_length:.2f} mm  delta {delta:+.2f} mm{percent}")

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
        try:
            counts, skipped = build_muscle_strings(context.scene)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
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
        if _hover_sample != previous:
            _tag_view_redraw()
        return {"PASS_THROUGH"}


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
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    _remove_load_handler()
