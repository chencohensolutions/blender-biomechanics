"""Point correspondence between paired Original and Rigid meshes.

Paired meshes share topology and vertex ordering, so a surface point on one mesh
maps to the same polygon-interpolated point on its pair. Hover inspection and the
maximum-distance search both use current world-space positions.
"""

import math

import blf  # pyright: ignore[reportMissingImports]
import bpy  # pyright: ignore[reportMissingImports]
import gpu  # pyright: ignore[reportMissingImports]
import numpy as np
from bpy_extras import view3d_utils  # pyright: ignore[reportMissingImports]
from gpu_extras.batch import batch_for_shader  # pyright: ignore[reportMissingImports]
from mathutils import Vector  # pyright: ignore[reportMissingImports]
from mathutils.interpolate import poly_3d_calc  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, world_vertices


HOVER_COLOR = (1.0, 0.8, 0.1, 1.0)
PAIR_COLOR = (0.1, 0.8, 1.0, 1.0)
LINE_COLOR = (1.0, 1.0, 1.0, 0.85)
MAX_COLOR = (1.0, 0.15, 0.35, 1.0)
POINT_RADIUS = 6.0
LINE_WIDTH = 2.0
MAX_RESULTS_SHOWN = 5

_hover_running = False
_hover_sample = None
_max_results = []
_draw_handler = None


def _millimeters_per_unit(scene):
    return (scene.unit_settings.scale_length or 1.0) * 1000.0


def _collection_contains(collection, obj):
    return collection is not None and collection.objects.get(obj.name) is obj


def paired_mesh(scene, obj):
    """Return (pair, error) for a mesh in Original or Rigid."""
    source_collection = scene.biomechanics_source_collection
    target_collection = scene.biomechanics_target_collection
    if obj is None or obj.type != "MESH":
        return None, "Not a mesh"
    if _collection_contains(source_collection, obj):
        other_collection = target_collection
    elif _collection_contains(target_collection, obj):
        other_collection = source_collection
    else:
        return None, "Not in Original or Rigid"
    if other_collection is None or other_collection == source_collection == target_collection:
        return None, "Choose different Original and Rigid collections"

    pair_name = base_object_name(obj.name)
    matches = [
        candidate
        for candidate in other_collection.objects
        if candidate.type == "MESH" and base_object_name(candidate.name) == pair_name
    ]
    if not matches:
        return None, f"No pair for {obj.name}"
    if len(matches) > 1:
        return None, f"Multiple pairs for {obj.name}"
    pair = matches[0]
    if len(pair.data.vertices) != len(obj.data.vertices):
        return None, f"{obj.name} and {pair.name} have different vertex counts"
    return pair, None


def _corresponding_point(obj, pair, polygon_index, world_location):
    """Map a world-space hit on obj's polygon to the same surface point on pair."""
    mesh = obj.data
    if polygon_index < 0 or polygon_index >= len(mesh.polygons):
        return None
    vertex_indices = mesh.polygons[polygon_index].vertices
    local_location = obj.matrix_world.inverted() @ world_location
    weights = poly_3d_calc([mesh.vertices[index].co for index in vertex_indices], local_location)
    pair_vertices = pair.data.vertices
    local_pair = Vector((0.0, 0.0, 0.0))
    for weight, index in zip(weights, vertex_indices):
        local_pair += weight * pair_vertices[index].co
    return pair.matrix_world @ local_pair


def _sample_under_mouse(context, region, region_data, mouse):
    scene = context.scene
    origin = view3d_utils.region_2d_to_origin_3d(region, region_data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region_data, mouse)
    hit, location, _normal, polygon_index, hit_object, _matrix = scene.ray_cast(
        context.evaluated_depsgraph_get(), origin, direction
    )
    if not hit or hit_object is None:
        return None
    obj = getattr(hit_object, "original", hit_object)
    pair, error = paired_mesh(scene, obj)
    if pair is None:
        return {"object": obj.name, "location": location.copy(), "error": error}
    pair_location = _corresponding_point(obj, pair, polygon_index, location)
    if pair_location is None:
        return {"object": obj.name, "location": location.copy(), "error": "Modified topology"}
    return {
        "object": obj.name,
        "pair": pair.name,
        "location": location.copy(),
        "pair_location": pair_location,
        "distance_mm": (pair_location - location).length * _millimeters_per_unit(scene),
    }


def _view_region_under_mouse(context, event):
    screen = context.window.screen if context.window is not None else None
    if screen is None:
        return None, None
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            if (
                region.x <= event.mouse_x < region.x + region.width
                and region.y <= event.mouse_y < region.y + region.height
            ):
                return region, region.data
    return None, None


def _tag_view_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def is_hover_active():
    return _hover_running


def max_distance_results():
    return _max_results


def _uniform_shader():
    try:
        return gpu.shader.from_builtin("UNIFORM_COLOR")
    except ValueError:
        return gpu.shader.from_builtin("2D_UNIFORM_COLOR")


def _draw_triangles(shader, triangles, color):
    if not triangles:
        return
    batch = batch_for_shader(shader, "TRIS", {"pos": triangles})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _circle_triangles(center, radius, segments=16):
    triangles = []
    for index in range(segments):
        start = 2.0 * math.pi * index / segments
        end = 2.0 * math.pi * (index + 1) / segments
        triangles.extend(
            (
                (center.x, center.y),
                (center.x + radius * math.cos(start), center.y + radius * math.sin(start)),
                (center.x + radius * math.cos(end), center.y + radius * math.sin(end)),
            )
        )
    return triangles


def _line_triangles(start, end, width):
    direction = end - start
    if direction.length < 1e-6:
        return []
    offset = Vector((-direction.y, direction.x)).normalized() * (width / 2.0)
    a, b, c, d = start + offset, start - offset, end - offset, end + offset
    return [tuple(a), tuple(b), tuple(c), tuple(a), tuple(c), tuple(d)]


def _draw_label(position, text, color):
    font_id = 0
    ui_scale = bpy.context.preferences.system.ui_scale
    blf.size(font_id, 13.0 * ui_scale)
    blf.color(font_id, *color)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.9)
    blf.position(font_id, position.x + 10.0 * ui_scale, position.y + 10.0 * ui_scale, 0.0)
    blf.draw(font_id, text)
    blf.disable(font_id, blf.SHADOW)


def _draw_pair(shader, region, region_data, first, second, first_color, second_color, label, label_color):
    first_2d = view3d_utils.location_3d_to_region_2d(region, region_data, first)
    second_2d = view3d_utils.location_3d_to_region_2d(region, region_data, second)
    ui_scale = bpy.context.preferences.system.ui_scale
    if first_2d is not None and second_2d is not None:
        _draw_triangles(shader, _line_triangles(first_2d, second_2d, LINE_WIDTH * ui_scale), LINE_COLOR)
    if second_2d is not None:
        _draw_triangles(shader, _circle_triangles(second_2d, POINT_RADIUS * ui_scale), second_color)
    if first_2d is not None:
        _draw_triangles(shader, _circle_triangles(first_2d, POINT_RADIUS * ui_scale), first_color)
        _draw_label(first_2d, label, label_color)


def _max_marker_points():
    if not _max_results:
        return None
    result = _max_results[0]
    original = bpy.data.objects.get(result["original"])
    rigid = bpy.data.objects.get(result["rigid"])
    index = result["vertex_index"]
    if (
        original is None
        or rigid is None
        or index >= len(original.data.vertices)
        or index >= len(rigid.data.vertices)
    ):
        return None
    return (
        original.matrix_world @ original.data.vertices[index].co,
        rigid.matrix_world @ rigid.data.vertices[index].co,
    )


def _draw_overlay():
    if _hover_sample is None and not _max_results:
        return
    region = bpy.context.region
    region_data = bpy.context.region_data
    if region is None or region_data is None:
        return

    shader = _uniform_shader()
    gpu.state.blend_set("ALPHA")
    try:
        marker = _max_marker_points()
        if marker is not None:
            result = _max_results[0]
            _draw_pair(
                shader,
                region,
                region_data,
                marker[0],
                marker[1],
                MAX_COLOR,
                MAX_COLOR,
                f"Max {result['distance_mm']:.3f} mm ({result['original']})",
                MAX_COLOR,
            )

        sample = _hover_sample
        if sample is not None:
            if "pair_location" in sample:
                _draw_pair(
                    shader,
                    region,
                    region_data,
                    sample["location"],
                    sample["pair_location"],
                    HOVER_COLOR,
                    PAIR_COLOR,
                    f"{sample['object']} -> {sample['pair']}: {sample['distance_mm']:.3f} mm",
                    HOVER_COLOR,
                )
            else:
                position = view3d_utils.location_3d_to_region_2d(region, region_data, sample["location"])
                if position is not None:
                    _draw_label(position, f"{sample['object']}: {sample['error']}", (1.0, 1.0, 1.0, 0.8))
    finally:
        gpu.state.blend_set("NONE")


class BIOMECHANICS_OT_toggle_correspondence_hover(bpy.types.Operator):
    """Toggle hover inspection of the corresponding point on the paired Original/Rigid mesh"""

    bl_idname = "biomechanics.toggle_correspondence_hover"
    bl_label = "Hover Correspondence"

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
            _hover_sample = _sample_under_mouse(context, region, region_data, mouse)
        if _hover_sample is not None or previous is not None:
            _tag_view_redraw()
        return {"PASS_THROUGH"}


class BIOMECHANICS_OT_find_max_correspondence_distance(bpy.types.Operator):
    """Find the corresponding vertex pair with the largest distance across all Original/Rigid meshes"""

    bl_idname = "biomechanics.find_max_correspondence_distance"
    bl_label = "Find Max Distance Point"

    def execute(self, context):
        global _max_results
        scene = context.scene
        source_collection = scene.biomechanics_source_collection
        if source_collection is None or scene.biomechanics_target_collection is None:
            self.report({"ERROR"}, "Choose the original and rigid collections")
            return {"CANCELLED"}
        if source_collection == scene.biomechanics_target_collection:
            self.report({"ERROR"}, "Original and rigid collections must be different")
            return {"CANCELLED"}

        millimeters_per_unit = _millimeters_per_unit(scene)
        results = []
        skipped = []
        for original in source_collection.objects:
            if original.type != "MESH":
                continue
            rigid, error = paired_mesh(scene, original)
            if rigid is None or len(original.data.vertices) == 0:
                skipped.append(error or f"{original.name} has no vertices")
                continue
            distances = np.linalg.norm(world_vertices(original) - world_vertices(rigid), axis=1)
            vertex_index = int(np.argmax(distances))
            results.append(
                {
                    "original": original.name,
                    "rigid": rigid.name,
                    "vertex_index": vertex_index,
                    "distance_mm": float(distances[vertex_index]) * millimeters_per_unit,
                }
            )

        if not results:
            _max_results = []
            _tag_view_redraw()
            self.report({"ERROR"}, "No valid Original/Rigid mesh pairs found")
            return {"CANCELLED"}

        results.sort(key=lambda item: item["distance_mm"], reverse=True)
        _max_results = results
        _tag_view_redraw()
        best = results[0]
        message = (
            f"Max distance {best['distance_mm']:.3f} mm: {best['original']} -> {best['rigid']} "
            f"vertex {best['vertex_index']} ({len(results)} pairs)"
        )
        if skipped:
            message += f"; skipped {len(skipped)}: {skipped[0]}"
        self.report({"WARNING"} if skipped else {"INFO"}, message)
        return {"FINISHED"}


class BIOMECHANICS_OT_clear_correspondence_max(bpy.types.Operator):
    """Clear the maximum-distance marker and results"""

    bl_idname = "biomechanics.clear_correspondence_max"
    bl_label = "Clear Max Distance"

    @classmethod
    def poll(cls, _context):
        return bool(_max_results)

    def execute(self, _context):
        global _max_results
        _max_results = []
        _tag_view_redraw()
        return {"FINISHED"}


@bpy.app.handlers.persistent
def _reset_on_load(_dummy):
    # Loading a file drops running modal handlers and invalidates stored results.
    global _hover_running, _hover_sample, _max_results
    _hover_running = False
    _hover_sample = None
    _max_results = []


def register_correspondence():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), "WINDOW", "POST_PIXEL")
    for handler in tuple(bpy.app.handlers.load_post):
        if getattr(handler, "__name__", "") == _reset_on_load.__name__ and getattr(
            handler, "__module__", ""
        ) == __name__:
            bpy.app.handlers.load_post.remove(handler)
    bpy.app.handlers.load_post.append(_reset_on_load)


def unregister_correspondence():
    global _draw_handler, _hover_running, _hover_sample, _max_results
    _hover_running = False
    _hover_sample = None
    _max_results = []
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    for handler in tuple(bpy.app.handlers.load_post):
        if getattr(handler, "__name__", "") == _reset_on_load.__name__ and getattr(
            handler, "__module__", ""
        ) == __name__:
            bpy.app.handlers.load_post.remove(handler)
