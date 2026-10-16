"""Per-vertex correspondence-distance heatmap on paired Original and Rigid meshes.

The heatmap is a temporary point-domain color attribute shown through solid-viewport
Attribute color mode. Turning it off removes the attribute and restores each mesh's
previous active/render color attribute and each viewport's previous color mode.
"""

import blf  # pyright: ignore[reportMissingImports]
import bpy  # pyright: ignore[reportMissingImports]
import gpu  # pyright: ignore[reportMissingImports]
import numpy as np
from gpu_extras.batch import batch_for_shader  # pyright: ignore[reportMissingImports]

from .alignment import local_vertices, world_vertices
from .correspondence import paired_mesh
from .properties import _update_transparency_overlay, _view_3d_spaces


HEATMAP_ATTRIBUTE = "biomechanics_correspondence_distance"
PREVIOUS_ACTIVE_COLOR_KEY = "_biomechanics_previous_active_color"
PREVIOUS_RENDER_COLOR_KEY = "_biomechanics_previous_render_color"
REFRESH_INTERVAL = 0.1
GLOBAL_MAX_TOLERANCE = 0.1
MATRIX_TOLERANCE = 1e-9
RAMP_LUT_SIZE = 256

# Near (blue) to far (red), in sRGB.
RAMP_POSITIONS = np.array((0.0, 0.25, 0.5, 0.75, 1.0))
RAMP_COLORS = np.array(
    (
        (0.05, 0.15, 0.75),
        (0.0, 0.65, 0.95),
        (0.15, 0.85, 0.25),
        (1.0, 0.85, 0.0),
        (0.9, 0.05, 0.05),
    )
)

_heatmap_active = False
_pair_distances = {}
_local_vertices = {}
_object_matrices = {}
_global_max_mm = None
_dirty_object_names = set()
_previous_viewport_color_types = {}
_draw_handler = None


def is_heatmap_active():
    return _heatmap_active


def _build_ramp_lut():
    samples = np.linspace(0.0, 1.0, RAMP_LUT_SIZE)
    srgb = np.column_stack(
        [np.interp(samples, RAMP_POSITIONS, RAMP_COLORS[:, channel]) for channel in range(3)]
    )
    # FLOAT_COLOR attributes are scene-linear.
    return np.column_stack((srgb**2.2, np.ones(RAMP_LUT_SIZE))).astype(np.float32)


RAMP_LUT = _build_ramp_lut()


def _ramp_colors(values):
    indices = (np.clip(values, 0.0, 1.0) * (RAMP_LUT_SIZE - 1) + 0.5).astype(np.intp)
    return RAMP_LUT[indices]


def _matrix_values(obj):
    return np.array(obj.matrix_world, dtype=np.float64)


def _cached_world_vertices(obj):
    coordinates = _local_vertices.get(obj.name)
    if coordinates is None or len(coordinates) != len(obj.data.vertices):
        coordinates = local_vertices(obj)
        _local_vertices[obj.name] = coordinates
    _object_matrices[obj.name] = _matrix_values(obj)
    return world_vertices(obj, coordinates)


def _color_attribute_name(attributes, index):
    if 0 <= index < len(attributes):
        return attributes[index].name
    return ""


def _write_colors(mesh, colors):
    attributes = mesh.color_attributes
    attribute = attributes.get(HEATMAP_ATTRIBUTE)
    if attribute is None:
        if PREVIOUS_ACTIVE_COLOR_KEY not in mesh:
            active = attributes.active_color
            mesh[PREVIOUS_ACTIVE_COLOR_KEY] = active.name if active is not None else ""
            mesh[PREVIOUS_RENDER_COLOR_KEY] = _color_attribute_name(attributes, attributes.render_color_index)
        attribute = attributes.new(HEATMAP_ATTRIBUTE, "FLOAT_COLOR", "POINT")
    attributes.active_color = attribute
    attribute.data.foreach_set("color", colors.ravel())
    mesh.update_tag()


def _restore_meshes():
    for mesh in bpy.data.meshes:
        attributes = mesh.color_attributes
        attribute = attributes.get(HEATMAP_ATTRIBUTE)
        if attribute is not None:
            attributes.remove(attribute)
        if PREVIOUS_ACTIVE_COLOR_KEY not in mesh:
            continue
        names = [item.name for item in attributes]
        previous_active = mesh[PREVIOUS_ACTIVE_COLOR_KEY]
        previous_render = mesh[PREVIOUS_RENDER_COLOR_KEY]
        if previous_active in names:
            attributes.active_color = attributes[previous_active]
        if previous_render in names:
            attributes.render_color_index = names.index(previous_render)
        del mesh[PREVIOUS_ACTIVE_COLOR_KEY]
        del mesh[PREVIOUS_RENDER_COLOR_KEY]
        mesh.update_tag()


def _apply_viewport_color_type():
    for space in _view_3d_spaces():
        if space.shading.type != "SOLID" or space.as_pointer() in _previous_viewport_color_types:
            continue
        _previous_viewport_color_types[space.as_pointer()] = space.shading.color_type
        space.shading.color_type = "VERTEX"


def _restore_viewport_color_types():
    for space in _view_3d_spaces():
        previous = _previous_viewport_color_types.get(space.as_pointer())
        if previous is not None and space.shading.color_type == "VERTEX":
            space.shading.color_type = previous
    _previous_viewport_color_types.clear()


def _scene_pairs(scene):
    source_collection = scene.biomechanics_source_collection
    if source_collection is None:
        return []
    pairs = []
    for original in source_collection.objects:
        if original.type != "MESH" or len(original.data.vertices) == 0:
            continue
        rigid, _error = paired_mesh(scene, original)
        if rigid is not None:
            pairs.append((original, rigid))
    return pairs


def refresh_heatmap(scene, changed_object_names=None):
    """Recompute distances for changed pairs (all when None) and rewrite their colors."""
    global _global_max_mm
    if not _heatmap_active:
        return
    millimeters_per_unit = (scene.unit_settings.scale_length or 1.0) * 1000.0
    pairs = _scene_pairs(scene)
    keys = set()
    rewrite = []
    for original, rigid in pairs:
        key = (original.name, rigid.name)
        keys.add(key)
        if (
            changed_object_names is None
            or key not in _pair_distances
            or original.name in changed_object_names
            or rigid.name in changed_object_names
        ):
            _pair_distances[key] = (
                np.linalg.norm(_cached_world_vertices(original) - _cached_world_vertices(rigid), axis=1)
                * millimeters_per_unit
            )
            rewrite.append((original, rigid))
    for key in tuple(_pair_distances):
        if key not in keys:
            del _pair_distances[key]

    per_bone = scene.biomechanics_heatmap_scale == "PER_BONE"
    global_max = max((float(distances.max()) for distances in _pair_distances.values()), default=0.0)
    if (
        not per_bone
        and _global_max_mm is not None
        and abs(global_max - _global_max_mm) > GLOBAL_MAX_TOLERANCE * max(_global_max_mm, 1e-12)
    ):
        rewrite = pairs
    if changed_object_names is None or rewrite is pairs or _global_max_mm is None:
        _global_max_mm = global_max

    for original, rigid in rewrite:
        distances = _pair_distances[(original.name, rigid.name)]
        scale = float(distances.max()) if per_bone else _global_max_mm
        values = distances / scale if scale > 1e-12 else np.zeros_like(distances)
        colors = _ramp_colors(values)
        _write_colors(original.data, colors)
        if rigid.data is not original.data:
            _write_colors(rigid.data, colors)


def _tag_view_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _refresh_dirty():
    names = set(_dirty_object_names)
    _dirty_object_names.clear()
    if _heatmap_active and names:
        refresh_heatmap(bpy.context.scene, names)
        _tag_view_redraw()
    return None


def _on_depsgraph_update(_scene, depsgraph):
    if not _heatmap_active:
        return
    for update in depsgraph.updates:
        if not update.is_updated_transform or not isinstance(update.id, bpy.types.Object):
            continue
        obj = update.id.original
        previous = _object_matrices.get(obj.name)
        # Bone-parented or animated objects report transform updates on every
        # evaluation, including the one caused by writing heatmap colors.
        if previous is not None and not np.allclose(
            previous, _matrix_values(obj), rtol=0.0, atol=MATRIX_TOLERANCE
        ):
            _dirty_object_names.add(obj.name)
    if _dirty_object_names and not bpy.app.timers.is_registered(_refresh_dirty):
        bpy.app.timers.register(_refresh_dirty, first_interval=REFRESH_INTERVAL)


def _remove_depsgraph_handler():
    for handler in tuple(bpy.app.handlers.depsgraph_update_post):
        if getattr(handler, "__name__", "") == _on_depsgraph_update.__name__ and getattr(
            handler, "__module__", ""
        ) == __name__:
            bpy.app.handlers.depsgraph_update_post.remove(handler)
    if bpy.app.timers.is_registered(_refresh_dirty):
        bpy.app.timers.unregister(_refresh_dirty)


def enable_heatmap(scene):
    global _heatmap_active, _global_max_mm
    _heatmap_active = True
    _pair_distances.clear()
    _local_vertices.clear()
    _object_matrices.clear()
    _global_max_mm = None
    refresh_heatmap(scene)
    # Switch an active transparent comparison to wireframe before taking the viewport.
    _update_transparency_overlay(scene, None)
    _apply_viewport_color_type()
    _remove_depsgraph_handler()
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    return len(_pair_distances)


def disable_heatmap():
    global _heatmap_active, _global_max_mm
    was_active = _heatmap_active
    _heatmap_active = False
    _remove_depsgraph_handler()
    _dirty_object_names.clear()
    _pair_distances.clear()
    _local_vertices.clear()
    _object_matrices.clear()
    _global_max_mm = None
    _restore_meshes()
    _restore_viewport_color_types()
    if was_active:
        _update_transparency_overlay(bpy.context.scene, None)


def update_heatmap_scale(scene, _context):
    if _heatmap_active:
        refresh_heatmap(scene)
        _tag_view_redraw()


def _smooth_color_shader():
    try:
        return gpu.shader.from_builtin("SMOOTH_COLOR")
    except ValueError:
        return gpu.shader.from_builtin("2D_SMOOTH_COLOR")


def _draw_legend():
    if not _heatmap_active:
        return
    region = bpy.context.region
    if region is None:
        return
    ui_scale = bpy.context.preferences.system.ui_scale
    left = 20.0 * ui_scale
    bottom = 30.0 * ui_scale
    width = min(220.0 * ui_scale, region.width - 2.0 * left)
    height = 10.0 * ui_scale
    if width <= 0.0:
        return

    positions = []
    colors = []
    for index in range(len(RAMP_POSITIONS) - 1):
        x0 = left + width * RAMP_POSITIONS[index]
        x1 = left + width * RAMP_POSITIONS[index + 1]
        c0 = (*RAMP_COLORS[index], 1.0)
        c1 = (*RAMP_COLORS[index + 1], 1.0)
        positions.extend(
            ((x0, bottom), (x1, bottom), (x1, bottom + height), (x0, bottom), (x1, bottom + height), (x0, bottom + height))
        )
        colors.extend((c0, c1, c1, c0, c1, c0))
    shader = _smooth_color_shader()
    batch_for_shader(shader, "TRIS", {"pos": positions, "color": colors}).draw(shader)

    scene = bpy.context.scene
    if scene.biomechanics_heatmap_scale == "PER_BONE":
        far_label = "each bone's max"
    else:
        far_label = f"{_global_max_mm or 0.0:.3f} mm"
    font_id = 0
    blf.size(font_id, 12.0 * ui_scale)
    blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.9)
    blf.position(font_id, left, bottom + height + 6.0 * ui_scale, 0.0)
    blf.draw(font_id, "Correspondence distance")
    blf.position(font_id, left, bottom - 16.0 * ui_scale, 0.0)
    blf.draw(font_id, "0 mm")
    far_width, _ = blf.dimensions(font_id, far_label)
    blf.position(font_id, left + width - far_width, bottom - 16.0 * ui_scale, 0.0)
    blf.draw(font_id, far_label)
    blf.disable(font_id, blf.SHADOW)


class BIOMECHANICS_OT_toggle_distance_heatmap(bpy.types.Operator):
    """Color Original and Rigid meshes by distance to their corresponding points (blue near, red far)"""

    bl_idname = "biomechanics.toggle_distance_heatmap"
    bl_label = "Distance Heatmap"

    def execute(self, context):
        if _heatmap_active:
            disable_heatmap()
            _tag_view_redraw()
            return {"FINISHED"}

        scene = context.scene
        if scene.biomechanics_source_collection is None or scene.biomechanics_target_collection is None:
            self.report({"ERROR"}, "Choose the original and rigid collections")
            return {"CANCELLED"}
        pair_count = enable_heatmap(scene)
        if pair_count == 0:
            disable_heatmap()
            self.report({"ERROR"}, "No valid Original/Rigid mesh pairs found")
            return {"CANCELLED"}
        _tag_view_redraw()
        return {"FINISHED"}


@bpy.app.handlers.persistent
def _reset_on_load(_dummy):
    # A saved file may still carry the temporary attribute; remove it on load.
    global _heatmap_active, _global_max_mm
    _heatmap_active = False
    _remove_depsgraph_handler()
    _dirty_object_names.clear()
    _pair_distances.clear()
    _local_vertices.clear()
    _object_matrices.clear()
    _previous_viewport_color_types.clear()
    _global_max_mm = None
    _restore_meshes()


def _remove_load_handler():
    for handler in tuple(bpy.app.handlers.load_post):
        if getattr(handler, "__name__", "") == _reset_on_load.__name__ and getattr(
            handler, "__module__", ""
        ) == __name__:
            bpy.app.handlers.load_post.remove(handler)


def register_heatmap():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(_draw_legend, (), "WINDOW", "POST_PIXEL")
    _remove_load_handler()
    bpy.app.handlers.load_post.append(_reset_on_load)


def unregister_heatmap():
    global _draw_handler
    disable_heatmap()
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    _remove_load_handler()
