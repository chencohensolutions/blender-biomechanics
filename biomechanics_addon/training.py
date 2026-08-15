import itertools
import json
import math

import bpy
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

from .schemas import JOINT_RADIUS_MM, build_package_v2, geometry_fingerprint, validate_package_v2


TRAINING_TEXT_PREFIX = "Biomechanics Training"


def planned_sample_count(
    translation_steps,
    rotation_steps,
    translation_limit=None,
    rotation_limit=None,
):
    translation_count = 1 if translation_limit == 0.0 else translation_steps
    rotation_count = 1 if rotation_limit == 0.0 else rotation_steps
    return translation_count ** 3 * rotation_count ** 3


def _sample_values(limit, steps):
    if steps <= 1 or limit == 0.0:
        return (0.0,)
    return tuple(-limit + (2.0 * limit * index / (steps - 1)) for index in range(steps))


def _matrix_values(matrix):
    return [value for row in matrix for value in row]


def _sample_vertices(vertices, maximum=750):
    if len(vertices) <= maximum:
        return vertices
    step = len(vertices) / maximum
    return [vertices[int(index * step)] for index in range(maximum)]


def _proxy_geometry(context, obj, maximum_faces):
    duplicate = bpy.data.objects.new(
        "Biomechanics Training Proxy",
        obj.data.copy(),
    )
    context.scene.collection.objects.link(duplicate)
    try:
        polygon_count = len(duplicate.data.polygons)
        if polygon_count > maximum_faces:
            modifier = duplicate.modifiers.new("Biomechanics Training Proxy", "DECIMATE")
            modifier.ratio = maximum_faces / polygon_count
            modifier.use_collapse_triangulate = True
        context.view_layer.update()
        evaluated = duplicate.evaluated_get(context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            vertices = [vertex.co.copy() for vertex in mesh.vertices]
            polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
        finally:
            evaluated.to_mesh_clear()
    finally:
        mesh_data = duplicate.data
        bpy.data.objects.remove(duplicate, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)
    return vertices, polygons


def _world_bvh(vertices, polygons, matrix):
    world_vertices = [matrix @ vertex for vertex in vertices]
    return world_vertices, BVHTree.FromPolygons(world_vertices, polygons, all_triangles=False)


def _full_mesh_geometry(context, obj):
    """Evaluated local-space geometry (modifiers applied), read-only: no duplicate object."""
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        # Vector (not tuple) to match _proxy_geometry: both feed into _world_bvh's matrix @ vertex.
        vertices = [vertex.co.copy() for vertex in mesh.vertices]
        polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
    finally:
        evaluated.to_mesh_clear()
    return vertices, polygons


def object_fingerprint(context, obj):
    """Content hash of an object's current evaluated geometry, independent of its transform."""
    vertices, polygons = _full_mesh_geometry(context, obj)
    return geometry_fingerprint(vertices, polygons)


def build_geometry_package(
    context,
    fixed_object,
    moving_object,
    collection_name,
    translation_limit_mm,
    rotation_limit_degrees,
    seed,
    proxy_max_faces,
    use_full_mesh_for_holdout=True,
    joint_radius_mm=JOINT_RADIUS_MM,
):
    """Build a schema-validated package v2 dict describing the current neutral pose."""
    scale_length = context.scene.unit_settings.scale_length or 1.0
    package = build_package_v2(
        collection_name=collection_name,
        fixed_object_name=fixed_object.name,
        moving_object_name=moving_object.name,
        neutral_fixed_matrix=_matrix_values(fixed_object.matrix_world),
        neutral_moving_matrix=_matrix_values(moving_object.matrix_world),
        fixed_fingerprint=object_fingerprint(context, fixed_object),
        moving_fingerprint=object_fingerprint(context, moving_object),
        scale_length=scale_length,
        translation_limit_mm=translation_limit_mm,
        rotation_limit_degrees=rotation_limit_degrees,
        seed=seed,
        proxy_max_faces=proxy_max_faces,
        use_full_mesh_for_holdout=use_full_mesh_for_holdout,
        joint_radius_mm=joint_radius_mm,
    )
    return validate_package_v2(package)


def _evaluate_local_pose(
    fixed_bvh,
    fixed_world_vertices,
    moving_vertices,
    moving_polygons,
    neutral_matrix,
    tx,
    ty,
    tz,
    rx,
    ry,
    rz,
    baseline_overlap_pairs,
    minimum_clearance,
):
    """Evaluate one local 6-DOF pose; shared by the legacy grid trainer and future samplers.

    tx/ty/tz are a desired real-world-magnitude translation along the moving object's local axes.
    Since `neutral_matrix @ local_pose` applies the translation before neutral_matrix's own
    scale, it must be pre-divided by that scale here or it would be shrunk (e.g. 1000x smaller
    for a real anatomical mesh with a legitimate uniform 0.001 world scale) before appearing in
    world space. Rotation is unaffected by uniform/near-uniform scale, so it needs no compensation.
    """
    scale = neutral_matrix.to_scale()
    local_translation = Vector((
        tx / scale.x if scale.x else 0.0,
        ty / scale.y if scale.y else 0.0,
        tz / scale.z if scale.z else 0.0,
    ))
    local_pose = Matrix.Translation(local_translation) @ Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()
    candidate_matrix = neutral_matrix @ local_pose
    moving_world_vertices, moving_bvh = _world_bvh(moving_vertices, moving_polygons, candidate_matrix)
    overlap_pairs = len(fixed_bvh.overlap(moving_bvh))
    surface_overlap = overlap_pairs > 0
    exceeds_baseline_overlap = overlap_pairs > baseline_overlap_pairs
    clearance = 0.0
    if not surface_overlap:
        clearance = _clearance(fixed_world_vertices, fixed_bvh, moving_world_vertices, moving_bvh)
    elif exceeds_baseline_overlap:
        # Baseline-tolerated overlap keeps clearance=0.0 (still a valid contact state), but overlap
        # beyond baseline is a real violation of varying severity - report signed penetration depth
        # instead of flattening every over-baseline overlap to the same zero value.
        clearance = -_clearance(fixed_world_vertices, fixed_bvh, moving_world_vertices, moving_bvh)
    valid = not exceeds_baseline_overlap and (surface_overlap or clearance >= minimum_clearance)
    return {
        "surface_overlap": surface_overlap,
        "overlap_pairs": overlap_pairs,
        "exceeds_baseline_overlap": exceeds_baseline_overlap,
        "clearance": clearance,
        "valid": valid,
    }


def _clearance(first_vertices, first_bvh, second_vertices, second_bvh):
    first_to_second = min(
        second_bvh.find_nearest(vertex)[3] for vertex in _sample_vertices(first_vertices)
    )
    second_to_first = min(
        first_bvh.find_nearest(vertex)[3] for vertex in _sample_vertices(second_vertices)
    )
    return min(first_to_second, second_to_first)


def train_two_objects(
    context,
    fixed_object,
    moving_object,
    translation_limit_mm,
    rotation_limit_degrees,
    translation_steps,
    rotation_steps,
    minimum_clearance_mm,
    maximum_proxy_faces,
    maximum_samples=10000,
    progress_callback=None,
    collection_name=None,
    write_text=True,
):
    scene = context.scene
    scale_length = scene.unit_settings.scale_length or 1.0
    millimeters_per_blender_unit = scale_length * 1000.0
    translation_limit = translation_limit_mm / millimeters_per_blender_unit
    minimum_clearance = minimum_clearance_mm / millimeters_per_blender_unit

    total_samples = planned_sample_count(
        translation_steps,
        rotation_steps,
        translation_limit,
        rotation_limit_degrees,
    )
    if total_samples > maximum_samples:
        raise ValueError(
            f"Planned sample count {total_samples:,} exceeds the hard limit "
            f"of {maximum_samples:,}"
        )

    fixed_vertices, fixed_polygons = _proxy_geometry(context, fixed_object, maximum_proxy_faces)
    moving_vertices, moving_polygons = _proxy_geometry(context, moving_object, maximum_proxy_faces)
    fixed_world_vertices, fixed_bvh = _world_bvh(
        fixed_vertices,
        fixed_polygons,
        fixed_object.matrix_world,
    )
    neutral_moving_vertices, neutral_moving_bvh = _world_bvh(
        moving_vertices,
        moving_polygons,
        moving_object.matrix_world,
    )
    baseline_overlap_pairs = len(fixed_bvh.overlap(neutral_moving_bvh))

    translations = _sample_values(translation_limit, translation_steps)
    rotations = _sample_values(math.radians(rotation_limit_degrees), rotation_steps)
    parameters = itertools.product(translations, translations, translations, rotations, rotations, rotations)
    neutral_matrix = moving_object.matrix_world.copy()
    samples = []
    valid_count = 0

    context.window_manager.progress_begin(0, total_samples)
    try:
        for index, (tx, ty, tz, rx, ry, rz) in enumerate(parameters):
            evaluation = _evaluate_local_pose(
                fixed_bvh,
                fixed_world_vertices,
                moving_vertices,
                moving_polygons,
                neutral_matrix,
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
                baseline_overlap_pairs,
                minimum_clearance,
            )
            valid_count += int(evaluation["valid"])
            samples.append(
                {
                    "translation_mm": [
                        tx * millimeters_per_blender_unit,
                        ty * millimeters_per_blender_unit,
                        tz * millimeters_per_blender_unit,
                    ],
                    "rotation_degrees": [math.degrees(rx), math.degrees(ry), math.degrees(rz)],
                    "surface_overlap": evaluation["surface_overlap"],
                    "overlap_pairs": evaluation["overlap_pairs"],
                    "exceeds_baseline_overlap": evaluation["exceeds_baseline_overlap"],
                    "clearance_mm": evaluation["clearance"] * millimeters_per_blender_unit,
                    "valid": evaluation["valid"],
                }
            )
            context.window_manager.progress_update(index + 1)
            if progress_callback is not None:
                progress_callback(index + 1, total_samples, valid_count)
    finally:
        context.window_manager.progress_end()

    result = {
        "format_version": 1,
        "collection": collection_name or scene.biomechanics_target_collection.name,
        "fixed_object": fixed_object.name,
        "moving_object": moving_object.name,
        "neutral_fixed_matrix": _matrix_values(fixed_object.matrix_world),
        "neutral_moving_matrix": _matrix_values(neutral_matrix),
        "settings": {
            "translation_limit_mm": translation_limit_mm,
            "rotation_limit_degrees": rotation_limit_degrees,
            "translation_steps": translation_steps,
            "rotation_steps": rotation_steps,
            "minimum_clearance_mm": minimum_clearance_mm,
            "maximum_proxy_faces": maximum_proxy_faces,
            "maximum_samples": maximum_samples,
        },
        "summary": {
            "total_samples": total_samples,
            "valid_samples": valid_count,
            "invalid_samples": total_samples - valid_count,
            "baseline_overlap_pairs": baseline_overlap_pairs,
        },
        "samples": samples,
    }

    text = None
    if write_text:
        text_name = f"{TRAINING_TEXT_PREFIX} - {fixed_object.name} - {moving_object.name}.json"
        text = bpy.data.texts.get(text_name) or bpy.data.texts.new(text_name)
        text.clear()
        text.write(json.dumps(result, indent=2))
    return text, result


# Public aliases for reuse by background dataset generation tooling (tools/generate_neural_dataset.py).
proxy_geometry = _proxy_geometry
full_mesh_geometry = _full_mesh_geometry
world_bvh = _world_bvh
evaluate_local_pose = _evaluate_local_pose
