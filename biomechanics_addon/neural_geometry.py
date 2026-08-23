import bpy  # pyright: ignore[reportMissingImports]
from mathutils import Euler, Matrix, Vector  # pyright: ignore[reportMissingImports]
from mathutils.bvhtree import BVHTree  # pyright: ignore[reportMissingImports]

from .schemas import JOINT_RADIUS_MM, build_package_v2, geometry_fingerprint, validate_package_v2


def _matrix_values(matrix):
    return [value for row in matrix for value in row]


def _sample_vertices(vertices, maximum=750):
    if len(vertices) <= maximum:
        return vertices
    step = len(vertices) / maximum
    return [vertices[int(index * step)] for index in range(maximum)]


def _proxy_geometry(context, obj, maximum_faces):
    duplicate = bpy.data.objects.new(
        "Biomechanics Neural Proxy",
        obj.data.copy(),
    )
    context.scene.collection.objects.link(duplicate)
    try:
        polygon_count = len(duplicate.data.polygons)
        if polygon_count > maximum_faces:
            modifier = duplicate.modifiers.new("Biomechanics Neural Proxy", "DECIMATE")
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
    """Evaluated local-space geometry with modifiers applied, read-only."""
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        vertices = [vertex.co.copy() for vertex in mesh.vertices]
        polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
    finally:
        evaluated.to_mesh_clear()
    return vertices, polygons


def object_fingerprint(context, obj):
    """Content hash of evaluated geometry, independent of the object's transform."""
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
    rotation_center_world=None,
    pose_transform_version=2,
):
    """Build and validate a package v2 describing the current neutral pose."""
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
        rotation_center_world=rotation_center_world,
        pose_transform_version=pose_transform_version,
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
    rotation_center_world,
    pose_transform_version,
    baseline_overlap_pairs,
    minimum_clearance,
):
    """Evaluate one local 6-DOF pose for neural dataset labeling."""
    if pose_transform_version == 1:
        scale = neutral_matrix.to_scale()
        local_translation = Vector((
            tx / scale.x if scale.x else 0.0,
            ty / scale.y if scale.y else 0.0,
            tz / scale.z if scale.z else 0.0,
        ))
    else:
        local_translation = Vector((tx, ty, tz))
    local_rotation = Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()
    neutral_location, neutral_rotation, _neutral_scale = neutral_matrix.decompose()
    rotation_center = Vector(rotation_center_world)
    world_rotation = neutral_rotation.to_matrix().to_4x4() @ local_rotation @ neutral_rotation.inverted().to_matrix().to_4x4()
    candidate_matrix = (
        Matrix.Translation(neutral_rotation @ local_translation)
        @ Matrix.Translation(rotation_center)
        @ world_rotation
        @ Matrix.Translation(-rotation_center)
        @ neutral_matrix
    )
    moving_world_vertices, moving_bvh = _world_bvh(moving_vertices, moving_polygons, candidate_matrix)
    overlap_pairs = len(fixed_bvh.overlap(moving_bvh))
    surface_overlap = overlap_pairs > 0
    exceeds_baseline_overlap = overlap_pairs > baseline_overlap_pairs
    clearance = 0.0
    if not surface_overlap:
        clearance = _clearance(fixed_world_vertices, fixed_bvh, moving_world_vertices, moving_bvh)
    elif exceeds_baseline_overlap:
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


# Public aliases used by the background-Blender neural dataset generator.
proxy_geometry = _proxy_geometry
full_mesh_geometry = _full_mesh_geometry
world_bvh = _world_bvh
evaluate_local_pose = _evaluate_local_pose
