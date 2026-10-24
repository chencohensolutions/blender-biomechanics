"""Signed distance fields and surface samples for Rigid bone meshes.

Fields are computed in a mesh's local frame from its own vertices and polygons
(not the evaluated mesh), so one field serves every object sharing that geometry
and every pose. Computed fields are written to `<cache dir>/<fingerprint>.npz`
and kept in memory for the session.
"""

import hashlib
import os
import time

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np
from mathutils import Vector  # pyright: ignore[reportMissingImports]
from mathutils.bvhtree import BVHTree  # pyright: ignore[reportMissingImports]

from .alignment import local_vertices
from .distance_field import (
    DEFAULT_PADDING_MM,
    DEFAULT_VOXEL_MM,
    DistanceField,
    grid_points,
    grid_spec,
    sign_distances,
)

DEFAULT_CACHE_DIR = "//biomechanics_cache"
SURFACE_SAMPLE_COUNT = 1200
SAMPLE_SEED = 7

_fields = {}
_samples = {}
_fingerprints = {}


def _millimeters_per_unit(scene):
    return (scene.unit_settings.scale_length or 1.0) * 1000.0


def mesh_fingerprint(obj):
    """Fast content hash of a mesh's local vertices and polygon count."""
    mesh = obj.data
    key = mesh.as_pointer()
    cached = _fingerprints.get(key)
    if cached is not None and cached[0] == (len(mesh.vertices), len(mesh.polygons)):
        return cached[1]
    coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coordinates)
    hasher = hashlib.sha256()
    hasher.update(len(mesh.vertices).to_bytes(8, "little"))
    hasher.update(len(mesh.polygons).to_bytes(8, "little"))
    hasher.update(coordinates.tobytes())
    digest = hasher.hexdigest()[:24]
    _fingerprints[key] = ((len(mesh.vertices), len(mesh.polygons)), digest)
    return digest


def cache_directory(scene):
    path = getattr(scene, "biomechanics_field_cache_dir", "") or DEFAULT_CACHE_DIR
    if path.startswith("//") and not bpy.data.filepath:
        return os.path.join(bpy.app.tempdir, "biomechanics_cache")
    return bpy.path.abspath(path)


def field_path(scene, obj, voxel_mm):
    return os.path.join(cache_directory(scene), f"{mesh_fingerprint(obj)}_{voxel_mm:g}mm.npz")


def _polygon_indices(mesh):
    polygons = mesh.polygons
    sizes = np.empty(len(polygons), dtype=np.int32)
    polygons.foreach_get("loop_total", sizes)
    starts = np.empty(len(polygons), dtype=np.int32)
    polygons.foreach_get("loop_start", starts)
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    return [tuple(int(index) for index in loops[start : start + size]) for start, size in zip(starts, sizes)]


def compute_field(obj, scene, voxel_mm=DEFAULT_VOXEL_MM, padding_mm=DEFAULT_PADDING_MM, report=None):
    """Compute the signed distance field of obj's mesh in its local frame."""
    mesh = obj.data
    units = _millimeters_per_unit(scene)
    vertices = local_vertices(obj)
    polygons = _polygon_indices(mesh)
    tree = BVHTree.FromPolygons([Vector(v) for v in vertices], polygons, all_triangles=False)

    origin, voxel, shape = grid_spec(vertices.min(axis=0), vertices.max(axis=0), voxel_mm / units, padding_mm / units)
    points = grid_points(origin, voxel, shape)
    unsigned = np.empty(len(points), dtype=np.float64)
    normal_dot = np.empty(len(points), dtype=np.float64)

    window_manager = bpy.context.window_manager
    window_manager.progress_begin(0, 100)
    started = time.perf_counter()
    try:
        chunk = max(1, len(points) // 100)
        for start in range(0, len(points), chunk):
            for index in range(start, min(start + chunk, len(points))):
                point = Vector(points[index])
                location, normal, _face, distance = tree.find_nearest(point)
                unsigned[index] = distance
                normal_dot[index] = (point - location).dot(normal)
            window_manager.progress_update(int(100 * start / len(points)))
    finally:
        window_manager.progress_end()

    values = sign_distances(unsigned.reshape(shape), normal_dot.reshape(shape), voxel)
    field = DistanceField(origin, voxel, values)
    if report is not None:
        report(
            f"{obj.name}: {shape[0]}x{shape[1]}x{shape[2]} cells at {voxel * units:.2f} mm "
            f"in {time.perf_counter() - started:.1f} s"
        )
    return field


def field_for_object(obj, scene, voxel_mm=DEFAULT_VOXEL_MM, compute=True, report=None):
    """Return the object's DistanceField from memory, disk, or a fresh computation."""
    key = (mesh_fingerprint(obj), voxel_mm)
    field = _fields.get(key)
    if field is not None:
        return field
    path = field_path(scene, obj, voxel_mm)
    if os.path.exists(path):
        with np.load(path) as data:
            field = DistanceField.from_npz_dict(data)
    elif compute:
        field = compute_field(obj, scene, voxel_mm=voxel_mm, report=report)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **field.to_npz_dict())
    else:
        return None
    _fields[key] = field
    return field


def has_cached_field(obj, scene, voxel_mm=DEFAULT_VOXEL_MM):
    return (mesh_fingerprint(obj), voxel_mm) in _fields or os.path.exists(field_path(scene, obj, voxel_mm))


def surface_samples(obj, count=SURFACE_SAMPLE_COUNT):
    """A fixed random subset of local vertices used as collision probes."""
    key = (mesh_fingerprint(obj), count)
    samples = _samples.get(key)
    if samples is None:
        vertices = local_vertices(obj)
        if len(vertices) > count:
            rng = np.random.default_rng(SAMPLE_SEED)
            vertices = vertices[rng.choice(len(vertices), size=count, replace=False)]
        samples = _samples[key] = np.ascontiguousarray(vertices)
    return samples


def clear_field_cache():
    _fields.clear()
    _samples.clear()
    _fingerprints.clear()
