"""Model-manifest loading, compatibility checks, and lazy ONNX Runtime session
management for the external neural constraint model (Phase 4).

Manifest/compatibility logic delegates to `schemas.py` (bpy-free, unit-testable);
this module owns the bpy-touching glue (reading object transforms/fingerprints,
importing/caching `onnxruntime`) so callers only need a `context` and a manifest
path.
"""

import json
import math
from pathlib import Path

from . import onnx_compat
from .schemas import SchemaError, check_model_compatibility, validate_model_manifest

_SESSION_CACHE = {}
_VALIDATED_MODEL_CACHE = {}
_SESSION_CACHE_GENERATION = 0


class ModelLoadError(ValueError):
    """Raised when a model bundle cannot be read/parsed, or ONNX Runtime is unavailable."""


class PoseOutOfRangeError(ValueError):
    """Raised when a requested pose is outside the model's trained range."""


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Could not read {manifest_path.name}: {exc}") from exc
    try:
        return validate_model_manifest(data)
    except SchemaError as exc:
        raise ModelLoadError(f"Invalid model manifest: {exc}") from exc


def onnx_path_for_manifest(manifest_path, manifest):
    return Path(manifest_path).resolve().parent / manifest["onnx_filename"]


def get_session(onnx_path):
    """Return a cached CPU ONNX Runtime session for `onnx_path`, creating it on first use."""
    path = Path(onnx_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    session = _SESSION_CACHE.get(key)
    if session is not None:
        return session
    import onnxruntime  # lazy: only required once a neural model is actually imported/used

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = session
    return session


def clear_session_cache(onnx_path=None):
    """Drop one cached session (`onnx_path`) or all of them (no argument). Call this
    on add-on unregister/reload and whenever the active model path changes."""
    global _SESSION_CACHE_GENERATION
    _SESSION_CACHE_GENERATION += 1
    if onnx_path is None:
        _SESSION_CACHE.clear()
        _VALIDATED_MODEL_CACHE.clear()
        return
    path = str(Path(onnx_path).resolve())
    for key in tuple(_SESSION_CACHE):
        if key[0] == path:
            del _SESSION_CACHE[key]
    for scene_key, entry in tuple(_VALIDATED_MODEL_CACHE.items()):
        if entry["onnx_path"] == path:
            del _VALIDATED_MODEL_CACHE[scene_key]


def session_cache_generation():
    return _SESSION_CACHE_GENERATION


def _scene_pointer(scene):
    pointer = getattr(scene, "as_pointer", None)
    return pointer() if pointer is not None else id(scene)


def _object_pointer(obj):
    pointer = getattr(obj, "as_pointer", None)
    return pointer() if pointer is not None else id(obj)


def _matrix_values(matrix):
    return tuple(float(value) for row in matrix for value in row)


def _model_scene_signature(scene, fixed_object=None, moving_object=None):
    fixed_object = fixed_object or scene.biomechanics_training_fixed_object
    moving_object = moving_object or scene.biomechanics_training_moving_object
    return (
        _object_pointer(fixed_object),
        _object_pointer(moving_object),
        fixed_object.name,
        moving_object.name,
        _matrix_values(fixed_object.matrix_world),
        float(scene.unit_settings.scale_length or 1.0),
    )


def invalidate_model_cache(scene=None, depsgraph=None):
    """Invalidate cached compatibility when a relevant mesh datablock changes."""
    if scene is None:
        _VALIDATED_MODEL_CACHE.clear()
        return
    scene_key = _scene_pointer(scene)
    entry = _VALIDATED_MODEL_CACHE.get(scene_key)
    if entry is None or depsgraph is None:
        return
    fixed_data = entry["fixed_object"].data
    moving_data = entry["moving_object"].data
    for update in depsgraph.updates:
        updated_id = update.id
        if updated_id is fixed_data or updated_id is moving_data:
            _VALIDATED_MODEL_CACHE.pop(scene_key, None)
            return
        if updated_id in (entry["fixed_object"], entry["moving_object"]):
            if getattr(update, "is_updated_geometry", False):
                _VALIDATED_MODEL_CACHE.pop(scene_key, None)
                return


def verify_model_io(session, manifest):
    """Check the ONNX graph's actual input/output names match the manifest."""
    reasons = []
    input_names = {node.name for node in session.get_inputs()}
    output_names = {node.name for node in session.get_outputs()}
    if manifest["input_name"] not in input_names:
        reasons.append(f"ONNX model has no input named '{manifest['input_name']}' (found: {sorted(input_names)})")
    for label, name in manifest["output_names"].items():
        if name not in output_names:
            reasons.append(f"ONNX model has no output named '{name}' for '{label}' (found: {sorted(output_names)})")
    return reasons


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def predict_poses(session, manifest, poses):
    """Predict a sequence of raw [tx, ty, tz, rx, ry, rz] poses."""
    import numpy as np

    values = np.asarray(poses, dtype=np.float32)
    if values.size == 0:
        return []
    values = values.reshape((-1, 6))
    normalization = manifest["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    outputs = session.run(
        list(manifest["output_names"].values()),
        {manifest["input_name"]: values},
    )
    logits = np.asarray(outputs[0]).reshape(-1)
    clearance = np.asarray(outputs[1]).reshape(-1)
    threshold = float(manifest["validity_threshold"])
    return [
        {
            "validity_probability": _sigmoid(float(logit)),
            "valid": _sigmoid(float(logit)) >= threshold,
            "clearance_mm": float(clearance[index]),
        }
        for index, logit in enumerate(logits)
    ]


def neutral_pose_error(session, manifest):
    """Return why a model cannot begin conservative projection at neutral, if any."""
    prediction = predict_poses(session, manifest, ((0.0,) * 6,))[0]
    if prediction["valid"]:
        return None
    return (
        "trained neutral pose is rejected "
        f"(probability {prediction['validity_probability']:.3f} < "
        f"threshold {float(manifest['validity_threshold']):.3f}); retrain this pair"
    )


def pose_distance_mm(first, second, joint_radius_mm):
    """Return the combined translation plus rotational arc distance."""
    translation = math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))
    first_quaternion = _euler_xyz_to_quaternion(first[3:])
    second_quaternion = _euler_xyz_to_quaternion(second[3:])
    dot = abs(sum(first_quaternion[index] * second_quaternion[index] for index in range(4)))
    angle = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    return translation + joint_radius_mm * angle


def _euler_xyz_to_quaternion(angles):
    half_x, half_y, half_z = (math.radians(value) * 0.5 for value in angles)
    cx, sx = math.cos(half_x), math.sin(half_x)
    cy, sy = math.cos(half_y), math.sin(half_y)
    cz, sz = math.cos(half_z), math.sin(half_z)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


def _quaternion_to_euler_xyz(quaternion):
    w, x, y, z = quaternion
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _slerp(first, second, fraction):
    dot = sum(first[index] * second[index] for index in range(4))
    if dot < 0.0:
        second = tuple(-value for value in second)
        dot = -dot
    if dot > 0.9995:
        result = tuple(first[index] + fraction * (second[index] - first[index]) for index in range(4))
        length = math.sqrt(sum(value * value for value in result))
        return tuple(value / length for value in result)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    denominator = math.sin(angle)
    first_weight = math.sin((1.0 - fraction) * angle) / denominator
    second_weight = math.sin(fraction * angle) / denominator
    return tuple(first_weight * first[index] + second_weight * second[index] for index in range(4))


def interpolate_pose(first, second, fraction):
    """Interpolate translation linearly and rotation along the shortest slerp path."""
    first_rotation = _euler_xyz_to_quaternion(first[3:])
    second_rotation = _euler_xyz_to_quaternion(second[3:])
    rotation = _quaternion_to_euler_xyz(_slerp(first_rotation, second_rotation, fraction))
    translation = tuple(
        first[index] + fraction * (second[index] - first[index]) for index in range(3)
    )
    return translation + rotation


def _pose_in_range(pose, manifest):
    limits = manifest["range"]
    translation_limit = float(limits["translation_limit_mm"])
    rotation_limit = float(limits["rotation_limit_degrees"])
    return (
        all(abs(value) <= translation_limit + 1e-6 for value in pose[:3])
        and all(abs(value) <= rotation_limit + 1e-6 for value in pose[3:])
    )


def clamp_pose_to_range(pose, manifest):
    """Clamp each local translation/rotation component to the trained domain."""
    limits = manifest["range"]
    translation_limit = float(limits["translation_limit_mm"])
    rotation_limit = float(limits["rotation_limit_degrees"])
    return tuple(
        max(-translation_limit, min(translation_limit, value))
        for value in pose[:3]
    ) + tuple(
        max(-rotation_limit, min(rotation_limit, value))
        for value in pose[3:]
    )


def project_pose(session, manifest, start_pose, target_pose, scan_steps=16):
    """Return the farthest conservative valid point on the start-to-target path."""
    if not _pose_in_range(target_pose, manifest):
        raise PoseOutOfRangeError("requested pose is outside the trained model range")
    scan_fractions = tuple(index / scan_steps for index in range(1, scan_steps + 1))
    scan_poses = tuple(
        interpolate_pose(start_pose, target_pose, fraction)
        for fraction in scan_fractions
    )
    predictions = predict_poses(session, manifest, (start_pose,) + scan_poses)
    if not predictions[0]["valid"]:
        raise ModelLoadError("the last accepted pose is no longer valid")

    low_fraction = 0.0
    high_fraction = 1.0
    for fraction, prediction in zip(
        scan_fractions,
        predictions[1:],
    ):
        if not prediction["valid"]:
            high_fraction = fraction
            break
        low_fraction = fraction
    else:
        return tuple(target_pose), predictions[-1]

    target_distance = float(manifest["pose_distance_target_mm"])
    while pose_distance_mm(
        interpolate_pose(start_pose, target_pose, low_fraction),
        interpolate_pose(start_pose, target_pose, high_fraction),
        float(manifest["range"]["joint_radius_mm"]),
    ) > target_distance:
        midpoint = (low_fraction + high_fraction) * 0.5
        pose = interpolate_pose(start_pose, target_pose, midpoint)
        if predict_poses(session, manifest, (pose,))[0]["valid"]:
            low_fraction = midpoint
        else:
            high_fraction = midpoint
    projected_pose = interpolate_pose(start_pose, target_pose, low_fraction)
    return tuple(projected_pose), predict_poses(session, manifest, (projected_pose,))[0]


def load_and_check_model(context, manifest_path, fixed_object=None, moving_object=None):
    """Validate a model bundle end-to-end against the scene's currently selected
    fixed/moving training objects. Returns (manifest, onnx_path, reasons); a
    non-empty `reasons` list means the bundle must not be activated. Raises
    `ModelLoadError` only for I/O/parse failures or a missing ONNX Runtime -
    never for an incompatible/stale bundle."""
    from . import neural_geometry

    scene = context.scene
    fixed_object = fixed_object or scene.biomechanics_training_fixed_object
    moving_object = moving_object or scene.biomechanics_training_moving_object
    if fixed_object is None or moving_object is None:
        raise ModelLoadError("Choose fixed and moving bones before importing a model")

    manifest = load_manifest(manifest_path)
    onnx_path = onnx_path_for_manifest(manifest_path, manifest)
    if not onnx_path.is_file():
        raise ModelLoadError(f"Sibling ONNX file not found next to the manifest: {onnx_path.name}")

    onnx_report = onnx_compat.check_onnx_runtime()
    if not onnx_report["onnxruntime_available"]:
        raise ModelLoadError(f"ONNX Runtime unavailable: {onnx_report['error']}")

    fixed_matrix_world = [value for row in fixed_object.matrix_world for value in row]
    reasons = check_model_compatibility(
        manifest,
        fixed_object_name=fixed_object.name,
        moving_object_name=moving_object.name,
        fixed_fingerprint=neural_geometry.object_fingerprint(context, fixed_object),
        moving_fingerprint=neural_geometry.object_fingerprint(context, moving_object),
        scale_length=scene.unit_settings.scale_length or 1.0,
        fixed_matrix_world=fixed_matrix_world,
    )

    try:
        session = get_session(onnx_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load ONNX model: {exc}") from exc
    reasons.extend(verify_model_io(session, manifest))
    neutral_error = neutral_pose_error(session, manifest)
    if neutral_error is not None:
        reasons.append(neutral_error)

    global _SESSION_CACHE_GENERATION
    _SESSION_CACHE_GENERATION += 1
    result = (manifest, onnx_path, reasons)
    _VALIDATED_MODEL_CACHE[_scene_pointer(scene)] = {
        "manifest_path": str(manifest_path),
        "manifest_signature": (manifest_path.stat().st_mtime_ns, manifest_path.stat().st_size),
        "onnx_path": str(onnx_path),
        "onnx_signature": (onnx_path.stat().st_mtime_ns, onnx_path.stat().st_size),
        "fixed_object": fixed_object,
        "moving_object": moving_object,
        "scene_signature": _model_scene_signature(scene, fixed_object, moving_object),
        "result": result,
    }
    return result


def load_cached_model(context, manifest_path, fixed_object=None, moving_object=None):
    """Return a previously validated model when its cheap scene/file signature is unchanged."""
    path = Path(manifest_path).resolve()
    scene = context.scene
    entry = _VALIDATED_MODEL_CACHE.get(_scene_pointer(scene))
    if entry is not None:
        try:
            onnx_path = path.parent / entry["result"][0]["onnx_filename"]
            signature_matches = (
                entry["manifest_path"] == str(path)
                and entry["manifest_signature"] == (path.stat().st_mtime_ns, path.stat().st_size)
                and entry["onnx_signature"] == (onnx_path.stat().st_mtime_ns, onnx_path.stat().st_size)
                and entry["scene_signature"] == _model_scene_signature(scene, fixed_object, moving_object)
            )
        except (AttributeError, OSError):
            signature_matches = False
        if signature_matches:
            return entry["result"]
    return load_and_check_model(context, path, fixed_object, moving_object)
