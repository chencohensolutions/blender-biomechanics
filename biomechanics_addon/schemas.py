"""Versioned schemas for external neural constraint training artifacts.

This module has no `bpy` dependency so it can be imported both by the add-on
(inside Blender) and by external tooling (plain Python, no Blender installed).
External tools should load it with `tools/_addon_import.py` rather than
`import biomechanics_addon.schemas`, because importing the `biomechanics_addon`
package executes its `__init__.py`, which requires `bpy`.
"""

import hashlib
import math


PACKAGE_SCHEMA_VERSION = 2
DATASET_SCHEMA_VERSION = 1
MODEL_MANIFEST_SCHEMA_VERSION = 1

# Fixed geometric assumptions shared by every trained joint pair.
JOINT_RADIUS_MM = 50.0
POSE_DISTANCE_TARGET_MM = 0.5

VALID_LABEL_SOURCES = ("proxy", "full_mesh")
VALID_SPLITS = ("train", "validation", "holdout")


class SchemaError(ValueError):
    """Raised when a package, dataset row, or model manifest fails validation."""


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def _raise_if_errors(errors):
    if errors:
        raise SchemaError("; ".join(errors))


def geometry_fingerprint(vertices, polygons, precision=6):
    """Stable content hash of mesh geometry, used to detect a stale trained model."""
    hasher = hashlib.sha256()
    hasher.update(len(vertices).to_bytes(8, "little"))
    hasher.update(len(polygons).to_bytes(8, "little"))
    for vertex in vertices:
        for value in vertex:
            hasher.update(f"{round(value, precision):.{precision}f}".encode("utf-8"))
    for polygon in polygons:
        hasher.update(",".join(str(index) for index in polygon).encode("utf-8"))
        hasher.update(b";")
    return hasher.hexdigest()


def pose_path_distance_mm(translation_delta_mm, rotation_delta_radians, joint_radius_mm=JOINT_RADIUS_MM):
    """Combined 6-DOF path distance: translation magnitude plus rotational arc length."""
    translation_magnitude = math.sqrt(sum(value * value for value in translation_delta_mm))
    rotational_arc_length = joint_radius_mm * abs(rotation_delta_radians)
    return translation_magnitude + rotational_arc_length


# ---------------------------------------------------------------------------
# Package v2: geometry + settings exported from interactive Blender.
# ---------------------------------------------------------------------------

REQUIRED_PACKAGE_KEYS = (
    "package_version",
    "collection",
    "fixed_object",
    "moving_object",
    "scale_length",
    "range",
    "sampling",
)


def build_package_v2(
    *,
    collection_name,
    fixed_object_name,
    moving_object_name,
    neutral_fixed_matrix,
    neutral_moving_matrix,
    fixed_fingerprint,
    moving_fingerprint,
    scale_length,
    translation_limit_mm,
    rotation_limit_degrees,
    seed,
    proxy_max_faces,
    use_full_mesh_for_holdout=True,
    joint_radius_mm=JOINT_RADIUS_MM,
    rotation_center_world=None,
    pose_transform_version=2,
):
    errors = []
    _require(fixed_object_name != moving_object_name, "fixed_object and moving_object must differ", errors)
    _require(len(neutral_fixed_matrix) == 16, "neutral_fixed_matrix must have 16 values", errors)
    _require(len(neutral_moving_matrix) == 16, "neutral_moving_matrix must have 16 values", errors)
    _require(scale_length > 0, "scale_length must be positive", errors)
    _require(translation_limit_mm > 0, "translation_limit_mm must be positive", errors)
    _require(rotation_limit_degrees > 0, "rotation_limit_degrees must be positive", errors)
    _require(joint_radius_mm > 0, "joint_radius_mm must be positive", errors)
    if rotation_center_world is not None:
        _require(len(rotation_center_world) == 3, "rotation_center_world must have 3 values", errors)
    _raise_if_errors(errors)

    return {
        "package_version": PACKAGE_SCHEMA_VERSION,
        "collection": collection_name,
        "fixed_object": {
            "name": fixed_object_name,
            "matrix_world": list(neutral_fixed_matrix),
            "fingerprint": fixed_fingerprint,
        },
        "moving_object": {
            "name": moving_object_name,
            "matrix_world": list(neutral_moving_matrix),
            "fingerprint": moving_fingerprint,
        },
        "scale_length": scale_length,
        "pose_transform_version": pose_transform_version,
        "rotation_center_world": list(
            rotation_center_world
            if rotation_center_world is not None
            else (neutral_moving_matrix[3], neutral_moving_matrix[7], neutral_moving_matrix[11])
        ),
        "range": {
            "translation_limit_mm": translation_limit_mm,
            "rotation_limit_degrees": rotation_limit_degrees,
            "joint_radius_mm": joint_radius_mm,
        },
        "sampling": {
            "seed": seed,
            "proxy_max_faces": proxy_max_faces,
            "use_full_mesh_for_holdout": use_full_mesh_for_holdout,
        },
        "pose_distance_target_mm": POSE_DISTANCE_TARGET_MM,
    }


def validate_package_v2(data):
    errors = []
    if not isinstance(data, dict):
        raise SchemaError("package must be a JSON object")
    for key in REQUIRED_PACKAGE_KEYS:
        _require(key in data, f"package missing required key '{key}'", errors)
    _raise_if_errors(errors)

    _require(
        data["package_version"] == PACKAGE_SCHEMA_VERSION,
        f"unsupported package_version {data['package_version']!r}, expected {PACKAGE_SCHEMA_VERSION}",
        errors,
    )
    for role in ("fixed_object", "moving_object"):
        obj = data.get(role)
        if not isinstance(obj, dict):
            errors.append(f"{role} must be an object")
            continue
        _require("name" in obj, f"{role}.name is required", errors)
        _require(
            len(obj.get("matrix_world", [])) == 16,
            f"{role}.matrix_world must have 16 values",
            errors,
        )
        _require("fingerprint" in obj, f"{role}.fingerprint is required", errors)

    if isinstance(data.get("fixed_object"), dict) and isinstance(data.get("moving_object"), dict):
        _require(
            data["fixed_object"].get("name") != data["moving_object"].get("name"),
            "fixed_object and moving_object must differ",
            errors,
        )

    range_block = data.get("range", {})
    for key in ("translation_limit_mm", "rotation_limit_degrees", "joint_radius_mm"):
        _require(key in range_block, f"range.{key} is required", errors)

    _raise_if_errors(errors)

    # Packages produced before cursor-pivot support rotate about the moving
    # object's neutral origin, which remains their compatible default.
    if "rotation_center_world" not in data:
        matrix = data["moving_object"]["matrix_world"]
        data["rotation_center_world"] = [matrix[3], matrix[7], matrix[11]]
    data.setdefault("pose_transform_version", 1)
    _require(data["pose_transform_version"] in (1, 2), "unsupported pose_transform_version", errors)
    _require(len(data["rotation_center_world"]) == 3, "rotation_center_world must have 3 values", errors)
    _raise_if_errors(errors)
    return data


# ---------------------------------------------------------------------------
# Dataset v1: one row per labeled local 6-DOF pose.
# ---------------------------------------------------------------------------

DATASET_ROW_KEYS = (
    "dataset_version",
    "pose_id",
    "translation_mm",
    "rotation_degrees",
    "valid",
    "clearance_mm",
    "overlap_pairs",
    "label_source",
    "split",
    "group_id",
)


def build_dataset_row(
    *,
    pose_id,
    translation_mm,
    rotation_degrees,
    valid,
    clearance_mm,
    overlap_pairs,
    label_source,
    split,
    group_id,
):
    errors = []
    _require(len(translation_mm) == 3, "translation_mm must have 3 values", errors)
    _require(len(rotation_degrees) == 3, "rotation_degrees must have 3 values", errors)
    _require(label_source in VALID_LABEL_SOURCES, f"label_source must be one of {VALID_LABEL_SOURCES}", errors)
    _require(split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}", errors)
    _require(not valid or clearance_mm >= 0, "clearance_mm must be non-negative for valid poses", errors)
    _raise_if_errors(errors)

    return {
        "dataset_version": DATASET_SCHEMA_VERSION,
        "pose_id": pose_id,
        "translation_mm": list(translation_mm),
        "rotation_degrees": list(rotation_degrees),
        "valid": bool(valid),
        "clearance_mm": clearance_mm,
        "overlap_pairs": overlap_pairs,
        "label_source": label_source,
        "split": split,
        "group_id": group_id,
    }


def validate_dataset_row(row):
    errors = []
    if not isinstance(row, dict):
        raise SchemaError("dataset row must be an object")
    for key in DATASET_ROW_KEYS:
        _require(key in row, f"dataset row missing '{key}'", errors)
    _raise_if_errors(errors)

    _require(
        row["dataset_version"] == DATASET_SCHEMA_VERSION,
        f"unsupported dataset_version {row['dataset_version']!r}, expected {DATASET_SCHEMA_VERSION}",
        errors,
    )
    _require(len(row["translation_mm"]) == 3, "translation_mm must have 3 values", errors)
    _require(len(row["rotation_degrees"]) == 3, "rotation_degrees must have 3 values", errors)
    _require(row["label_source"] in VALID_LABEL_SOURCES, "invalid label_source", errors)
    _require(row["split"] in VALID_SPLITS, "invalid split", errors)
    _raise_if_errors(errors)
    return row


# ---------------------------------------------------------------------------
# Model manifest v1: describes an exported ONNX bundle for Blender import.
# ---------------------------------------------------------------------------

MODEL_MANIFEST_KEYS = (
    "model_version",
    "onnx_filename",
    "opset",
    "input_name",
    "output_names",
    "normalization",
    "fixed_object",
    "moving_object",
    "fixed_fingerprint",
    "moving_fingerprint",
    "scale_length",
    "neutral_fixed_matrix",
    "neutral_moving_matrix",
    "range",
    "validity_threshold",
    "pose_distance_target_mm",
    "metrics",
)


def build_model_manifest_v1(
    *,
    onnx_filename,
    opset,
    input_name,
    validity_output_name,
    clearance_output_name,
    normalization,
    fixed_object,
    moving_object,
    fixed_fingerprint,
    moving_fingerprint,
    scale_length,
    neutral_fixed_matrix,
    neutral_moving_matrix,
    translation_limit_mm,
    rotation_limit_degrees,
    validity_threshold,
    metrics,
    joint_radius_mm=JOINT_RADIUS_MM,
    rotation_center_world=None,
    pose_transform_version=2,
):
    errors = []
    _require(opset > 0, "opset must be positive", errors)
    _require(0.0 <= validity_threshold <= 1.0, "validity_threshold must be between 0 and 1", errors)
    _require(scale_length > 0, "scale_length must be positive", errors)
    _require(len(neutral_fixed_matrix) == 16, "neutral_fixed_matrix must have 16 values", errors)
    _require(len(neutral_moving_matrix) == 16, "neutral_moving_matrix must have 16 values", errors)
    if rotation_center_world is not None:
        _require(len(rotation_center_world) == 3, "rotation_center_world must have 3 values", errors)
    _raise_if_errors(errors)

    return {
        "model_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "onnx_filename": onnx_filename,
        "opset": opset,
        "input_name": input_name,
        "output_names": {
            "validity_logit": validity_output_name,
            "clearance_mm": clearance_output_name,
        },
        "normalization": normalization,
        "fixed_object": fixed_object,
        "moving_object": moving_object,
        "fixed_fingerprint": fixed_fingerprint,
        "moving_fingerprint": moving_fingerprint,
        "scale_length": scale_length,
        "pose_transform_version": pose_transform_version,
        "neutral_fixed_matrix": list(neutral_fixed_matrix),
        "neutral_moving_matrix": list(neutral_moving_matrix),
        "rotation_center_world": list(
            rotation_center_world
            if rotation_center_world is not None
            else (neutral_moving_matrix[3], neutral_moving_matrix[7], neutral_moving_matrix[11])
        ),
        "range": {
            "translation_limit_mm": translation_limit_mm,
            "rotation_limit_degrees": rotation_limit_degrees,
            "joint_radius_mm": joint_radius_mm,
        },
        "validity_threshold": validity_threshold,
        "pose_distance_target_mm": POSE_DISTANCE_TARGET_MM,
        "metrics": metrics,
    }


def validate_model_manifest(data):
    errors = []
    if not isinstance(data, dict):
        raise SchemaError("model manifest must be an object")
    for key in MODEL_MANIFEST_KEYS:
        _require(key in data, f"model manifest missing '{key}'", errors)
    _raise_if_errors(errors)

    _require(
        data["model_version"] == MODEL_MANIFEST_SCHEMA_VERSION,
        f"unsupported model_version {data['model_version']!r}, expected {MODEL_MANIFEST_SCHEMA_VERSION}",
        errors,
    )
    output_names = data.get("output_names", {})
    _require("validity_logit" in output_names, "output_names.validity_logit is required", errors)
    _require("clearance_mm" in output_names, "output_names.clearance_mm is required", errors)
    _require(0.0 <= data.get("validity_threshold", -1) <= 1.0, "validity_threshold must be between 0 and 1", errors)
    _raise_if_errors(errors)

    if "rotation_center_world" not in data:
        matrix = data["neutral_moving_matrix"]
        data["rotation_center_world"] = [matrix[3], matrix[7], matrix[11]]
    data.setdefault("pose_transform_version", 1)
    _require(data["pose_transform_version"] in (1, 2), "unsupported pose_transform_version", errors)
    _require(len(data["rotation_center_world"]) == 3, "rotation_center_world must have 3 values", errors)
    _raise_if_errors(errors)
    return data


def check_model_compatibility(
    manifest,
    *,
    fixed_object_name,
    moving_object_name,
    fixed_fingerprint,
    moving_fingerprint,
    scale_length=None,
    fixed_matrix_world=None,
    scale_tolerance=1e-6,
    matrix_tolerance=1e-4,
):
    """Return human-readable incompatibility reasons; an empty list means compatible.
    `scale_length`/`fixed_matrix_world` are optional so existing callers that only
    have object names/fingerprints (no live scene state) keep working unchanged."""
    reasons = []
    if manifest.get("fixed_object", {}).get("name") != fixed_object_name:
        reasons.append("fixed object name does not match the trained model")
    if manifest.get("moving_object", {}).get("name") != moving_object_name:
        reasons.append("moving object name does not match the trained model")
    if manifest.get("fixed_fingerprint") != fixed_fingerprint:
        reasons.append("fixed object geometry has changed since training")
    if manifest.get("moving_fingerprint") != moving_fingerprint:
        reasons.append("moving object geometry has changed since training")
    if scale_length is not None:
        manifest_scale_length = manifest.get("scale_length")
        if manifest_scale_length is None or abs(manifest_scale_length - scale_length) > scale_tolerance:
            reasons.append(f"scene scale_length {scale_length} does not match the trained scale_length {manifest_scale_length}")
    if fixed_matrix_world is not None:
        neutral_fixed_matrix = manifest.get("neutral_fixed_matrix")
        if not neutral_fixed_matrix or len(neutral_fixed_matrix) != 16:
            reasons.append("model manifest is missing the fixed object's neutral transform")
        else:
            max_difference = max(abs(a - b) for a, b in zip(fixed_matrix_world, neutral_fixed_matrix))
            if max_difference > matrix_tolerance:
                reasons.append(
                    f"{fixed_object_name} has moved since training (max matrix component difference "
                    f"{max_difference:.6g}); re-align it to its trained neutral transform before importing this model"
                )
    return reasons
