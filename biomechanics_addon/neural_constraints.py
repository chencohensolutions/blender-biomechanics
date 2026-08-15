"""Model-manifest loading, compatibility checks, and lazy ONNX Runtime session
management for the external neural constraint model (Phase 4).

Manifest/compatibility logic delegates to `schemas.py` (bpy-free, unit-testable);
this module owns the bpy-touching glue (reading object transforms/fingerprints,
importing/caching `onnxruntime`) so callers only need a `context` and a manifest
path.
"""

import json
from pathlib import Path

from . import onnx_compat, training
from .schemas import SchemaError, check_model_compatibility, validate_model_manifest

_SESSION_CACHE = {}  # resolved onnx path (str) -> onnxruntime.InferenceSession


class ModelLoadError(ValueError):
    """Raised when a model bundle cannot be read/parsed, or ONNX Runtime is unavailable."""


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
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
    key = str(Path(onnx_path).resolve())
    session = _SESSION_CACHE.get(key)
    if session is not None:
        return session
    import onnxruntime  # lazy: only required once a neural model is actually imported/used

    session = onnxruntime.InferenceSession(key, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = session
    return session


def clear_session_cache(onnx_path=None):
    """Drop one cached session (`onnx_path`) or all of them (no argument). Call this
    on add-on unregister/reload and whenever the active model path changes."""
    if onnx_path is None:
        _SESSION_CACHE.clear()
        return
    _SESSION_CACHE.pop(str(Path(onnx_path).resolve()), None)


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


def load_and_check_model(context, manifest_path):
    """Validate a model bundle end-to-end against the scene's currently selected
    fixed/moving training objects. Returns (manifest, onnx_path, reasons); a
    non-empty `reasons` list means the bundle must not be activated. Raises
    `ModelLoadError` only for I/O/parse failures or a missing ONNX Runtime -
    never for an incompatible/stale bundle."""
    scene = context.scene
    fixed_object = scene.biomechanics_training_fixed_object
    moving_object = scene.biomechanics_training_moving_object
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
        fixed_fingerprint=training.object_fingerprint(context, fixed_object),
        moving_fingerprint=training.object_fingerprint(context, moving_object),
        scale_length=scene.unit_settings.scale_length or 1.0,
        fixed_matrix_world=fixed_matrix_world,
    )

    try:
        session = get_session(onnx_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load ONNX model: {exc}") from exc
    reasons.extend(verify_model_io(session, manifest))

    return manifest, onnx_path, reasons
