"""Chunked/compressed dataset row I/O, shared by the Blender-side generator and the
external (bpy-free) trainer.

This module has no `bpy` dependency and does not import `schemas.py`, so it can be
loaded either as a normal package submodule (inside Blender, where the package
`__init__.py` succeeds because `bpy` is present) or standalone via
`tools/_addon_import.py` (outside Blender). Callers are responsible for validating
rows against `schemas.build_dataset_row`/`validate_dataset_row` before/after using
these functions.
"""

import json

import numpy as np

ARRAY_DTYPES = {
    "dataset_version": np.int64,
    "pose_id": "U64",
    "rotation_degrees": np.float64,
    "translation_mm": np.float64,
    "valid": bool,
    "clearance_mm": np.float64,
    "overlap_pairs": np.int64,
    "label_source": "U16",
    "split": "U16",
    "group_id": "U64",
}


def rows_to_arrays(rows):
    return {key: np.array([row[key] for row in rows], dtype=dtype) for key, dtype in ARRAY_DTYPES.items()}


def arrays_to_rows(arrays):
    return [
        {
            "dataset_version": int(arrays["dataset_version"][index]),
            "pose_id": str(arrays["pose_id"][index]),
            "translation_mm": arrays["translation_mm"][index].tolist(),
            "rotation_degrees": arrays["rotation_degrees"][index].tolist(),
            "valid": bool(arrays["valid"][index]),
            "clearance_mm": float(arrays["clearance_mm"][index]),
            "overlap_pairs": int(arrays["overlap_pairs"][index]),
            "label_source": str(arrays["label_source"][index]),
            "split": str(arrays["split"][index]),
            "group_id": str(arrays["group_id"][index]),
        }
        for index in range(len(arrays["pose_id"]))
    ]


def write_stage_rows(output_dir, stage, rows, chunk_size):
    stage_dir = output_dir / "chunks" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    for start in range(0, len(rows), chunk_size):
        arrays = rows_to_arrays(rows[start:start + chunk_size])
        final_path = stage_dir / f"chunk_{chunk_count:05d}.npz"
        temporary_path = final_path.with_name(final_path.name + ".tmp")
        # Pass an open handle, not a path string: np.savez_compressed appends ".npz" to
        # any string/PathLike argument that doesn't already end with it.
        with open(temporary_path, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary_path.replace(final_path)
        chunk_count += 1
    return chunk_count


def load_stage_rows(output_dir, stage, chunk_count):
    rows = []
    stage_dir = output_dir / "chunks" / stage
    for index in range(chunk_count):
        with np.load(stage_dir / f"chunk_{index:05d}.npz") as data:
            rows.extend(arrays_to_rows(data))
    return rows


def write_meta(output_dir, meta):
    meta_path = output_dir / "dataset_meta.json"
    temporary_path = meta_path.with_name(meta_path.name + ".tmp")
    temporary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temporary_path.replace(meta_path)


def load_meta(output_dir):
    meta_path = output_dir / "dataset_meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_dataset(output_dir):
    """Load every completed stage's rows from a dataset directory written by
    `tools/generate_neural_dataset.py`. Returns (rows, meta)."""
    meta = load_meta(output_dir)
    rows = []
    for stage in meta["completed_stages"]:
        rows.extend(load_stage_rows(output_dir, stage, meta["stage_chunk_counts"].get(stage, 0)))
    return rows, meta
