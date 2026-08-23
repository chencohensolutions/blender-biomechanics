"""Generate a labeled neural-constraint dataset from an exported package v2 in background Blender.

Usage:
    blender --background --factory-startup --python tools/generate_neural_dataset.py -- \\
        --package-dir <export_directory> [--output-dir <dir>] [--sample-budget 2000]

Resumable: progress is checkpointed per stage in <output-dir>/dataset_meta.json; rerunning
with the same package/options skips already-completed stages and loads their rows back from
the compressed NPZ chunks instead of re-evaluating geometry.
"""

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix  # pyright: ignore[reportMissingImports]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from biomechanics_addon import dataset_io, sampling
from biomechanics_addon.schemas import build_dataset_row, validate_dataset_row, validate_package_v2
from biomechanics_addon.neural_geometry import (
    evaluate_local_pose,
    full_mesh_geometry,
    object_fingerprint,
    proxy_geometry,
    world_bvh,
)


SAMPLING_ALGORITHM_VERSION = 2


def parse_arguments():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-budget", type=int, default=2000)
    parser.add_argument("--minimum-clearance-mm", type=float, default=0.5)
    parser.add_argument("--max-balance-rounds", type=int, default=2)
    parser.add_argument("--balance-round-size", type=int, default=250)
    parser.add_argument("--max-refinement-pairs", type=int, default=150)
    parser.add_argument("--refinement-rounds", type=int, default=3)
    parser.add_argument("--refinement-search-subset", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--label-poses-file", type=Path, default=None,
        help="Skip the normal sampling pipeline and instead label exactly the poses in this JSON "
        "file (a list of {pose_id, translation_mm, rotation_degrees[, split, group_id]} objects) "
        "with full-mesh geometry, writing them as a new dataset stage. Used by the active-learning "
        "loop (tools/active_learning_loop.py) to request labels for specific uncertain poses.",
    )
    parser.add_argument("--stage-name", type=str, default=None, help="Stage name to write under when using --label-poses-file.")
    return parser.parse_args(arguments)


def _matrix_from_values(values):
    return Matrix(tuple(tuple(values[index:index + 4]) for index in range(0, 16, 4)))


def _configure_logging(output_dir):
    # Some Blender launchers (e.g. the Microsoft Store package) do not forward
    # stdout/stderr to the invoking console, so always mirror progress to a file.
    log_path = output_dir / "dataset_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [dataset] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _stage_rng(seed, stage_name):
    stage_seed = int(hashlib.sha256(f"{seed}:{stage_name}".encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(stage_seed)


def _config_fingerprint(package, args):
    payload = json.dumps(
        {
            "fixed_fingerprint": package["fixed_object"]["fingerprint"],
            "moving_fingerprint": package["moving_object"]["fingerprint"],
            "range": package["range"],
            "sampling": package["sampling"],
            "pose_transform_version": package["pose_transform_version"],
            "sample_budget": args.sample_budget,
            "minimum_clearance_mm": args.minimum_clearance_mm,
            "max_balance_rounds": args.max_balance_rounds,
            "balance_round_size": args.balance_round_size,
            "max_refinement_pairs": args.max_refinement_pairs,
            "refinement_rounds": args.refinement_rounds,
            "refinement_search_subset": args.refinement_search_subset,
            "chunk_size": args.chunk_size,
            "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_package_objects(package_dir, package):
    input_path = package_dir / "training_input.blend"
    object_names = [package["fixed_object"]["name"], package["moving_object"]["name"]]
    with bpy.data.libraries.load(str(input_path), link=False) as (data_from, data_to):
        missing = [name for name in object_names if name not in data_from.objects]
        if missing:
            raise ValueError(f"Objects missing from training input: {', '.join(missing)}")
        data_to.objects = object_names

    # Look up by position, not by .name: Blender renames on a collision with an
    # object already present in the scene (e.g. the source anatomy is still open).
    fixed_object, moving_object = data_to.objects
    if fixed_object is None or moving_object is None:
        raise ValueError("Failed to load fixed/moving objects from training input")

    for obj in (fixed_object, moving_object):
        bpy.context.scene.collection.objects.link(obj)

    fixed_object.matrix_world = _matrix_from_values(package["fixed_object"]["matrix_world"])
    moving_object.matrix_world = _matrix_from_values(package["moving_object"]["matrix_world"])
    bpy.context.view_layer.update()
    return fixed_object, moving_object


def verify_fingerprints(context, package, fixed_object, moving_object):
    mismatches = []
    if object_fingerprint(context, fixed_object) != package["fixed_object"]["fingerprint"]:
        mismatches.append(f"{fixed_object.name} geometry does not match the exported package")
    if object_fingerprint(context, moving_object) != package["moving_object"]["fingerprint"]:
        mismatches.append(f"{moving_object.name} geometry does not match the exported package")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def _make_evaluator(
    fixed_bvh,
    fixed_world_vertices,
    moving_vertices,
    moving_polygons,
    neutral_matrix,
    rotation_center_world,
    pose_transform_version,
    baseline_overlap_pairs,
    minimum_clearance,
    millimeters_per_blender_unit,
):
    def evaluate(translation_mm, rotation_degrees):
        tx, ty, tz = (value / millimeters_per_blender_unit for value in translation_mm)
        rx, ry, rz = (math.radians(value) for value in rotation_degrees)
        return evaluate_local_pose(
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
        )

    return evaluate


def _build_row(pose_id, translation_mm, rotation_degrees, evaluation, label_source, split, group_id, millimeters_per_blender_unit):
    return build_dataset_row(
        pose_id=pose_id,
        translation_mm=list(translation_mm),
        rotation_degrees=list(rotation_degrees),
        valid=evaluation["valid"],
        clearance_mm=evaluation["clearance"] * millimeters_per_blender_unit,
        overlap_pairs=evaluation["overlap_pairs"],
        label_source=label_source,
        split=split,
        group_id=group_id,
    )


def _generate_anchor_rows(evaluate_full, translation_limit_mm, rotation_limit_degrees, millimeters_per_blender_unit):
    rows = []
    neutral_translation, neutral_rotation = sampling.neutral_pose()
    evaluation = evaluate_full(neutral_translation, neutral_rotation)
    rows.append(
        _build_row(
            "anchor-neutral", neutral_translation, neutral_rotation, evaluation,
            "full_mesh", "train", "anchor-neutral", millimeters_per_blender_unit,
        )
    )
    for translation, rotation, label in sampling.axis_anchor_poses(translation_limit_mm, rotation_limit_degrees):
        if label.startswith("r"):
            continue
        evaluation = evaluate_full(translation, rotation)
        pose_id = f"anchor-{label}"
        rows.append(
            _build_row(pose_id, translation, rotation, evaluation, "full_mesh", "validation", pose_id, millimeters_per_blender_unit)
        )
    for translation, rotation, label, split in sampling.rotation_sweep_poses(rotation_limit_degrees):
        evaluation = evaluate_full(translation, rotation)
        pose_id = f"rotation-sweep-{label}"
        rows.append(
            _build_row(
                pose_id,
                translation,
                rotation,
                evaluation,
                "full_mesh",
                split,
                pose_id,
                millimeters_per_blender_unit,
            )
        )
    return rows


def _generate_lhs_rows(
    evaluate_proxy, evaluate_full, translation_limit_mm, rotation_limit_degrees, count, seed, millimeters_per_blender_unit
):
    if count <= 0:
        return []
    translation_mm, rotation_degrees = sampling.multiscale_pose_samples(
        _stage_rng(seed, "lhs"),
        count,
        translation_limit_mm,
        rotation_limit_degrees,
    )
    splits = sampling.split_indices(count, _stage_rng(seed, "lhs-split"))
    split_of_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_of_index[int(index)] = split_name

    rows = []
    for index in range(count):
        split_name = split_of_index[index]
        label_source = "full_mesh" if split_name == "holdout" else "proxy"
        evaluator = evaluate_full if label_source == "full_mesh" else evaluate_proxy
        translation = tuple(translation_mm[index])
        rotation = tuple(rotation_degrees[index])
        evaluation = evaluator(translation, rotation)
        pose_id = f"lhs-{index:05d}"
        rows.append(_build_row(pose_id, translation, rotation, evaluation, label_source, split_name, pose_id, millimeters_per_blender_unit))
    return rows


def _minority_centroid(rows, minority_threshold=0.3):
    """Return ((translation_centroid, rotation_centroid), minority_fraction) for `proxy` rows."""
    proxy_rows = [row for row in rows if row["label_source"] == "proxy"]
    if not proxy_rows:
        return None, None
    valid_count = sum(1 for row in proxy_rows if row["valid"])
    invalid_count = len(proxy_rows) - valid_count
    if valid_count == 0 or invalid_count == 0:
        return None, None
    minority_is_valid = valid_count <= invalid_count
    minority_rows = [row for row in proxy_rows if row["valid"] == minority_is_valid]
    minority_fraction = len(minority_rows) / len(proxy_rows)
    if minority_fraction >= minority_threshold:
        return None, minority_fraction
    translation_centroid = np.mean([row["translation_mm"] for row in minority_rows], axis=0)
    rotation_centroid = np.mean([row["rotation_degrees"] for row in minority_rows], axis=0)
    return (translation_centroid, rotation_centroid), minority_fraction


def _generate_balance_rows(
    existing_rows, evaluate_proxy, translation_limit_mm, rotation_limit_degrees, seed, round_index, round_size, millimeters_per_blender_unit
):
    centroid, _minority_fraction = _minority_centroid(existing_rows)
    if centroid is None:
        return []
    translation_centroid, rotation_centroid = centroid
    unit_samples = sampling.latin_hypercube_unit(_stage_rng(seed, f"balance-{round_index}"), round_size, 6)
    scaled = unit_samples * 2.0 - 1.0
    # Sample a shrunk half-range box around the minority centroid, clipped to the trained domain.
    translation_mm = np.clip(
        translation_centroid + scaled[:, 0:3] * (translation_limit_mm * 0.5),
        -translation_limit_mm,
        translation_limit_mm,
    )
    rotation_degrees = np.clip(
        rotation_centroid + scaled[:, 3:6] * (rotation_limit_degrees * 0.5),
        -rotation_limit_degrees,
        rotation_limit_degrees,
    )
    rows = []
    for offset in range(round_size):
        translation = tuple(translation_mm[offset])
        rotation = tuple(rotation_degrees[offset])
        evaluation = evaluate_proxy(translation, rotation)
        pose_id = f"balance-{round_index}-{offset:05d}"
        rows.append(_build_row(pose_id, translation, rotation, evaluation, "proxy", "train", pose_id, millimeters_per_blender_unit))
    return rows


def _generate_refinement_rows(
    existing_rows, evaluate_proxy, evaluate_full, seed, round_index, max_pairs, search_subset, millimeters_per_blender_unit
):
    candidates = [row for row in existing_rows if row["label_source"] == "proxy"]
    if len(candidates) < 2:
        return []
    translation_mm = np.array([row["translation_mm"] for row in candidates])
    rotation_degrees = np.array([row["rotation_degrees"] for row in candidates])
    valid = np.array([row["valid"] for row in candidates])
    pairs = sampling.find_disagreeing_pairs(
        translation_mm, rotation_degrees, valid, _stage_rng(seed, f"refinement-{round_index}"),
        search_subset=search_subset, max_pairs=max_pairs,
    )

    rows = []
    for pair_index, (valid_index, invalid_index) in enumerate(pairs):
        valid_row = candidates[valid_index]
        invalid_row = candidates[invalid_index]
        valid_end = (valid_row["translation_mm"], valid_row["rotation_degrees"])
        invalid_end = (invalid_row["translation_mm"], invalid_row["rotation_degrees"])
        for _step in range(3):
            mid_translation, mid_rotation = sampling.bisect_pose(*valid_end, *invalid_end, 0.5)
            evaluation = evaluate_proxy(mid_translation, mid_rotation)
            if evaluation["valid"]:
                valid_end = (mid_translation, mid_rotation)
            else:
                invalid_end = (mid_translation, mid_rotation)
        # The bisection search itself can use cheap proxy geometry, but the recorded boundary
        # sample is exactly the "uncertain/boundary" case that needs an authoritative full-mesh label.
        evaluation = evaluate_full(mid_translation, mid_rotation)
        split_name = "holdout" if "holdout" in (valid_row["split"], invalid_row["split"]) else valid_row["split"]
        pose_id = f"refine-{round_index}-{pair_index:04d}"
        rows.append(
            _build_row(pose_id, mid_translation, mid_rotation, evaluation, "full_mesh", split_name, pose_id, millimeters_per_blender_unit)
        )
    return rows


def _generate_pose_label_rows(poses, evaluate_full, millimeters_per_blender_unit):
    """Label an explicit list of poses (from an active-learning candidate query) with
    full-mesh geometry. Each pose dict must have pose_id/translation_mm/rotation_degrees;
    split and group_id default to "train" and the pose_id itself."""
    rows = []
    for pose in poses:
        translation_mm = tuple(pose["translation_mm"])
        rotation_degrees = tuple(pose["rotation_degrees"])
        evaluation = evaluate_full(translation_mm, rotation_degrees)
        rows.append(
            _build_row(
                pose["pose_id"], translation_mm, rotation_degrees, evaluation, "full_mesh",
                pose.get("split", "train"), pose.get("group_id", pose["pose_id"]), millimeters_per_blender_unit,
            )
        )
    return rows


# --- Chunked, compressed, atomic checkpoint I/O -----------------------------------------------
# Mechanical array<->row conversion and NPZ chunk I/O live in `dataset_io.py` (bpy-free, shared
# with the external trainer). This module adds the schema re-validation safety net on load, since
# it has `schemas` available (bpy is present here, so the normal package import succeeds).


def _load_stage_rows(output_dir, stage, chunk_count):
    rows = dataset_io.load_stage_rows(output_dir, stage, chunk_count)
    for row in rows:
        validate_dataset_row(row)
    return rows


def _write_meta(output_dir, meta):
    dataset_io.write_meta(output_dir, meta)


def _load_or_init_meta(output_dir, config_hash):
    meta_path = output_dir / "dataset_meta.json"
    if meta_path.exists():
        meta = dataset_io.load_meta(output_dir)
        if meta.get("config_fingerprint") == config_hash:
            return meta
        logging.info("configuration changed since the last run; starting fresh in this folder")
    return {
        "dataset_schema_version": 1,
        "config_fingerprint": config_hash,
        "completed_stages": [],
        "stage_chunk_counts": {},
        "stage_row_counts": {},
        "complete": False,
    }


def _load_all_rows(output_dir, meta):
    all_rows = []
    for stage in meta["completed_stages"]:
        all_rows.extend(_load_stage_rows(output_dir, stage, meta["stage_chunk_counts"].get(stage, 0)))
    return all_rows


def _load_meta_for_pose_labeling(output_dir):
    # Pose-labeling stages append to whatever dataset already exists in output_dir (produced by
    # a prior full pipeline run and/or earlier active-learning rounds), so unlike the full
    # pipeline's config-fingerprint check, a mismatched/absent fingerprint must never reset or
    # discard already-completed stages here - only the requested stage name is (re)run.
    meta_path = output_dir / "dataset_meta.json"
    if meta_path.exists():
        return dataset_io.load_meta(output_dir)
    return {
        "dataset_schema_version": 1,
        "config_fingerprint": None,
        "completed_stages": [],
        "stage_chunk_counts": {},
        "stage_row_counts": {},
        "complete": False,
    }


def _run_stage(output_dir, meta, stage_name, chunk_size, generate_fn):
    if stage_name in meta["completed_stages"]:
        rows = _load_stage_rows(output_dir, stage_name, meta["stage_chunk_counts"].get(stage_name, 0))
        logging.info("%s: loaded %d rows from checkpoint", stage_name, len(rows))
        return rows
    started_at = time.perf_counter()
    rows = generate_fn()
    chunk_count = dataset_io.write_stage_rows(output_dir, stage_name, rows, chunk_size) if rows else 0
    meta["completed_stages"].append(stage_name)
    meta["stage_chunk_counts"][stage_name] = chunk_count
    meta["stage_row_counts"][stage_name] = len(rows)
    _write_meta(output_dir, meta)
    logging.info("%s: generated %d rows in %.1fs", stage_name, len(rows), time.perf_counter() - started_at)
    return rows


def _finalize_meta(output_dir, meta, all_rows):
    valid_count = sum(1 for row in all_rows if row["valid"])
    meta["complete"] = True
    meta["summary"] = {
        "total_rows": len(all_rows),
        "valid_rows": valid_count,
        "invalid_rows": len(all_rows) - valid_count,
        "by_label_source": {
            source: sum(1 for row in all_rows if row["label_source"] == source) for source in ("proxy", "full_mesh")
        },
        "by_split": {
            split: sum(1 for row in all_rows if row["split"] == split) for split in ("train", "validation", "holdout")
        },
    }
    _write_meta(output_dir, meta)


def _run_pose_labeling_mode(args, package, package_dir, output_dir):
    """Label exactly the poses in `args.label_poses_file` (an active-learning candidate
    query) under `args.stage_name`, appending to whatever dataset already exists in
    `output_dir` instead of running the full sampling pipeline."""
    if not args.stage_name:
        raise ValueError("--stage-name is required when using --label-poses-file")
    poses = json.loads(args.label_poses_file.read_text(encoding="utf-8"))
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{args.label_poses_file} must contain a non-empty JSON list of poses")

    meta = _load_meta_for_pose_labeling(output_dir)
    if args.stage_name not in meta["completed_stages"]:
        existing_pose_ids = set()
        for stage in meta["completed_stages"]:
            for row in _load_stage_rows(output_dir, stage, meta["stage_chunk_counts"].get(stage, 0)):
                existing_pose_ids.add(row["pose_id"])
        duplicate_ids = sorted({pose["pose_id"] for pose in poses} & existing_pose_ids)
        if duplicate_ids:
            raise ValueError(f"pose id(s) already present in dataset, refusing to duplicate: {', '.join(duplicate_ids)}")

    context = bpy.context
    context.scene.unit_settings.scale_length = package["scale_length"]
    fixed_object, moving_object = load_package_objects(package_dir, package)
    verify_fingerprints(context, package, fixed_object, moving_object)

    scale_length = package["scale_length"]
    millimeters_per_blender_unit = scale_length * 1000.0
    neutral_matrix = moving_object.matrix_world.copy()
    rotation_center_world = package["rotation_center_world"]
    pose_transform_version = package["pose_transform_version"]
    minimum_clearance = args.minimum_clearance_mm / millimeters_per_blender_unit

    fixed_full_vertices, fixed_full_polygons = full_mesh_geometry(context, fixed_object)
    moving_full_vertices, moving_full_polygons = full_mesh_geometry(context, moving_object)
    fixed_full_world_vertices, fixed_full_bvh = world_bvh(fixed_full_vertices, fixed_full_polygons, fixed_object.matrix_world)
    _, neutral_moving_full_bvh = world_bvh(moving_full_vertices, moving_full_polygons, moving_object.matrix_world)
    full_baseline_overlap_pairs = len(fixed_full_bvh.overlap(neutral_moving_full_bvh))

    evaluate_full = _make_evaluator(
        fixed_full_bvh, fixed_full_world_vertices, moving_full_vertices, moving_full_polygons,
        neutral_matrix, rotation_center_world, pose_transform_version, full_baseline_overlap_pairs, minimum_clearance, millimeters_per_blender_unit,
    )

    _run_stage(
        output_dir, meta, args.stage_name, args.chunk_size,
        lambda: _generate_pose_label_rows(poses, evaluate_full, millimeters_per_blender_unit),
    )
    _finalize_meta(output_dir, meta, _load_all_rows(output_dir, meta))
    logging.info("pose-labeling stage '%s' complete -> %s", args.stage_name, output_dir)


def main():
    args = parse_arguments()
    package_dir = args.package_dir.resolve()
    package = validate_package_v2(json.loads((package_dir / "package.json").read_text(encoding="utf-8")))
    output_dir = (args.output_dir or (package_dir / "dataset")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir)

    if args.label_poses_file is not None:
        _run_pose_labeling_mode(args, package, package_dir, output_dir)
        return

    config_hash = _config_fingerprint(package, args)
    meta = _load_or_init_meta(output_dir, config_hash)

    context = bpy.context
    context.scene.unit_settings.scale_length = package["scale_length"]
    fixed_object, moving_object = load_package_objects(package_dir, package)
    verify_fingerprints(context, package, fixed_object, moving_object)

    scale_length = package["scale_length"]
    millimeters_per_blender_unit = scale_length * 1000.0
    translation_limit_mm = package["range"]["translation_limit_mm"]
    rotation_limit_degrees = package["range"]["rotation_limit_degrees"]
    proxy_max_faces = package["sampling"]["proxy_max_faces"]
    seed = package["sampling"]["seed"]
    neutral_matrix = moving_object.matrix_world.copy()
    rotation_center_world = package["rotation_center_world"]
    pose_transform_version = package["pose_transform_version"]
    minimum_clearance = args.minimum_clearance_mm / millimeters_per_blender_unit

    fixed_proxy_vertices, fixed_proxy_polygons = proxy_geometry(context, fixed_object, proxy_max_faces)
    moving_proxy_vertices, moving_proxy_polygons = proxy_geometry(context, moving_object, proxy_max_faces)
    fixed_proxy_world_vertices, fixed_proxy_bvh = world_bvh(fixed_proxy_vertices, fixed_proxy_polygons, fixed_object.matrix_world)
    _, neutral_moving_proxy_bvh = world_bvh(moving_proxy_vertices, moving_proxy_polygons, moving_object.matrix_world)
    proxy_baseline_overlap_pairs = len(fixed_proxy_bvh.overlap(neutral_moving_proxy_bvh))

    fixed_full_vertices, fixed_full_polygons = full_mesh_geometry(context, fixed_object)
    moving_full_vertices, moving_full_polygons = full_mesh_geometry(context, moving_object)
    fixed_full_world_vertices, fixed_full_bvh = world_bvh(fixed_full_vertices, fixed_full_polygons, fixed_object.matrix_world)
    _, neutral_moving_full_bvh = world_bvh(moving_full_vertices, moving_full_polygons, moving_object.matrix_world)
    full_baseline_overlap_pairs = len(fixed_full_bvh.overlap(neutral_moving_full_bvh))

    evaluate_proxy = _make_evaluator(
        fixed_proxy_bvh, fixed_proxy_world_vertices, moving_proxy_vertices, moving_proxy_polygons,
        neutral_matrix, rotation_center_world, pose_transform_version, proxy_baseline_overlap_pairs, minimum_clearance, millimeters_per_blender_unit,
    )
    evaluate_full = _make_evaluator(
        fixed_full_bvh, fixed_full_world_vertices, moving_full_vertices, moving_full_polygons,
        neutral_matrix, rotation_center_world, pose_transform_version, full_baseline_overlap_pairs, minimum_clearance, millimeters_per_blender_unit,
    )

    all_rows = _run_stage(
        output_dir, meta, "anchors", args.chunk_size,
        lambda: _generate_anchor_rows(evaluate_full, translation_limit_mm, rotation_limit_degrees, millimeters_per_blender_unit),
    )

    lhs_count = max(0, args.sample_budget - len(all_rows))
    all_rows += _run_stage(
        output_dir, meta, "lhs", args.chunk_size,
        lambda: _generate_lhs_rows(
            evaluate_proxy, evaluate_full, translation_limit_mm, rotation_limit_degrees,
            lhs_count, seed, millimeters_per_blender_unit,
        ),
    )

    for round_index in range(args.max_balance_rounds):
        round_rows = _run_stage(
            output_dir, meta, f"balance-{round_index}", args.chunk_size,
            lambda round_index=round_index: _generate_balance_rows(
                all_rows, evaluate_proxy, translation_limit_mm, rotation_limit_degrees,
                seed, round_index, args.balance_round_size, millimeters_per_blender_unit,
            ),
        )
        all_rows += round_rows
        if not round_rows:
            break

    for round_index in range(args.refinement_rounds):
        round_rows = _run_stage(
            output_dir, meta, f"refinement-{round_index}", args.chunk_size,
            lambda round_index=round_index: _generate_refinement_rows(
                all_rows, evaluate_proxy, evaluate_full, seed, round_index,
                args.max_refinement_pairs, args.refinement_search_subset, millimeters_per_blender_unit,
            ),
        )
        all_rows += round_rows
        if not round_rows:
            break

    _finalize_meta(output_dir, meta, all_rows)
    logging.info("complete: %d rows -> %s", len(all_rows), output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("dataset generation failed")
        raise
