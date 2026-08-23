"""Bounded active-learning loop for the L5/sacrum neural constraint model (Phase 2, step 8).

Run with the EXTERNAL training environment (this workspace's `.venv`, NOT Blender's
embedded Python). Needs a Blender executable to label new candidate poses in the
background - auto-detected the same way as `tools/run_neural_training.ps1`, or pass
`--blender-exe` / set the `BLENDER_EXE` environment variable.

    .venv/Scripts/python.exe tools/active_learning_loop.py --package-dir <export_dir> \\
        [--rounds 5] [--candidates-per-round 4000] [--max-new-samples-per-round 60] \\
        [--max-total-new-samples 300] [--uncertainty-band 0.15] [--max-runtime-seconds 3600] \\
        [--train-arg=--epochs=300 ...]

Each round: (re)train `tools/train_constraint_model.py` on the current dataset, score a
fresh Latin-hypercube candidate pool with the resulting ONNX model, select the poses
whose predicted validity probability is closest to the calibrated threshold (within
`--uncertainty-band`), invoke background Blender to label ONLY those poses with
full-mesh geometry (`tools/generate_neural_dataset.py --label-poses-file ...`), and
merge them into the dataset as a new stage named `active-round-N`. A round is skipped
early once `--max-total-new-samples` or `--max-runtime-seconds` is reached, or as soon
as a round finds no candidates in the uncertainty band (treated as convergence). A
final retrain always runs after the loop so the published model bundle in
`--model-dir` reflects every labeled round.

Stage-level resume is inherited from `tools/generate_neural_dataset.py`: rerunning this
loop after an interruption reuses already-labeled `active-round-N` stages from disk
(matched by stage name, not by the freshly-sampled candidate pool) rather than
relabeling them.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _addon_import import load_addon_module  # noqa: E402 - path setup must run first

schemas = load_addon_module("schemas")
sampling = load_addon_module("sampling")

GENERATE_DATASET_SCRIPT = TOOLS_DIR / "generate_neural_dataset.py"
TRAIN_SCRIPT = TOOLS_DIR / "train_constraint_model.py"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=4000)
    parser.add_argument("--max-new-samples-per-round", type=int, default=60)
    parser.add_argument("--max-total-new-samples", type=int, default=300)
    parser.add_argument(
        "--uncertainty-band", type=float, default=0.15,
        help="Select candidates whose predicted validity probability is within this distance of the calibrated threshold.",
    )
    parser.add_argument("--minimum-clearance-mm", type=float, default=0.5, help="Must match the value used to build the dataset.")
    parser.add_argument("--max-runtime-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blender-exe", type=Path, default=None)
    parser.add_argument(
        "--train-arg", action="append", default=[],
        help="Extra argument forwarded verbatim to train_constraint_model.py (repeatable), e.g. --train-arg=--epochs=150.",
    )
    return parser.parse_args()


def _stage_rng(seed, stage_name):
    stage_seed = int(hashlib.sha256(f"{seed}:{stage_name}".encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(stage_seed)


def _find_blender_executable(explicit_path):
    if explicit_path is not None:
        return explicit_path
    env_path = os.environ.get("BLENDER_EXE")
    if env_path:
        return Path(env_path)
    for candidate in ("blender", "blender-launcher.exe"):
        found = shutil.which(candidate)
        if found:
            return Path(found)

    search_patterns = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation" / "Blender *" / "blender.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam" / "steamapps" / "common" / "Blender" / "blender.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Blender Foundation" / "Blender *" / "blender.exe",
        Path(os.environ.get("USERPROFILE", "")) / "scoop" / "apps" / "blender" / "current" / "blender.exe",
    ]
    matches = [match for pattern in search_patterns for match in sorted(pattern.parent.glob(pattern.name))]
    if matches:
        return sorted(matches, key=lambda path: str(path))[-1]

    store_alias = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "blender-launcher.exe"
    if store_alias.is_file():
        return store_alias

    raise FileNotFoundError("Blender was not found. Add blender.exe to PATH, set BLENDER_EXE, or pass --blender-exe.")


def _run_training(args, dataset_dir, model_dir, seed):
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--package-dir", str(args.package_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(model_dir),
        "--seed", str(seed),
        *args.train_arg,
    ]
    print(f"[active-learning] training: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"train_constraint_model.py failed with exit code {result.returncode}")


def _load_model_bundle(model_dir):
    manifest = schemas.validate_model_manifest(json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8")))
    session = onnxruntime.InferenceSession(str(model_dir / manifest["onnx_filename"]), providers=["CPUExecutionProvider"])
    return manifest, session


def _score_candidates(manifest, session, translation_limit_mm, rotation_limit_degrees, count, rng):
    unit_samples = sampling.latin_hypercube_unit(rng, count, 6)
    translation_mm, rotation_degrees = sampling.scale_unit_samples(unit_samples, translation_limit_mm, rotation_limit_degrees)
    pose = np.concatenate([translation_mm, rotation_degrees], axis=1).astype(np.float32)
    output_names = manifest["output_names"]
    validity_logit, _clearance_mm = session.run(
        [output_names["validity_logit"], output_names["clearance_mm"]], {manifest["input_name"]: pose}
    )
    probability = 1.0 / (1.0 + np.exp(-validity_logit))
    return translation_mm, rotation_degrees, probability


def _select_uncertain(translation_mm, rotation_degrees, probability, threshold, band, max_count):
    """Poses with predicted validity probability closest to the calibrated threshold
    first, capped at `max_count` and restricted to within `band` of the threshold."""
    distance_to_threshold = np.abs(probability - threshold)
    order = np.argsort(distance_to_threshold)
    selected_indices = [index for index in order if distance_to_threshold[index] <= band][:max_count]
    return [(tuple(translation_mm[index]), tuple(rotation_degrees[index])) for index in selected_indices]


def _write_poses_file(path, poses, stage_name):
    payload = [
        {"pose_id": f"{stage_name}-{index:05d}", "translation_mm": list(translation), "rotation_degrees": list(rotation), "split": "train"}
        for index, (translation, rotation) in enumerate(poses)
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_labeling(blender_exe, package_dir, dataset_dir, poses_file, stage_name, minimum_clearance_mm):
    command = [
        str(blender_exe), "--background", "--factory-startup", "--python", str(GENERATE_DATASET_SCRIPT), "--",
        "--package-dir", str(package_dir), "--output-dir", str(dataset_dir),
        "--label-poses-file", str(poses_file), "--stage-name", stage_name,
        "--minimum-clearance-mm", str(minimum_clearance_mm),
    ]
    print(f"[active-learning] labeling: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"background Blender labeling failed with exit code {result.returncode} (see dataset_run.log)")


def main():
    args = parse_arguments()
    package_dir = args.package_dir.resolve()
    package = schemas.validate_package_v2(json.loads((package_dir / "package.json").read_text(encoding="utf-8")))
    dataset_dir = (args.dataset_dir or (package_dir / "dataset")).resolve()
    model_dir = (args.model_dir or (package_dir / "model")).resolve()
    seed = args.seed if args.seed is not None else package["sampling"]["seed"]
    blender_exe = _find_blender_executable(args.blender_exe)

    translation_limit_mm = package["range"]["translation_limit_mm"]
    rotation_limit_degrees = package["range"]["rotation_limit_degrees"]

    started_at = time.perf_counter()
    total_new_samples = 0
    labeled_any_round = False

    for round_index in range(1, args.rounds + 1):
        if args.max_runtime_seconds is not None and time.perf_counter() - started_at > args.max_runtime_seconds:
            print(f"[active-learning] stopping: exceeded --max-runtime-seconds={args.max_runtime_seconds}")
            break
        remaining_budget = args.max_total_new_samples - total_new_samples
        if remaining_budget <= 0:
            print(f"[active-learning] stopping: reached --max-total-new-samples={args.max_total_new_samples}")
            break

        _run_training(args, dataset_dir, model_dir, seed)
        manifest, session = _load_model_bundle(model_dir)

        rng = _stage_rng(seed, f"active-round-{round_index}")
        translation_mm, rotation_degrees, probability = _score_candidates(
            manifest, session, translation_limit_mm, rotation_limit_degrees, args.candidates_per_round, rng
        )
        round_max_samples = min(args.max_new_samples_per_round, remaining_budget)
        selected = _select_uncertain(
            translation_mm, rotation_degrees, probability, manifest["validity_threshold"], args.uncertainty_band, round_max_samples
        )
        if not selected:
            print(f"[active-learning] round {round_index}: no candidates within the uncertainty band; stopping (converged)")
            break

        stage_name = f"active-round-{round_index}"
        poses_file = dataset_dir / f"{stage_name}_poses.json"
        poses_payload = _write_poses_file(poses_file, selected, stage_name)
        _run_labeling(blender_exe, package_dir, dataset_dir, poses_file, stage_name, args.minimum_clearance_mm)

        total_new_samples += len(poses_payload)
        labeled_any_round = True
        print(f"[active-learning] round {round_index}: labeled {len(poses_payload)} poses (total new so far: {total_new_samples})")

    if labeled_any_round:
        print("[active-learning] final retrain on the merged dataset")
        _run_training(args, dataset_dir, model_dir, seed)

    print(f"[active-learning] complete: model bundle -> {model_dir}")


if __name__ == "__main__":
    main()
