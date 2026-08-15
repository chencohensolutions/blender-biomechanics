"""External PyTorch trainer for the L5/sacrum neural constraint model (Phase 3).

Run with the EXTERNAL training environment (this workspace's `.venv`, NOT Blender's
embedded Python):

    .venv/Scripts/python.exe tools/train_constraint_model.py \\
        --package-dir <export_dir> --dataset-dir <export_dir>/dataset \\
        --output-dir <export_dir>/model

Reads a dataset written by `tools/generate_neural_dataset.py` (chunked NPZ rows +
`dataset_meta.json`) and the `package.json` it was generated from, trains a small
CPU-friendly MLP with a validity-logit head and a nonnegative-clearance head,
calibrates a conservative validity threshold on the full-mesh holdout split, and
publishes an ONNX model bundle (`constraint_model.onnx` + `model_manifest.json` +
`metrics.json`) after a PyTorch-vs-ONNX-Runtime parity check.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _addon_import import load_addon_module  # noqa: E402 - path setup must run first

schemas = load_addon_module("schemas")
dataset_io = load_addon_module("dataset_io")

POSE_COLUMNS = ("tx_mm", "ty_mm", "tz_mm", "rx_deg", "ry_deg", "rz_deg")
SPLIT_PRECEDENCE = {"holdout": 2, "validation": 1, "train": 0}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--leakage-radius-mm", type=float, default=2.0)
    parser.add_argument("--full-mesh-authority-weight", type=float, default=3.0)
    parser.add_argument("--target-false-safe-rate", type=float, default=0.0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--parity-tolerance", type=float, default=1e-3)
    return parser.parse_args()


# --- Loading and validation --------------------------------------------------------------------


def load_inputs(args):
    package = schemas.validate_package_v2(json.loads((args.package_dir / "package.json").read_text(encoding="utf-8")))
    dataset_dir = args.dataset_dir or (args.package_dir / "dataset")
    rows, meta = dataset_io.load_dataset(dataset_dir)
    if not rows:
        raise ValueError(f"No dataset rows found in {dataset_dir}")
    for row in rows:
        schemas.validate_dataset_row(row)
    return package, rows, meta


# --- Leakage-safe split reassignment ------------------------------------------------------------


class _UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first, second):
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[first_root] = second_root


def reassign_splits_for_leakage(rows, radius_mm, joint_radius_mm):
    """Group poses within `radius_mm` combined pose-distance of each other (translation
    magnitude plus a small-angle rotation-arc approximation) and promote every row in a
    group to the group's most conservative split (holdout > validation > train). This
    prevents a bisection-refined boundary pose (or any two coincidentally nearby samples)
    from leaking near-duplicate information across the train/validation/holdout boundary.
    `group_id` in the dataset schema is currently 1 row per group (not a real multi-row
    link), so this proximity clustering is done independently rather than trusting it.
    """
    translations = np.array([row["translation_mm"] for row in rows], dtype=np.float64)
    rotations_radians = np.radians(np.array([row["rotation_degrees"] for row in rows], dtype=np.float64))

    translation_distance = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1)
    rotation_distance = np.linalg.norm(rotations_radians[:, None, :] - rotations_radians[None, :, :], axis=-1)
    combined_distance = translation_distance + joint_radius_mm * rotation_distance

    union_find = _UnionFind(len(rows))
    close_pairs = np.argwhere(np.triu(combined_distance <= radius_mm, k=1))
    for first, second in close_pairs:
        union_find.union(int(first), int(second))

    groups = {}
    for index in range(len(rows)):
        groups.setdefault(union_find.find(index), []).append(index)

    reassigned_count = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        most_conservative = max((rows[index]["split"] for index in indices), key=lambda split: SPLIT_PRECEDENCE[split])
        for index in indices:
            if rows[index]["split"] != most_conservative:
                rows[index]["split"] = most_conservative
                reassigned_count += 1
    return rows, reassigned_count


# --- Array construction -------------------------------------------------------------------------


def build_arrays(rows):
    pose = np.array(
        [list(row["translation_mm"]) + list(row["rotation_degrees"]) for row in rows], dtype=np.float32
    )
    valid = np.array([row["valid"] for row in rows], dtype=np.float32)
    clearance_mm = np.array([row["clearance_mm"] for row in rows], dtype=np.float32)
    non_overlapping = np.array([row["overlap_pairs"] == 0 for row in rows], dtype=bool)
    is_full_mesh = np.array([row["label_source"] == "full_mesh" for row in rows], dtype=bool)
    split = np.array([row["split"] for row in rows])
    return {
        "pose": pose,
        "valid": valid,
        "clearance_mm": clearance_mm,
        "non_overlapping": non_overlapping,
        "is_full_mesh": is_full_mesh,
        "split": split,
    }


def split_mask(arrays, split_name):
    return arrays["split"] == split_name


def sample_weights(valid, is_full_mesh, authority_weight):
    valid_count = float(valid.sum())
    invalid_count = float(len(valid) - valid_count)
    if valid_count == 0 or invalid_count == 0:
        class_weight = np.ones_like(valid)
    else:
        total = valid_count + invalid_count
        class_weight = np.where(valid > 0.5, total / (2.0 * valid_count), total / (2.0 * invalid_count))
    return class_weight * np.where(is_full_mesh, authority_weight, 1.0)


# --- Model ---------------------------------------------------------------------------------------


class ConstraintModel(nn.Module):
    def __init__(self, hidden_size, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.trunk = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.validity_head = nn.Linear(hidden_size, 1)
        self.clearance_head = nn.Linear(hidden_size, 1)

    def forward(self, pose):
        normalized = (pose - self.mean) / self.std
        features = self.trunk(normalized)
        validity_logit = self.validity_head(features).squeeze(-1)
        clearance_mm = nn.functional.softplus(self.clearance_head(features)).squeeze(-1)
        return validity_logit, clearance_mm


def _compute_losses(model, pose, valid, clearance_mm, clearance_mask, weight):
    validity_logit, predicted_clearance = model(pose)
    validity_loss = (nn.functional.binary_cross_entropy_with_logits(validity_logit, valid, reduction="none") * weight).mean()
    clearance_error = nn.functional.smooth_l1_loss(predicted_clearance, clearance_mm, reduction="none")
    masked_weight = clearance_mask * weight
    denominator = masked_weight.sum().clamp(min=1e-8)
    clearance_loss = (clearance_error * masked_weight).sum() / denominator
    return validity_loss, clearance_loss, validity_logit, predicted_clearance


def train_model(model, train_tensors, validation_tensors, epochs, batch_size, learning_rate, weight_decay, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_count = train_tensors["pose"].shape[0]

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    loss_curve = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_count, generator=generator)
        epoch_train_loss = 0.0
        for start in range(0, train_count, batch_size):
            batch_indices = permutation[start:start + batch_size]
            validity_loss, clearance_loss, _, _ = _compute_losses(
                model,
                train_tensors["pose"][batch_indices],
                train_tensors["valid"][batch_indices],
                train_tensors["clearance_mm"][batch_indices],
                train_tensors["clearance_mask"][batch_indices],
                train_tensors["weight"][batch_indices],
            )
            loss = validity_loss + clearance_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += float(loss.detach()) * len(batch_indices)
        epoch_train_loss /= train_count

        model.eval()
        with torch.no_grad():
            validity_loss, clearance_loss, _, _ = _compute_losses(
                model,
                validation_tensors["pose"],
                validation_tensors["valid"],
                validation_tensors["clearance_mm"],
                validation_tensors["clearance_mask"],
                validation_tensors["weight"],
            )
            epoch_validation_loss = float(validity_loss + clearance_loss)

        loss_curve.append({"epoch": epoch, "train_loss": epoch_train_loss, "validation_loss": epoch_validation_loss})
        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, loss_curve, best_validation_loss


# --- Threshold calibration and metrics -----------------------------------------------------------


def calibrate_threshold(probabilities, actual_valid, target_false_safe_rate):
    """Pick the smallest (most lenient) threshold whose false-safe rate (predicting valid
    when the pose is actually invalid) does not exceed `target_false_safe_rate`, prioritizing
    false-safe errors (dangerous) over false-invalid errors (merely overly conservative)."""
    invalid_mask = actual_valid < 0.5
    invalid_count = int(invalid_mask.sum())
    candidates = np.unique(np.clip(np.concatenate([probabilities, [0.0, 1.0]]), 0.0, 1.0))
    best_threshold = 1.0
    best_false_safe_rate = 0.0
    satisfied = False
    for threshold in np.sort(candidates):
        predicted_valid = probabilities >= threshold
        false_safe_count = int(np.logical_and(predicted_valid, invalid_mask).sum())
        false_safe_rate = false_safe_count / invalid_count if invalid_count else 0.0
        if false_safe_rate <= target_false_safe_rate:
            best_threshold = float(threshold)
            best_false_safe_rate = false_safe_rate
            satisfied = True
            break
    if not satisfied:
        # No threshold reaches the target; fall back to the most conservative one available.
        best_threshold = 1.0
        predicted_valid = probabilities >= best_threshold
        false_safe_count = int(np.logical_and(predicted_valid, invalid_mask).sum())
        best_false_safe_rate = false_safe_count / invalid_count if invalid_count else 0.0
    return best_threshold, best_false_safe_rate, satisfied


def confusion_matrix(probabilities, actual_valid, threshold):
    predicted_valid = probabilities >= threshold
    actual_valid_bool = actual_valid > 0.5
    true_positive = int(np.logical_and(predicted_valid, actual_valid_bool).sum())
    false_positive = int(np.logical_and(predicted_valid, ~actual_valid_bool).sum())
    true_negative = int(np.logical_and(~predicted_valid, ~actual_valid_bool).sum())
    false_negative = int(np.logical_and(~predicted_valid, actual_valid_bool).sum())
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


# --- ONNX export and parity -----------------------------------------------------------------------


def export_onnx(model, opset, onnx_path):
    model.eval()
    dummy_input = torch.zeros(1, 6, dtype=torch.float32)
    temporary_path = onnx_path.with_name(onnx_path.name + ".tmp")
    torch.onnx.export(
        model,
        dummy_input,
        str(temporary_path),
        input_names=["pose"],
        output_names=["validity_logit", "clearance_mm"],
        dynamic_axes={"pose": {0: "batch"}, "validity_logit": {0: "batch"}, "clearance_mm": {0: "batch"}},
        opset_version=opset,
    )
    # torch's exporter may write weights to a companion "<name>.tmp.data" file; reload with
    # external data inlined and re-save fully self-contained so the bundle is a single .onnx
    # file (a rename of the .onnx alone would otherwise leave a dangling external-data reference).
    exported_model = onnx.load(str(temporary_path), load_external_data=True)
    onnx.checker.check_model(exported_model)
    onnx.save_model(exported_model, str(temporary_path), save_as_external_data=False)
    companion_data_path = temporary_path.with_name(temporary_path.name + ".data")
    companion_data_path.unlink(missing_ok=True)
    temporary_path.replace(onnx_path)


def onnx_parity_test(model, onnx_path, translation_limit_mm, rotation_limit_degrees, seed):
    model.eval()
    generator = np.random.default_rng(seed)
    random_poses = generator.uniform(
        low=[-translation_limit_mm] * 3 + [-rotation_limit_degrees] * 3,
        high=[translation_limit_mm] * 3 + [rotation_limit_degrees] * 3,
        size=(64, 6),
    )
    extreme_poses = np.array(
        [
            [0.0] * 6,
            [translation_limit_mm, 0, 0, 0, 0, 0],
            [-translation_limit_mm, 0, 0, 0, 0, 0],
            [0, 0, 0, rotation_limit_degrees, 0, 0],
            [0, 0, 0, 0, 0, -rotation_limit_degrees],
            [translation_limit_mm] * 3 + [rotation_limit_degrees] * 3,
            [-translation_limit_mm] * 3 + [-rotation_limit_degrees] * 3,
        ]
    )
    poses = np.concatenate([random_poses, extreme_poses]).astype(np.float32)

    with torch.no_grad():
        torch_validity, torch_clearance = model(torch.from_numpy(poses))

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_validity, onnx_clearance = session.run(["validity_logit", "clearance_mm"], {"pose": poses})

    max_diff = max(
        float(np.max(np.abs(torch_validity.numpy() - onnx_validity))),
        float(np.max(np.abs(torch_clearance.numpy() - onnx_clearance))),
    )
    return max_diff


# --- Bundle assembly -------------------------------------------------------------------------------


def write_bundle(output_dir, model, package, threshold, mean, std, metrics, opset, parity_tolerance):
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "constraint_model.onnx"
    export_onnx(model, opset, onnx_path)

    max_diff = onnx_parity_test(
        model, onnx_path, package["range"]["translation_limit_mm"], package["range"]["rotation_limit_degrees"],
        metrics["config"]["seed"],
    )
    metrics["onnx_parity_max_abs_diff"] = max_diff
    if max_diff > parity_tolerance:
        # Do not publish metrics.json/model_manifest.json for a bundle that failed parity -
        # model_manifest.json's presence is the "this bundle is complete and trustworthy" signal.
        raise RuntimeError(
            f"PyTorch/ONNX parity check failed: max abs diff {max_diff} exceeds tolerance {parity_tolerance}. "
            f"Refusing to publish the bundle at {output_dir} (onnx file written but manifest withheld)."
        )

    metrics_path = output_dir / "metrics.json"
    metrics_temp = metrics_path.with_name(metrics_path.name + ".tmp")
    metrics_temp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_temp.replace(metrics_path)

    manifest = schemas.build_model_manifest_v1(
        onnx_filename=onnx_path.name,
        opset=opset,
        input_name="pose",
        validity_output_name="validity_logit",
        clearance_output_name="clearance_mm",
        normalization={"order": list(POSE_COLUMNS), "mean": mean.tolist(), "std": std.tolist()},
        fixed_object={"name": package["fixed_object"]["name"]},
        moving_object={"name": package["moving_object"]["name"]},
        fixed_fingerprint=package["fixed_object"]["fingerprint"],
        moving_fingerprint=package["moving_object"]["fingerprint"],
        scale_length=package["scale_length"],
        neutral_fixed_matrix=package["fixed_object"]["matrix_world"],
        neutral_moving_matrix=package["moving_object"]["matrix_world"],
        translation_limit_mm=package["range"]["translation_limit_mm"],
        rotation_limit_degrees=package["range"]["rotation_limit_degrees"],
        validity_threshold=threshold,
        metrics=metrics["holdout_full_mesh"],
        joint_radius_mm=package["range"]["joint_radius_mm"],
    )
    schemas.validate_model_manifest(manifest)
    manifest_path = output_dir / "model_manifest.json"
    manifest_temp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Write model_manifest.json last: its presence signals a complete, consistent bundle.
    manifest_temp.replace(manifest_path)
    return manifest_path, max_diff


# --- Main ------------------------------------------------------------------------------------------


def main():
    args = parse_arguments()
    package, rows, meta = load_inputs(args)
    seed = args.seed if args.seed is not None else package["sampling"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows, reassigned_count = reassign_splits_for_leakage(rows, args.leakage_radius_mm, package["range"]["joint_radius_mm"])
    print(f"loaded {len(rows)} rows; reassigned {reassigned_count} rows to prevent train/validation/holdout leakage")

    arrays = build_arrays(rows)
    train_mask = split_mask(arrays, "train")
    validation_mask = split_mask(arrays, "validation")
    holdout_mask = split_mask(arrays, "holdout")
    for name, mask in (("train", train_mask), ("validation", validation_mask), ("holdout", holdout_mask)):
        if not mask.any():
            raise ValueError(f"'{name}' split is empty after leakage-safe grouping; increase --sample-budget")

    mean = arrays["pose"][train_mask].mean(axis=0)
    std = arrays["pose"][train_mask].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    def to_tensors(mask):
        weight = sample_weights(arrays["valid"][mask], arrays["is_full_mesh"][mask], args.full_mesh_authority_weight)
        return {
            "pose": torch.from_numpy(arrays["pose"][mask]),
            "valid": torch.from_numpy(arrays["valid"][mask]),
            "clearance_mm": torch.from_numpy(arrays["clearance_mm"][mask]),
            "clearance_mask": torch.from_numpy(arrays["non_overlapping"][mask].astype(np.float32)),
            "weight": torch.from_numpy(weight.astype(np.float32)),
        }

    train_tensors = to_tensors(train_mask)
    validation_tensors = to_tensors(validation_mask)

    model = ConstraintModel(args.hidden_size, mean, std)
    started_at = time.perf_counter()
    model, loss_curve, best_validation_loss = train_model(
        model, train_tensors, validation_tensors,
        args.epochs, args.batch_size, args.learning_rate, args.weight_decay, seed,
    )
    print(f"trained {args.epochs} epochs in {time.perf_counter() - started_at:.1f}s, best validation loss {best_validation_loss:.4f}")

    holdout_full_mesh_mask = holdout_mask & arrays["is_full_mesh"]
    if not holdout_full_mesh_mask.any():
        raise ValueError("no full_mesh holdout rows available for threshold calibration")
    model.eval()
    with torch.no_grad():
        holdout_logit, _ = model(torch.from_numpy(arrays["pose"][holdout_full_mesh_mask]))
    holdout_probabilities = torch.sigmoid(holdout_logit).numpy()
    holdout_actual_valid = arrays["valid"][holdout_full_mesh_mask]

    threshold, false_safe_rate, satisfied = calibrate_threshold(
        holdout_probabilities, holdout_actual_valid, args.target_false_safe_rate
    )
    if not satisfied:
        print(
            f"warning: could not reach target false-safe rate {args.target_false_safe_rate}; "
            f"using most conservative threshold {threshold} (false-safe rate {false_safe_rate:.4f})"
        )
    holdout_full_mesh_metrics = {
        "threshold": threshold,
        "false_safe_rate": false_safe_rate,
        "sample_count": int(holdout_full_mesh_mask.sum()),
        "confusion_matrix": confusion_matrix(holdout_probabilities, holdout_actual_valid, threshold),
    }

    with torch.no_grad():
        all_holdout_logit, all_holdout_clearance = model(torch.from_numpy(arrays["pose"][holdout_mask]))
    all_holdout_probabilities = torch.sigmoid(all_holdout_logit).numpy()
    clearance_mask = arrays["non_overlapping"][holdout_mask]
    clearance_mae = None
    if clearance_mask.any():
        clearance_mae = float(
            np.mean(np.abs(all_holdout_clearance.numpy()[clearance_mask] - arrays["clearance_mm"][holdout_mask][clearance_mask]))
        )

    metrics = {
        "dataset_summary": meta.get("summary", {}),
        "reassigned_rows_for_leakage": reassigned_count,
        "loss_curve": loss_curve,
        "best_validation_loss": best_validation_loss,
        "holdout_full_mesh": holdout_full_mesh_metrics,
        "holdout_all_sources": {
            "sample_count": int(holdout_mask.sum()),
            "confusion_matrix": confusion_matrix(all_holdout_probabilities, arrays["valid"][holdout_mask], threshold),
            "clearance_mae_mm": clearance_mae,
        },
        "config": vars(args) | {"seed": seed},
    }
    metrics["config"] = {key: (str(value) if isinstance(value, Path) else value) for key, value in metrics["config"].items()}

    output_dir = args.output_dir or (args.package_dir / "model")
    manifest_path, max_diff = write_bundle(
        output_dir, model, package, threshold, mean, std, metrics, args.opset, args.parity_tolerance
    )
    print(f"published model bundle at {manifest_path} (validity_threshold={threshold}, onnx parity max diff={max_diff:.2e})")


if __name__ == "__main__":
    main()
