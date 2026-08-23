"""Deterministic pose sampling for background neural-constraint dataset generation.

No `bpy` dependency: pure NumPy (which Blender's embedded Python also bundles), so this
module works identically inside background Blender and in plain external Python/tests.
"""

import numpy as np


AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


def latin_hypercube_unit(rng, count, dimensions):
    """Stratified Latin Hypercube samples in [0, 1) with shape (count, dimensions)."""
    samples = np.empty((count, dimensions), dtype=np.float64)
    cut_points = np.linspace(0.0, 1.0, count + 1)
    for dimension in range(dimensions):
        jitter = rng.uniform(size=count)
        points = cut_points[:count] + jitter * (cut_points[1:] - cut_points[:count])
        rng.shuffle(points)
        samples[:, dimension] = points
    return samples


def scale_unit_samples(unit_samples, translation_limit_mm, rotation_limit_degrees):
    """Map [0, 1)^6 samples to (translation_mm, rotation_degrees), each shaped (count, 3)."""
    scaled = unit_samples * 2.0 - 1.0
    translation_mm = scaled[:, 0:3] * translation_limit_mm
    rotation_degrees = scaled[:, 3:6] * rotation_limit_degrees
    return translation_mm, rotation_degrees


def multiscale_pose_samples(rng, count, translation_limit_mm, rotation_limit_degrees):
    """Sample the full domain while reserving coverage for regional and local motion.

    A single uniform six-dimensional design becomes extremely sparse near neutral when
    rotation spans a full circle. Keep half the samples global, 30% within +/-45 degrees,
    and 20% within +/-10 degrees. Translation is narrowed with rotation so the local tier
    actually maps the interaction neighborhood instead of the corners of the translation box.
    """
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    global_count = count // 2
    regional_count = count * 3 // 10
    local_count = count - global_count - regional_count
    tiers = (
        (global_count, translation_limit_mm, rotation_limit_degrees),
        (regional_count, translation_limit_mm * 0.75, min(rotation_limit_degrees, 45.0)),
        (local_count, min(translation_limit_mm, 5.0), min(rotation_limit_degrees, 10.0)),
    )
    translations = []
    rotations = []
    for tier_count, translation_range, rotation_range in tiers:
        if tier_count <= 0:
            continue
        unit_samples = latin_hypercube_unit(rng, tier_count, 6)
        translation, rotation = scale_unit_samples(
            unit_samples,
            translation_range,
            rotation_range,
        )
        translations.append(translation)
        rotations.append(rotation)

    translation_mm = np.concatenate(translations, axis=0)
    rotation_degrees = np.concatenate(rotations, axis=0)
    order = rng.permutation(count)
    return translation_mm[order], rotation_degrees[order]


def neutral_pose():
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


def axis_anchor_poses(translation_limit_mm, rotation_limit_degrees):
    """One +/- extreme pose per axis with every other axis held at neutral."""
    anchors = []
    limits = (translation_limit_mm,) * 3 + (rotation_limit_degrees,) * 3
    for axis_index, (axis_name, limit) in enumerate(zip(AXIS_NAMES, limits)):
        for sign in (-1.0, 1.0):
            translation = [0.0, 0.0, 0.0]
            rotation = [0.0, 0.0, 0.0]
            if axis_index < 3:
                translation[axis_index] = sign * limit
            else:
                rotation[axis_index - 3] = sign * limit
            label = f"{axis_name}{'+' if sign > 0 else '-'}"
            anchors.append((tuple(translation), tuple(rotation), label))
    return anchors


def rotation_sweep_poses(rotation_limit_degrees, global_step_degrees=15.0):
    """Return full-mesh single-axis rotation anchors from neutral to the domain seam.

    Local magnitudes deliberately populate every split. Wider samples continue at a fixed
    angular step and cycle through splits, including both -180 and +180 when the domain is a
    full circle so the periodic model sees both representations of the seam orientation.
    """
    limit = float(rotation_limit_degrees)
    local_specs = (
        (1.0, "train"),
        (2.0, "validation"),
        (3.0, "train"),
        (4.0, "holdout"),
        (5.0, "train"),
        (6.0, "validation"),
        (8.0, "holdout"),
        (10.0, "train"),
    )
    magnitude_splits = {
        magnitude: split for magnitude, split in local_specs if magnitude <= limit + 1e-9
    }
    split_cycle = ("train", "validation", "holdout")
    magnitude = global_step_degrees
    global_index = 0
    while magnitude < limit - 1e-9:
        magnitude_splits.setdefault(magnitude, split_cycle[global_index % len(split_cycle)])
        magnitude += global_step_degrees
        global_index += 1
    magnitude_splits.setdefault(limit, split_cycle[global_index % len(split_cycle)])

    poses = []
    for axis_index, axis_name in enumerate(("rx", "ry", "rz")):
        for magnitude, split in sorted(magnitude_splits.items()):
            for sign in (-1.0, 1.0):
                rotation = [0.0, 0.0, 0.0]
                rotation[axis_index] = sign * magnitude
                label = f"{axis_name}{'+' if sign > 0 else '-'}-{magnitude:g}deg"
                poses.append(((0.0, 0.0, 0.0), tuple(rotation), label, split))
    return poses


def split_indices(count, rng, train_fraction=0.7, validation_fraction=0.15):
    """Deterministic (rng-driven) train/validation/holdout partition of range(count)."""
    order = rng.permutation(count)
    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)
    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "holdout": order[validation_end:],
    }


def bisect_pose(translation_a, rotation_a, translation_b, rotation_b, t):
    """Linear interpolation midpoint; adequate for the small +/-10 deg / +/-10 mm domain."""
    translation = tuple(a + (b - a) * t for a, b in zip(translation_a, translation_b))
    rotation = tuple(a + (b - a) * t for a, b in zip(rotation_a, rotation_b))
    return translation, rotation


def find_disagreeing_pairs(translation_mm, rotation_degrees, valid, rng, search_subset=500, max_pairs=200):
    """Find nearby index pairs whose `valid` label differs, within a bounded random subset."""
    count = len(valid)
    if count < 2:
        return []
    subset_size = min(count, search_subset)
    subset = rng.choice(count, size=subset_size, replace=False) if count > subset_size else np.arange(count)

    features = np.concatenate(
        [np.asarray(translation_mm)[subset], np.asarray(rotation_degrees)[subset]], axis=1
    )
    subset_valid = np.asarray(valid)[subset]

    valid_mask = subset_valid
    invalid_mask = ~subset_valid
    if not valid_mask.any() or not invalid_mask.any():
        return []

    valid_points = features[valid_mask]
    invalid_points = features[invalid_mask]
    valid_indices = subset[valid_mask]
    invalid_indices = subset[invalid_mask]

    distances = np.linalg.norm(valid_points[:, None, :] - invalid_points[None, :, :], axis=2)
    flat_order = np.argsort(distances, axis=None)[:max_pairs]
    pairs = []
    for flat_index in flat_order:
        valid_row, invalid_col = np.unravel_index(flat_index, distances.shape)
        pairs.append((int(valid_indices[valid_row]), int(invalid_indices[invalid_col])))
    return pairs
