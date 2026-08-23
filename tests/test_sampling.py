"""Tests for biomechanics_addon/sampling.py (pure NumPy, no bpy).

Run: python -m unittest tests.test_sampling -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

sampling = load_addon_module("sampling")


class LatinHypercubeTest(unittest.TestCase):
    def test_shape_and_bounds(self):
        rng = np.random.default_rng(1)
        samples = sampling.latin_hypercube_unit(rng, 20, 6)
        self.assertEqual(samples.shape, (20, 6))
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples < 1.0))

    def test_stratification_covers_each_bin(self):
        rng = np.random.default_rng(2)
        count = 10
        samples = sampling.latin_hypercube_unit(rng, count, 1)
        bins = np.floor(samples[:, 0] * count).astype(int)
        self.assertEqual(sorted(bins.tolist()), list(range(count)))

    def test_deterministic_given_seed(self):
        first = sampling.latin_hypercube_unit(np.random.default_rng(42), 15, 6)
        second = sampling.latin_hypercube_unit(np.random.default_rng(42), 15, 6)
        np.testing.assert_array_equal(first, second)


class ScaleUnitSamplesTest(unittest.TestCase):
    def test_scaling_range(self):
        unit_samples = np.array([[0.0, 0.5, 1.0, 0.0, 0.5, 1.0]])
        translation_mm, rotation_degrees = sampling.scale_unit_samples(unit_samples, 10.0, 10.0)
        np.testing.assert_allclose(translation_mm[0], [-10.0, 0.0, 10.0])
        np.testing.assert_allclose(rotation_degrees[0], [-10.0, 0.0, 10.0])


class AxisAnchorPosesTest(unittest.TestCase):
    def test_twelve_anchors(self):
        anchors = sampling.axis_anchor_poses(10.0, 10.0)
        self.assertEqual(len(anchors), 12)
        labels = {label for _t, _r, label in anchors}
        self.assertEqual(len(labels), 12)

    def test_single_axis_extreme(self):
        anchors = sampling.axis_anchor_poses(10.0, 10.0)
        translation, rotation, label = next(a for a in anchors if a[2] == "tx+")
        self.assertEqual(translation, (10.0, 0.0, 0.0))
        self.assertEqual(rotation, (0.0, 0.0, 0.0))


class RotationSweepPosesTest(unittest.TestCase):
    def test_full_circle_includes_local_and_seam_samples_on_each_axis(self):
        poses = sampling.rotation_sweep_poses(180.0)
        for axis in range(3):
            axis_values = {
                rotation[axis]
                for _translation, rotation, _label, _split in poses
                if sum(value != 0.0 for value in rotation) == 1
            }
            self.assertTrue({-180.0, -10.0, -5.0, -2.0, -1.0}.issubset(axis_values))
            self.assertTrue({1.0, 2.0, 5.0, 10.0, 180.0}.issubset(axis_values))

    def test_local_sweep_populates_every_split(self):
        poses = sampling.rotation_sweep_poses(180.0)
        local_splits = {
            split
            for _translation, rotation, _label, split in poses
            if max(abs(value) for value in rotation) <= 10.0
        }
        self.assertEqual(local_splits, {"train", "validation", "holdout"})


class MultiscalePoseSamplesTest(unittest.TestCase):
    def test_full_domain_keeps_dense_local_rotation_coverage(self):
        translation, rotation = sampling.multiscale_pose_samples(
            np.random.default_rng(9),
            1000,
            translation_limit_mm=10.0,
            rotation_limit_degrees=180.0,
        )
        self.assertEqual(translation.shape, (1000, 3))
        self.assertEqual(rotation.shape, (1000, 3))
        self.assertTrue(np.all(np.abs(rotation) <= 180.0))
        local = np.all(np.abs(rotation) <= 10.0, axis=1)
        self.assertGreaterEqual(int(local.sum()), 200)
        self.assertTrue(np.any(np.abs(rotation) > 170.0))


class SplitIndicesTest(unittest.TestCase):
    def test_partition_covers_all_indices_without_overlap(self):
        rng = np.random.default_rng(7)
        splits = sampling.split_indices(100, rng)
        all_indices = np.concatenate(list(splits.values()))
        self.assertEqual(sorted(all_indices.tolist()), list(range(100)))

    def test_fractions_approximate(self):
        rng = np.random.default_rng(7)
        splits = sampling.split_indices(1000, rng, train_fraction=0.7, validation_fraction=0.15)
        self.assertEqual(len(splits["train"]), 700)
        self.assertEqual(len(splits["validation"]), 150)
        self.assertEqual(len(splits["holdout"]), 150)


class BisectPoseTest(unittest.TestCase):
    def test_midpoint(self):
        translation, rotation = sampling.bisect_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 10.0), 0.5)
        self.assertEqual(translation, (5.0, 0.0, 0.0))
        self.assertEqual(rotation, (0.0, 0.0, 5.0))


class FindDisagreeingPairsTest(unittest.TestCase):
    def test_finds_nearest_opposite_label_pair(self):
        translation_mm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        rotation_degrees = np.zeros((3, 3))
        valid = np.array([True, False, False])
        rng = np.random.default_rng(3)
        pairs = sampling.find_disagreeing_pairs(translation_mm, rotation_degrees, valid, rng, max_pairs=1)
        self.assertEqual(len(pairs), 1)
        valid_index, invalid_index = pairs[0]
        self.assertEqual(valid_index, 0)
        self.assertEqual(invalid_index, 1)

    def test_no_pairs_when_all_same_label(self):
        translation_mm = np.zeros((5, 3))
        rotation_degrees = np.zeros((5, 3))
        valid = np.array([True] * 5)
        rng = np.random.default_rng(4)
        pairs = sampling.find_disagreeing_pairs(translation_mm, rotation_degrees, valid, rng)
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
