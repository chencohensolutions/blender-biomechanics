"""Tests for biomechanics_addon/distance_field.py (pure NumPy, no bpy).

Run: python -m unittest tests.test_distance_field -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

df = load_addon_module("distance_field")


def _box_field(half_extent=(10.0, 6.0, 4.0), voxel=1.0, padding=5.0):
    origin, voxel, shape = df.grid_spec(-np.array(half_extent), np.array(half_extent), voxel, padding)
    points = df.grid_points(origin, voxel, shape)
    unsigned = np.abs(df.box_sdf(points, half_extent)).reshape(shape)
    # A fake normal test: the exact sign is known, so use it directly for the band.
    normal_dot = df.box_sdf(points, half_extent).reshape(shape)
    values = df.sign_distances(unsigned, normal_dot, voxel)
    return df.DistanceField(origin, voxel, values), half_extent


class GridSpecTest(unittest.TestCase):
    def test_coarsens_until_cell_budget_fits(self):
        origin, voxel, shape = df.grid_spec((0, 0, 0), (100, 100, 100), 1.0, 0.0, max_cells=200_000)
        self.assertLessEqual(int(np.prod(shape)), 200_000)
        self.assertGreater(voxel, 1.0)
        self.assertTrue(np.allclose(origin, 0.0))


class SignTest(unittest.TestCase):
    def test_flood_fill_marks_interior_negative_and_exterior_positive(self):
        field, half = _box_field()
        inside = field.sample([[0.0, 0.0, 0.0], [5.0, 2.0, 1.0]])
        outside = field.sample([[14.0, 0.0, 0.0], [0.0, 0.0, 8.0]])
        self.assertTrue(np.all(inside < 0.0))
        self.assertTrue(np.all(outside > 0.0))

    def test_interior_sign_survives_a_small_hole_in_the_band(self):
        half = (10.0, 6.0, 4.0)
        origin, voxel, shape = df.grid_spec(-np.array(half), np.array(half), 1.0, 5.0)
        points = df.grid_points(origin, voxel, shape)
        exact = df.box_sdf(points, half).reshape(shape)
        unsigned = np.abs(exact)
        # Punch a one-cell hole through the +x face: those cells read as far from any surface.
        ix = int(round((half[0] - origin[0]) / voxel))
        iy = int(round((0.0 - origin[1]) / voxel))
        iz = int(round((0.0 - origin[2]) / voxel))
        unsigned[ix, iy, iz] = 3.0
        values = df.sign_distances(unsigned, exact, voxel)
        field = df.DistanceField(origin, voxel, values)
        self.assertLess(field.sample([[0.0, 0.0, 0.0]])[0], 0.0)


class SampleTest(unittest.TestCase):
    def test_trilinear_matches_exact_box_distance_near_surface(self):
        field, half = _box_field()
        rng = np.random.default_rng(3)
        points = rng.uniform(-12.0, 12.0, size=(500, 3))
        exact = df.box_sdf(points, half)
        sampled = field.sample(points)
        near = np.abs(exact) < 3.0
        self.assertLess(np.abs(sampled[near] - exact[near]).max(), 0.35)

    def test_outside_grid_distance_keeps_growing(self):
        field, _half = _box_field()
        near_edge = field.sample([[20.0, 0.0, 0.0]])[0]
        far = field.sample([[40.0, 0.0, 0.0]])[0]
        self.assertGreater(far, near_edge + 15.0)

    def test_gradient_points_away_from_the_box(self):
        field, _half = _box_field()
        _distances, gradient = field.sample_with_gradient([[12.0, 0.0, 0.0], [0.0, 0.0, -6.0]])
        self.assertGreater(gradient[0][0], 0.95)
        self.assertLess(gradient[1][2], -0.95)

    def test_npz_round_trip(self):
        field, _half = _box_field()
        restored = df.DistanceField.from_npz_dict(field.to_npz_dict())
        points = [[1.0, 2.0, 3.0]]
        self.assertAlmostEqual(float(restored.sample(points)[0]), float(field.sample(points)[0]))


if __name__ == "__main__":
    unittest.main()
