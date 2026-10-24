"""Tests for biomechanics_addon/chain_solver.py (pure NumPy, no bpy).

Run: python -m unittest tests.test_chain_solver -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

cs = load_addon_module("chain_solver")
df = load_addon_module("distance_field")

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _box_field(half=(0.01, 0.01, 0.01)):
    origin, voxel, shape = df.grid_spec(-np.array(half), np.array(half), 0.001, 0.006)
    points = df.grid_points(origin, voxel, shape)
    exact = df.box_sdf(points, half).reshape(shape)
    return df.DistanceField(origin, voxel, df.sign_distances(np.abs(exact), exact, voxel))


class QuaternionTest(unittest.TestCase):
    def test_matrix_round_trip(self):
        q = np.array([0.8, 0.1, -0.5, 0.3])
        q /= np.linalg.norm(q)
        back = cs.matrix_to_quat(cs.quat_to_matrix(q))
        self.assertTrue(np.allclose(q, back, atol=1e-9) or np.allclose(q, -back, atol=1e-9))


class DistanceConstraintTest(unittest.TestCase):
    def test_free_body_is_pulled_to_rest_length(self):
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0.05, 0, 0], IDENTITY, mass=1.0, radius=0.01)
        constraint = cs.DistanceConstraint(fixed, [0, 0, 0], free, [0, 0, 0], rest_length=0.02)
        solver = cs.ChainSolver([fixed, free], [constraint], [], iterations=10)
        solver.solve()
        self.assertAlmostEqual(constraint.current_length(), 0.02, places=6)

    def test_unilateral_constraint_ignores_compression_and_allows_slack(self):
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0.01, 0, 0], IDENTITY)
        slack_constraint = cs.DistanceConstraint(fixed, [0, 0, 0], free, [0, 0, 0], 0.02, unilateral=True, slack=0.1)
        cs.ChainSolver([fixed, free], [slack_constraint], [], iterations=5).solve()
        self.assertAlmostEqual(free.position[0], 0.01)  # compressed: untouched
        free.position[:] = [0.05, 0, 0]
        cs.ChainSolver([fixed, free], [slack_constraint], [], iterations=10).solve()
        self.assertAlmostEqual(slack_constraint.current_length(), 0.022, places=6)

    def test_compliance_leaves_residual_stretch_under_a_pose_pull(self):
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0.02, 0, 0], IDENTITY)
        ligament = cs.DistanceConstraint(fixed, [0, 0, 0], free, [0, 0, 0], 0.02, compliance=1.0, unilateral=True)
        pull = cs.PoseConstraint(free, [0.03, 0, 0], IDENTITY, position_compliance=1.0, max_position_step=1.0)
        cs.ChainSolver([fixed, free], [ligament], [], [pull], iterations=10, substeps=30).solve()
        # Equal compliances: across substeps the body settles halfway between rest and target.
        self.assertAlmostEqual(ligament.current_length(), 0.025, places=4)


class GravityTest(unittest.TestCase):
    def test_hanging_body_sags_until_the_spring_holds_it(self):
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0, 0, -0.02], IDENTITY)
        spring = cs.DistanceConstraint(fixed, [0, 0, 0], free, [0, 0, 0], 0.02, compliance=0.5)
        solver = cs.ChainSolver([fixed, free], [spring], [], iterations=10, substeps=1, gravity=(0, 0, -0.001))
        for _solve in range(40):
            solver.solve()
        self.assertLess(free.position[2], -0.0201)  # sagged
        self.assertGreater(free.position[2], -0.03)  # but held
        self.assertTrue(np.allclose(fixed.position, 0.0))

    def test_gravity_does_not_move_a_body_resting_on_a_box(self):
        field = _box_field()
        samples = np.array([[0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0], [0, 0, 0.01], [0, 0, -0.01]])
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0, 0, 0.02], IDENTITY, mass=1.0, radius=0.01)
        pair = cs.CollisionPair(free, samples, field, fixed, samples, field)
        solver = cs.ChainSolver([fixed, free], [], [pair], iterations=10, substeps=2, gravity=(0, 0, -0.0005))
        for _solve in range(20):
            solver.solve()
        self.assertAlmostEqual(free.position[2], 0.02, places=3)


class CollisionTest(unittest.TestCase):
    def test_overlapping_boxes_are_separated(self):
        field = _box_field()
        samples = np.array([[0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0], [0, 0, 0.01], [0, 0, -0.01]])
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0.015, 0, 0], IDENTITY, mass=1.0, radius=0.01)
        pair = cs.CollisionPair(free, samples, field, fixed, samples, field)
        solver = cs.ChainSolver([fixed, free], [], [pair], iterations=10)
        solver.solve()
        self.assertGreaterEqual(free.position[0], 0.02 - 1e-4)
        self.assertLess(solver.max_penetration(), 1e-3)

    def test_target_inside_bone_stops_at_the_surface(self):
        field = _box_field()
        samples = np.array([[0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0], [0, 0, 0.01], [0, 0, -0.01]])
        fixed = cs.Body("fixed", [0, 0, 0], IDENTITY, fixed=True)
        free = cs.Body("free", [0.03, 0, 0], IDENTITY, mass=1.0, radius=0.01)
        pair = cs.CollisionPair(free, samples, field, fixed, samples, field)
        pull = cs.PoseConstraint(free, [0.0, 0, 0], IDENTITY, position_compliance=1e-3)
        solver = cs.ChainSolver([fixed, free], [], [pair], [pull], iterations=20, substeps=2)
        solver.solve()
        # The handle walks 1.5 mm per sweep toward the target and stops on the box face.
        self.assertGreater(free.position[0], 0.019)
        self.assertLess(free.position[0], 0.021)


if __name__ == "__main__":
    unittest.main()
