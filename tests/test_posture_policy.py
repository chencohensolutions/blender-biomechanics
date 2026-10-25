"""Tests for posture_policy.py and chain_package.py (pure NumPy, no bpy).

Run: python -m unittest tests.test_posture_policy -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

cs = load_addon_module("chain_solver")
pp = load_addon_module("posture_policy")
cp = load_addon_module("chain_package")
df = load_addon_module("distance_field")

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _toy_solver():
    base = cs.Body("base", [0, 0, 0], IDENTITY, fixed=True)
    mid = cs.Body("mid", [0, 0, 0.04], IDENTITY, radius=0.03)
    top = cs.Body("top", [0, 0, 0.08], IDENTITY, radius=0.03)
    constraints = [
        cs.DistanceConstraint(base, [0, 0, 0.02], mid, [0, 0, -0.02], 0.0, compliance=0.0, mode="compression", name="strut1"),
        cs.DistanceConstraint(mid, [0, 0, 0.02], top, [0, 0, -0.02], 0.0, compliance=0.0, mode="compression", name="strut2"),
        cs.DistanceConstraint(base, [0.01, 0, 0.02], mid, [0.01, 0, -0.02], 0.0, compliance=0.05, name="ring1"),
        cs.DistanceConstraint(mid, [0.01, 0, 0.02], top, [0.01, 0, -0.02], 0.0, compliance=0.05, name="ring2"),
    ]
    muscles = [
        cs.MuscleElement(base, [-0.02, 0, 0], top, [-0.02, 0, 0], 0, 0.0005, name="back"),
        cs.MuscleElement(base, [0.02, 0, 0], top, [0.02, 0, 0], 1, 0.0005, name="front"),
    ]
    return cs.ChainSolver([base, mid, top], constraints, [], iterations=4, substeps=1, gravity=(0, 0, -0.0002), muscles=muscles, group_count=2)


class ObservationTest(unittest.TestCase):
    def test_observation_is_zero_at_rest_and_tracks_motion(self):
        solver = _toy_solver()
        state = pp.ChainState(solver, 1.0 / 1000.0)
        self.assertEqual(state.size, 12)
        self.assertTrue(np.allclose(state.observe(), 0.0))
        solver.bodies[2].position[2] -= 0.01  # 1 cm down
        observation = state.observe()
        self.assertAlmostEqual(observation[5], -1.0, places=6)  # top body z offset in cm

    def test_static_and_network_controllers_produce_activations(self):
        static = pp.static_controller(np.array([0.0, 100.0]))
        activations = static(None)
        self.assertAlmostEqual(activations[0], 0.5)
        self.assertAlmostEqual(activations[1], 1.0)
        policy = pp.Policy(12, 2, hidden=8)
        rng = np.random.default_rng(0)
        controller = policy.controller(rng.standard_normal(policy.size))
        output = controller(np.zeros(12))
        self.assertEqual(output.shape, (2,))
        self.assertTrue(np.all((output >= 0.0) & (output <= 1.0)))

    def test_policy_round_trip_and_fit_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, pp.POLICY_FILE)
            pp.save_policy(path, "static", np.array([1.0, -1.0]), 0.5, 12, 2, 8, 20, 0.3)
            policy = pp.load_policy(path)
            self.assertEqual(policy["mode"], "static")
            controller = pp.controller_from_policy(policy, 12, 2)
            self.assertAlmostEqual(controller(None)[0], pp.sigmoid(1.0))
            with self.assertRaises(ValueError):
                pp.controller_from_policy(policy, 12, 3)


class PackageRoundTripTest(unittest.TestCase):
    def test_saved_package_rebuilds_an_equivalent_solver(self):
        solver = _toy_solver()

        class FakeSession:
            pass

        session = FakeSession()
        session.solver = solver
        session.bodies = {body.name: body for body in solver.bodies}
        session.centers = {body.name: np.zeros(3) for body in solver.bodies}
        session.muscle_groups = [("back", "M"), ("front", "M")]
        meta, arrays = cp.package_from_session(
            session, {"base": "base", "top": "top", "units_per_mm": 0.001, "field_paths": {}, "group_labels": ["a", "b"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            cp.save_package(directory, meta, arrays)
            loaded_meta, loaded_arrays = cp.load_package(directory)
            rebuilt = cp.build_solver(loaded_meta, loaded_arrays)
        self.assertEqual(len(rebuilt.c_a), 4)
        self.assertEqual(len(rebuilt.m_a), 2)
        self.assertEqual(rebuilt.distance_constraints[0].mode, "compression")
        for original, copy in ((solver, rebuilt),):
            original.activations[:] = [1.0, 0.0]
            copy.activations[:] = [1.0, 0.0]
            for _solve in range(5):
                original.solve()
                copy.solve()
        self.assertTrue(np.allclose(solver.bodies[2].position, rebuilt.bodies[2].position, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
