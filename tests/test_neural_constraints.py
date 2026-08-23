"""Host-Python tests for neural pose projection helpers."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ADDON_DIR = ROOT / "biomechanics_addon"

package = types.ModuleType("biomechanics_addon")
package.__path__ = [str(ADDON_DIR)]
sys.modules.setdefault("biomechanics_addon", package)
sys.modules.setdefault("biomechanics_addon.onnx_compat", types.ModuleType("biomechanics_addon.onnx_compat"))
schemas_spec = importlib.util.spec_from_file_location("biomechanics_addon.schemas", ADDON_DIR / "schemas.py")
schemas = importlib.util.module_from_spec(schemas_spec)
schemas_spec.loader.exec_module(schemas)
sys.modules["biomechanics_addon.schemas"] = schemas
neural_spec = importlib.util.spec_from_file_location(
    "biomechanics_addon.neural_constraints", ADDON_DIR / "neural_constraints.py"
)
neural_constraints = importlib.util.module_from_spec(neural_spec)
neural_spec.loader.exec_module(neural_constraints)


class FakeSession:
    def __init__(self):
        self.run_count = 0

    def run(self, _output_names, feeds):
        self.run_count += 1
        poses = next(iter(feeds.values()))
        logits = np.where(poses[:, 0] <= 2.0, 8.0, -8.0).astype(np.float32)
        clearance = np.maximum(0.0, 2.0 - poses[:, 0]).astype(np.float32)
        return [logits, clearance]


class FakeScene:
    def __init__(self):
        self.biomechanics_training_fixed_object = types.SimpleNamespace(
            name="fixed",
            matrix_world=np.eye(4),
            data=object(),
        )
        self.biomechanics_training_moving_object = types.SimpleNamespace(
            name="moving",
            matrix_world=np.eye(4),
            data=object(),
        )
        self.unit_settings = types.SimpleNamespace(scale_length=1.0)


def _manifest():
    return {
        "input_name": "pose",
        "output_names": {"validity_logit": "validity_logit", "clearance_mm": "clearance_mm"},
        "normalization": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "validity_threshold": 0.5,
        "pose_distance_target_mm": 0.5,
        "range": {"translation_limit_mm": 10.0, "rotation_limit_degrees": 10.0, "joint_radius_mm": 50.0},
    }


class NeuralProjectionTest(unittest.TestCase):
    def test_session_loader_passes_onnx_path_not_cache_key(self):
        loaded_sources = []

        class FakeOnnxRuntime:
            @staticmethod
            def InferenceSession(source, providers):
                loaded_sources.append((source, providers))
                return object()

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.write_bytes(b"test")
            previous_module = sys.modules.get("onnxruntime")
            sys.modules["onnxruntime"] = FakeOnnxRuntime
            neural_constraints.clear_session_cache()
            try:
                neural_constraints.get_session(model_path)
            finally:
                neural_constraints.clear_session_cache()
                if previous_module is None:
                    sys.modules.pop("onnxruntime", None)
                else:
                    sys.modules["onnxruntime"] = previous_module

        self.assertEqual(loaded_sources, [(str(model_path.resolve()), ["CPUExecutionProvider"])])

    def test_valid_target_uses_one_batched_prediction(self):
        session = FakeSession()
        projected, prediction = neural_constraints.project_pose(
            session, _manifest(), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        self.assertEqual(projected, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertTrue(prediction["valid"])
        self.assertEqual(session.run_count, 1)

    def test_neutral_pose_error_rejects_model_that_cannot_start_at_neutral(self):
        class InvalidNeutralSession(FakeSession):
            def run(self, _output_names, feeds):
                poses = next(iter(feeds.values()))
                logits = np.full(len(poses), -8.0, dtype=np.float32)
                clearance = np.zeros(len(poses), dtype=np.float32)
                return [logits, clearance]

        error = neural_constraints.neutral_pose_error(InvalidNeutralSession(), _manifest())

        self.assertIn("trained neutral pose is rejected", error)

    def test_neutral_pose_error_accepts_model_that_can_start_at_neutral(self):
        self.assertIsNone(neural_constraints.neutral_pose_error(FakeSession(), _manifest()))

    def test_projection_stops_at_first_invalid_interval(self):
        projected, prediction = neural_constraints.project_pose(
            FakeSession(), _manifest(), (0.0,) * 6, (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        self.assertLessEqual(projected[0], 2.5)
        self.assertTrue(prediction["valid"])

    def test_projection_constrains_rotation(self):
        class RotationSession(FakeSession):
            def run(self, _output_names, feeds):
                self.run_count += 1
                poses = next(iter(feeds.values()))
                logits = np.where(np.abs(poses[:, 5]) <= 2.0, 8.0, -8.0).astype(np.float32)
                clearance = np.zeros(len(poses), dtype=np.float32)
                return [logits, clearance]

        projected, prediction = neural_constraints.project_pose(
            RotationSession(),
            _manifest(),
            (0.0,) * 6,
            (0.0, 0.0, 0.0, 0.0, 0.0, 10.0),
        )
        self.assertLessEqual(abs(projected[5]), 2.5)
        self.assertTrue(prediction["valid"])

    def test_rejects_out_of_range_target(self):
        with self.assertRaises(neural_constraints.PoseOutOfRangeError):
            neural_constraints.project_pose(
                FakeSession(), _manifest(), (0.0,) * 6, (11.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )

    def test_clamps_pose_to_trained_range(self):
        clamped = neural_constraints.clamp_pose_to_range(
            (11.0, -12.0, 3.0, 15.0, -20.0, 5.0),
            _manifest(),
        )
        self.assertEqual(clamped, (10.0, -10.0, 3.0, 10.0, -10.0, 5.0))

    def test_rotation_contributes_joint_arc_distance(self):
        distance = neural_constraints.pose_distance_mm(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 10.0),
            50.0,
        )
        self.assertAlmostEqual(distance, np.deg2rad(10.0) * 50.0, places=6)

    def test_scene_signature_ignores_moving_transform(self):
        scene = FakeScene()
        first = neural_constraints._model_scene_signature(scene)
        scene.biomechanics_training_moving_object.matrix_world[0, 3] = 2.0
        second = neural_constraints._model_scene_signature(scene)
        self.assertEqual(first, second)

    def test_scene_signature_changes_fixed_transform(self):
        scene = FakeScene()
        first = neural_constraints._model_scene_signature(scene)
        scene.biomechanics_training_fixed_object.matrix_world[0, 3] = 2.0
        second = neural_constraints._model_scene_signature(scene)
        self.assertNotEqual(first, second)

    def test_mesh_data_update_invalidates_model_cache(self):
        scene = FakeScene()
        scene_key = neural_constraints._scene_pointer(scene)
        neural_constraints._VALIDATED_MODEL_CACHE[scene_key] = {
            "fixed_object": scene.biomechanics_training_fixed_object,
            "moving_object": scene.biomechanics_training_moving_object,
        }
        depsgraph = types.SimpleNamespace(
            updates=[types.SimpleNamespace(id=scene.biomechanics_training_moving_object.data)]
        )
        neural_constraints.invalidate_model_cache(scene, depsgraph)
        self.assertNotIn(scene_key, neural_constraints._VALIDATED_MODEL_CACHE)

    def test_projection_does_not_jump_to_disconnected_valid_region(self):
        class DisconnectedSession(FakeSession):
            def run(self, _output_names, feeds):
                poses = next(iter(feeds.values()))
                logits = np.where(
                    (poses[:, 0] <= 2.0) | (poses[:, 0] >= 8.0),
                    8.0,
                    -8.0,
                ).astype(np.float32)
                clearance = np.zeros(len(poses), dtype=np.float32)
                return [logits, clearance]

        projected, prediction = neural_constraints.project_pose(
            DisconnectedSession(),
            _manifest(),
            (0.0,) * 6,
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertLessEqual(projected[0], 2.5)
        self.assertTrue(prediction["valid"])


if __name__ == "__main__":
    unittest.main()
