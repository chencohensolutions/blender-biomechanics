import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "biomechanics_addon" / "neural_training.py"
SPEC = importlib.util.spec_from_file_location("neural_training_test_module", MODULE_PATH)
NEURAL_TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NEURAL_TRAINING)


class NeuralTrainingHelpersTests(unittest.TestCase):
    def test_atomic_status_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "neural_training_status.json"
            payload = {"state": "completed", "exit_code": 0, "run_id": "test"}
            NEURAL_TRAINING.write_json_atomic(status_path, payload)
            self.assertEqual(NEURAL_TRAINING.read_status(status_path), payload)
            self.assertFalse(status_path.with_name(f".{status_path.name}.{__import__('os').getpid()}.tmp").exists())

    def test_malformed_status_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "neural_training_status.json"
            status_path.write_text("not json", encoding="utf-8")
            status = NEURAL_TRAINING.read_status(status_path)
            self.assertEqual(status["state"], "unknown")
            self.assertIn("Could not read training status", status["error"])

    def test_build_command_preserves_paths_with_spaces(self):
        command = NEURAL_TRAINING.build_command(
            Path(r"C:\Training Runs\sacrum L5"),
            module_file=MODULE_PATH,
            blender_path=Path(r"C:\Program Files\Blender\blender.exe"),
            python_path=Path(r"C:\Training Runs\.venv\Scripts\python.exe"),
            sample_budget=10000,
            active_learning=True,
            active_rounds=2,
            max_total_new_samples=25,
        )
        self.assertEqual(command[0:4], ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass"])
        self.assertIn("-ActiveLearning", command)
        self.assertEqual(command[command.index("-SampleBudget") + 1], "10000")
        self.assertEqual(command[command.index("-PackagePath") + 1], str(Path(r"C:\Training Runs\sacrum L5").resolve()))
        rendered = NEURAL_TRAINING.format_command(command)
        self.assertIn('"C:\\Training Runs\\sacrum L5"', rendered)
        self.assertIn('"C:\\Program Files\\Blender\\blender.exe"', rendered)

    def test_model_bundle_requires_manifest_and_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self.assertFalse(NEURAL_TRAINING.model_bundle_is_complete(model_dir))
            (model_dir / "constraint_model.onnx").write_bytes(b"partial")
            self.assertFalse(NEURAL_TRAINING.model_bundle_is_complete(model_dir))
            (model_dir / "model_manifest.json").write_text(json.dumps({}), encoding="utf-8")
            self.assertTrue(NEURAL_TRAINING.model_bundle_is_complete(model_dir))

    def test_training_package_directory_is_pair_specific_and_safe(self):
        root = Path(r"C:\Training Runs")
        self.assertEqual(
            NEURAL_TRAINING.training_package_directory(root, "sacrum", "L5"),
            root.resolve() / "sacrum__L5",
        )
        self.assertEqual(
            NEURAL_TRAINING.training_package_directory(root, "ilium/R", "femur:L"),
            root.resolve() / "ilium_R__femur_L",
        )

    def test_discovers_legacy_and_pair_model_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_model = root / "model"
            pair_model = root / "sacrum__L5" / "model"
            for model_dir in (legacy_model, pair_model):
                model_dir.mkdir(parents=True)
                (model_dir / "model_manifest.json").write_text("{}", encoding="utf-8")
                (model_dir / "constraint_model.onnx").write_bytes(b"model")
            incomplete_model = root / "L5__L4" / "model"
            incomplete_model.mkdir(parents=True)
            (incomplete_model / "model_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                NEURAL_TRAINING.discover_model_manifests(root),
                [
                    (legacy_model / "model_manifest.json").resolve(),
                    (pair_model / "model_manifest.json").resolve(),
                ],
            )

    def test_migrates_complete_legacy_bundle_to_its_pair_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_model = root / "model"
            legacy_model.mkdir()
            manifest = {
                "fixed_object": {"name": "L5"},
                "moving_object": {"name": "L4"},
            }
            (legacy_model / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (legacy_model / "constraint_model.onnx").write_bytes(b"model")
            (legacy_model / "metrics.json").write_text("{}", encoding="utf-8")

            migrated = NEURAL_TRAINING.migrate_legacy_model_bundle(root)

            destination = root / "L5__L4" / "model"
            self.assertEqual(migrated, destination / "model_manifest.json")
            self.assertFalse(legacy_model.exists())
            self.assertEqual(
                NEURAL_TRAINING.discover_model_manifests(root),
                [destination / "model_manifest.json"],
            )

    def test_legacy_migration_does_not_overwrite_existing_pair_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_model = root / "model"
            destination = root / "L5__L4" / "model"
            for model_dir in (legacy_model, destination):
                model_dir.mkdir(parents=True)
                (model_dir / "model_manifest.json").write_text(
                    json.dumps({"fixed_object": {"name": "L5"}, "moving_object": {"name": "L4"}}),
                    encoding="utf-8",
                )
                (model_dir / "constraint_model.onnx").write_bytes(b"model")

            self.assertIsNone(NEURAL_TRAINING.migrate_legacy_model_bundle(root))
            self.assertTrue(legacy_model.is_dir())
            self.assertTrue(destination.is_dir())

    def test_completed_model_manifest_requires_success_and_full_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            status = {"state": "completed", "exit_code": 0, "model_dir": str(model_dir)}
            self.assertIsNone(NEURAL_TRAINING.completed_model_manifest(status))
            (model_dir / "constraint_model.onnx").write_bytes(b"model")
            (model_dir / "model_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                NEURAL_TRAINING.completed_model_manifest(status),
                model_dir.resolve() / "model_manifest.json",
            )
            self.assertIsNone(
                NEURAL_TRAINING.completed_model_manifest({**status, "exit_code": 1})
            )


if __name__ == "__main__":
    unittest.main()
