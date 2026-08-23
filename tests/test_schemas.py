"""Round-trip and validation tests for biomechanics_addon/schemas.py.

Runs with plain Python (no bpy, no pytest required):
    python -m unittest tests.test_schemas -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

schemas = load_addon_module("schemas")


def _identity_matrix():
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class PackageV2Test(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            collection_name="rigid",
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            neutral_fixed_matrix=_identity_matrix(),
            neutral_moving_matrix=_identity_matrix(),
            fixed_fingerprint="abc",
            moving_fingerprint="def",
            scale_length=1.0,
            translation_limit_mm=10.0,
            rotation_limit_degrees=10.0,
            seed=1234,
            proxy_max_faces=2000,
        )
        kwargs.update(overrides)
        return schemas.build_package_v2(**kwargs)

    def test_round_trip(self):
        package = self._build()
        validated = schemas.validate_package_v2(package)
        self.assertEqual(validated["package_version"], schemas.PACKAGE_SCHEMA_VERSION)
        self.assertEqual(validated["range"]["joint_radius_mm"], schemas.JOINT_RADIUS_MM)
        self.assertEqual(validated["rotation_center_world"], [0.0, 0.0, 0.0])
        self.assertEqual(validated["pose_transform_version"], 2)

    def test_preserves_explicit_rotation_center(self):
        package = self._build(rotation_center_world=[1.0, 2.0, 3.0])
        self.assertEqual(schemas.validate_package_v2(package)["rotation_center_world"], [1.0, 2.0, 3.0])

    def test_rejects_identical_objects(self):
        with self.assertRaises(schemas.SchemaError):
            self._build(moving_object_name="Sacrum")

    def test_validate_rejects_missing_key(self):
        package = self._build()
        del package["scale_length"]
        with self.assertRaises(schemas.SchemaError):
            schemas.validate_package_v2(package)

    def test_validate_rejects_wrong_version(self):
        package = self._build()
        package["package_version"] = 1
        with self.assertRaises(schemas.SchemaError):
            schemas.validate_package_v2(package)


class DatasetRowTest(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            pose_id="0001",
            translation_mm=[1.0, 2.0, 3.0],
            rotation_degrees=[0.5, -0.5, 0.0],
            valid=True,
            clearance_mm=0.75,
            overlap_pairs=0,
            label_source="proxy",
            split="train",
            group_id="cell-0001",
        )
        kwargs.update(overrides)
        return schemas.build_dataset_row(**kwargs)

    def test_round_trip(self):
        row = self._build()
        validated = schemas.validate_dataset_row(row)
        self.assertEqual(validated["dataset_version"], schemas.DATASET_SCHEMA_VERSION)

    def test_rejects_bad_label_source(self):
        with self.assertRaises(schemas.SchemaError):
            self._build(label_source="guess")

    def test_rejects_bad_split(self):
        with self.assertRaises(schemas.SchemaError):
            self._build(split="test")

    def test_rejects_negative_clearance_when_valid(self):
        with self.assertRaises(schemas.SchemaError):
            self._build(valid=True, clearance_mm=-1.0)


class ModelManifestTest(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            onnx_filename="constraint_model.onnx",
            opset=17,
            input_name="pose",
            validity_output_name="validity_logit",
            clearance_output_name="clearance_mm",
            normalization={"mean": [0.0] * 6, "scale": [1.0] * 6},
            fixed_object={"name": "Sacrum"},
            moving_object={"name": "L5"},
            fixed_fingerprint="abc",
            moving_fingerprint="def",
            scale_length=0.001,
            neutral_fixed_matrix=list(range(16)),
            neutral_moving_matrix=list(range(16)),
            translation_limit_mm=10.0,
            rotation_limit_degrees=10.0,
            validity_threshold=0.5,
            metrics={"holdout_accuracy": 0.99},
        )
        kwargs.update(overrides)
        return schemas.build_model_manifest_v1(**kwargs)

    def test_round_trip(self):
        manifest = self._build()
        validated = schemas.validate_model_manifest(manifest)
        self.assertEqual(validated["output_names"]["validity_logit"], "validity_logit")
        self.assertEqual(validated["rotation_center_world"], [3, 7, 11])
        self.assertEqual(validated["pose_transform_version"], 2)

    def test_preserves_explicit_rotation_center(self):
        manifest = self._build(rotation_center_world=[1.0, 2.0, 3.0])
        self.assertEqual(schemas.validate_model_manifest(manifest)["rotation_center_world"], [1.0, 2.0, 3.0])

    def test_rejects_bad_threshold(self):
        with self.assertRaises(schemas.SchemaError):
            self._build(validity_threshold=1.5)

    def test_accepts_conservative_threshold_boundary(self):
        manifest = self._build(validity_threshold=1.0)
        self.assertEqual(schemas.validate_model_manifest(manifest)["validity_threshold"], 1.0)

    def test_check_model_compatibility_detects_mismatch(self):
        manifest = self._build()
        reasons = schemas.check_model_compatibility(
            manifest,
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            fixed_fingerprint="changed",
            moving_fingerprint="def",
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("fixed object geometry", reasons[0])

    def test_check_model_compatibility_matches(self):
        manifest = self._build()
        reasons = schemas.check_model_compatibility(
            manifest,
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            fixed_fingerprint="abc",
            moving_fingerprint="def",
        )
        self.assertEqual(reasons, [])

    def test_check_model_compatibility_detects_scale_mismatch(self):
        manifest = self._build()
        reasons = schemas.check_model_compatibility(
            manifest,
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            fixed_fingerprint="abc",
            moving_fingerprint="def",
            scale_length=1.0,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("scale_length", reasons[0])

    def test_check_model_compatibility_detects_moved_fixed_object(self):
        manifest = self._build()
        reasons = schemas.check_model_compatibility(
            manifest,
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            fixed_fingerprint="abc",
            moving_fingerprint="def",
            fixed_matrix_world=[value + 5.0 for value in range(16)],
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("moved since training", reasons[0])

    def test_check_model_compatibility_accepts_matching_scale_and_transform(self):
        manifest = self._build()
        reasons = schemas.check_model_compatibility(
            manifest,
            fixed_object_name="Sacrum",
            moving_object_name="L5",
            fixed_fingerprint="abc",
            moving_fingerprint="def",
            scale_length=0.001,
            fixed_matrix_world=list(range(16)),
        )
        self.assertEqual(reasons, [])


class GeometryFingerprintTest(unittest.TestCase):
    def test_deterministic(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        polygons = [(0, 1, 2)]
        first = schemas.geometry_fingerprint(vertices, polygons)
        second = schemas.geometry_fingerprint(vertices, polygons)
        self.assertEqual(first, second)

    def test_sensitive_to_vertex_change(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        polygons = [(0, 1, 2)]
        moved_vertices = [(0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertNotEqual(
            schemas.geometry_fingerprint(vertices, polygons),
            schemas.geometry_fingerprint(moved_vertices, polygons),
        )


class PosePathDistanceTest(unittest.TestCase):
    def test_translation_only(self):
        distance = schemas.pose_path_distance_mm([3.0, 4.0, 0.0], 0.0)
        self.assertAlmostEqual(distance, 5.0)

    def test_rotation_only(self):
        distance = schemas.pose_path_distance_mm([0.0, 0.0, 0.0], 0.1, joint_radius_mm=50.0)
        self.assertAlmostEqual(distance, 5.0)


if __name__ == "__main__":
    unittest.main()
