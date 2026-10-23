"""Tests for biomechanics_addon/muscle_atlas.py (pure Python, no bpy).

Run: python -m unittest tests.test_muscle_atlas -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

atlas = load_addon_module("muscle_atlas")

SCENE_BONES = (
    set(atlas.SPINE)
    | {f"T{level:02d} rib {side}" for level in range(1, 13) for side in atlas.SIDES}
    | {"strenum", "sacrum", "ilium L", "ilium R", "femur L", "femur R"}
)


class MuscleAtlasTest(unittest.TestCase):
    def setUp(self):
        self.strings = atlas.muscle_strings()

    def test_every_point_names_a_scene_bone_and_known_landmark(self):
        for string in self.strings:
            self.assertGreaterEqual(len(string["points"]), 2)
            for bone, landmark in string["points"]:
                self.assertIn(bone, SCENE_BONES, string)
                atlas.landmark_spec(bone, landmark)

    def test_string_keys_are_unique(self):
        keys = [atlas.string_key(string) for string in self.strings]
        self.assertEqual(len(keys), len(set(keys)))

    def test_paired_sides_mirror_and_stay_on_their_side(self):
        by_key = {atlas.string_key(string): string for string in self.strings}
        for (muscle, side, index), string in by_key.items():
            if side == atlas.MIDLINE:
                continue
            other = "R" if side == "L" else "L"
            self.assertIn((muscle, other, index), by_key)
            for bone, _landmark in string["points"]:
                self.assertFalse(bone.endswith(f" {other}"), string)

    def test_counts_match_index_range(self):
        for string in self.strings:
            self.assertLess(string["index"], string["count"])
            self.assertIn(string["group"], atlas.GROUPS)


if __name__ == "__main__":
    unittest.main()
