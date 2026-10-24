"""Tests for biomechanics_addon/ligament_atlas.py (pure Python, no bpy).

Run: python -m unittest tests.test_ligament_atlas -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _addon_import import load_addon_module  # noqa: E402

atlas = load_addon_module("muscle_atlas")
ligaments = load_addon_module("ligament_atlas")

SCENE_BONES = (
    set(atlas.SPINE)
    | {f"T{level:02d} rib {side}" for level in range(1, 13) for side in atlas.SIDES}
    | {"strenum", "sacrum", "ilium L", "ilium R", "femur L", "femur R"}
)


def _point_bone(point):
    return point[0]


class LigamentAtlasTest(unittest.TestCase):
    def setUp(self):
        self.strings = ligaments.ligament_strings()

    def test_every_string_has_mechanics_and_scene_bones(self):
        for string in self.strings:
            self.assertIn(string["kind"], ("ligament", "disc"))
            self.assertGreater(string["compliance"], 0.0)
            self.assertIn(string["unilateral"], (True, False))
            self.assertGreaterEqual(len(string["points"]), 2)
            for point in string["points"]:
                self.assertIn(_point_bone(point), SCENE_BONES, string["label"])
                landmark = point[1]
                if isinstance(landmark, str):
                    atlas.landmark_spec(_point_bone(point), landmark)
                else:
                    self.assertIn("direction", landmark)
                    near = landmark.get("near")
                    if near is not None:
                        self.assertIn(near, SCENE_BONES)
                    nearest = landmark.get("nearest_to")
                    if nearest is not None:
                        self.assertIn(nearest[0], SCENE_BONES)

    def test_discs_are_bilateral_springs_with_stiffer_center(self):
        nucleus = [s for s in self.strings if s["group"] == "disc_nucleus"]
        annulus = [s for s in self.strings if s["group"] == "disc_annulus"]
        self.assertEqual(len(nucleus), 17)  # T01-T02 ... L4-L5 and L5-S1
        self.assertEqual(len(annulus), 17 * ligaments.RING_POINTS)
        for string in nucleus + annulus:
            self.assertFalse(string["unilateral"])
        self.assertLess(nucleus[0]["compliance"], annulus[0]["compliance"])

    def test_ligaments_resist_tension_only(self):
        for string in self.strings:
            if string["kind"] == "ligament":
                self.assertTrue(string["unilateral"], string["label"])

    def test_keys_are_unique(self):
        keys = [atlas.string_key(string) for string in self.strings]
        self.assertEqual(len(keys), len(set(keys)))

    def test_chain_levels(self):
        order = ligaments.SPINE
        self.assertEqual(order[0], "T01")
        self.assertEqual(order[-1], "L5")


if __name__ == "__main__":
    unittest.main()
