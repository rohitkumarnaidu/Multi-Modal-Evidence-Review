from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.issue_calibration import calibrate_issue_type
from calibration.severity_map import calibrate_severity


class TestIssueCalibration:
    def test_no_override_needed(self):
        assert calibrate_issue_type("car", "door", "dent") == "dent"

    def test_windshield_glass_shatter_to_crack(self):
        assert calibrate_issue_type("car", "windshield", "glass_shatter") == "crack"

    def test_headlight_glass_shatter_to_broken_part(self):
        assert calibrate_issue_type("car", "headlight", "glass_shatter") == "broken_part"

    def test_side_mirror_dent_to_broken_part(self):
        assert calibrate_issue_type("car", "side_mirror", "dent") == "broken_part"

    def test_front_bumper_dent_is_preserved(self):
        assert calibrate_issue_type("car", "front_bumper", "dent") == "dent"

    def test_rear_bumper_broken_part_is_preserved(self):
        assert calibrate_issue_type("car", "rear_bumper", "broken_part") == "broken_part"

    def test_door_broken_part_is_preserved(self):
        assert calibrate_issue_type("car", "door", "broken_part") == "broken_part"

    def test_laptop_screen_glass_shatter_to_crack(self):
        assert calibrate_issue_type("laptop", "screen", "glass_shatter") == "crack"

    def test_laptop_screen_scratch_is_preserved(self):
        assert calibrate_issue_type("laptop", "screen", "scratch") == "scratch"

    def test_laptop_keyboard_water_damage_to_stain(self):
        assert calibrate_issue_type("laptop", "keyboard", "water_damage") == "stain"

    def test_package_contents_crush_is_preserved(self):
        assert calibrate_issue_type("package", "contents", "crushed_packaging") == "crushed_packaging"

    def test_laptop_trackpad_stain_is_preserved(self):
        assert calibrate_issue_type("laptop", "trackpad", "stain") == "stain"

    def test_laptop_body_broken_part_is_preserved(self):
        assert calibrate_issue_type("laptop", "body", "broken_part") == "broken_part"

    def test_package_seal_water_damage_is_preserved(self):
        assert calibrate_issue_type("package", "seal", "water_damage") == "water_damage"

    def test_unknown_part_is_not_calibrated(self):
        assert calibrate_issue_type("laptop", "unknown", "water_damage") == "water_damage"


class TestSeverityCalibration:
    def test_no_override_needed(self):
        assert calibrate_severity("car", "door", "dent", "medium") == "medium"

    def test_visual_severity_is_preserved(self):
        assert calibrate_severity("car", "rear_bumper", "dent", "high") == "high"

    def test_visual_crack_severity_is_preserved(self):
        assert calibrate_severity("car", "windshield", "crack", "high") == "high"

    def test_visual_broken_part_severity_is_preserved(self):
        assert calibrate_severity("car", "front_bumper", "broken_part", "medium") == "medium"

    def test_visual_screen_crack_severity_is_preserved(self):
        assert calibrate_severity("laptop", "screen", "crack", "low") == "low"

    def test_visual_package_severity_is_preserved(self):
        assert calibrate_severity("package", "package_corner", "crushed_packaging", "low") == "low"

    def test_known_visual_severity_beats_object_default(self):
        assert calibrate_severity("car", "unknown", "dent", "low") == "low"

    def test_vlm_severity_preserved_when_no_override(self):
        assert calibrate_severity("car", "fender", "stain", "low") == "low"

    def test_unknown_visual_severity_uses_general_default(self):
        assert calibrate_severity("car", "door", "dent", "unknown") == "medium"
