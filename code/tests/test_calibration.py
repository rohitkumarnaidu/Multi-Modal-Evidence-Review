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

    def test_front_bumper_dent_to_broken_part(self):
        assert calibrate_issue_type("car", "front_bumper", "dent") == "broken_part"

    def test_rear_bumper_broken_part_to_dent(self):
        assert calibrate_issue_type("car", "rear_bumper", "broken_part") == "dent"

    def test_door_broken_part_to_dent(self):
        assert calibrate_issue_type("car", "door", "broken_part") == "dent"

    def test_laptop_screen_glass_shatter_to_crack(self):
        assert calibrate_issue_type("laptop", "screen", "glass_shatter") == "crack"

    def test_laptop_screen_scratch_to_crack(self):
        assert calibrate_issue_type("laptop", "screen", "scratch") == "crack"

    def test_laptop_keyboard_water_damage_to_stain(self):
        assert calibrate_issue_type("laptop", "keyboard", "water_damage") == "stain"

    def test_package_contents_crushed_to_missing_part(self):
        assert calibrate_issue_type("package", "contents", "crushed_packaging") == "missing_part"

    def test_laptop_trackpad_stain_to_none(self):
        assert calibrate_issue_type("laptop", "trackpad", "stain") == "none"

    def test_laptop_body_broken_part_to_dent(self):
        assert calibrate_issue_type("laptop", "body", "broken_part") == "dent"

    def test_package_seal_water_damage_to_torn_packaging(self):
        assert calibrate_issue_type("package", "seal", "water_damage") == "torn_packaging"

    def test_unknown_object_fallback(self):
        assert calibrate_issue_type("laptop", "unknown", "water_damage") == "stain"


class TestSeverityCalibration:
    def test_no_override_needed(self):
        assert calibrate_severity("car", "door", "dent", "medium") == "medium"

    def test_rear_bumper_dent_overrides_to_medium(self):
        assert calibrate_severity("car", "rear_bumper", "dent", "high") == "medium"

    def test_windshield_crack_overrides_to_medium(self):
        assert calibrate_severity("car", "windshield", "crack", "high") == "medium"

    def test_front_bumper_broken_part_overrides_to_high(self):
        assert calibrate_severity("car", "front_bumper", "broken_part", "medium") == "high"

    def test_laptop_screen_crack_overrides_to_medium(self):
        assert calibrate_severity("laptop", "screen", "crack", "low") == "medium"

    def test_package_corner_crushed_overrides_to_medium(self):
        assert calibrate_severity("package", "package_corner", "crushed_packaging", "low") == "medium"

    def test_object_level_fallback(self):
        assert calibrate_severity("car", "unknown", "dent", "low") == "medium"

    def test_vlm_severity_preserved_when_no_override(self):
        assert calibrate_severity("car", "fender", "stain", "low") == "low"

    def test_door_dent_overrides_to_medium(self):
        assert calibrate_severity("car", "door", "dent", "low") == "medium"
