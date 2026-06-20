"""
Issue type calibration.

These are domain-level normalizations for common VLM wording differences.
They deliberately avoid row- or sample-specific overrides: the visual issue
must still come from the image analysis.
"""

from __future__ import annotations


GLASS_PARTS = {"windshield", "screen"}
LIGHT_OR_MIRROR_PARTS = {"headlight", "taillight", "side_mirror"}
PACKAGE_SURFACE_PARTS = {"box", "package_corner", "package_side"}


def calibrate_issue_type(
    object_type: str,
    object_part: str,
    vlm_issue_type: str,
) -> str:
    """Apply broad, explainable issue normalizations."""
    issue = (vlm_issue_type or "unknown").strip().lower()
    part = (object_part or "unknown").strip().lower()

    if issue in ("none", "unknown"):
        return issue

    if part in GLASS_PARTS and issue in {"glass_shatter", "broken_part"}:
        return "crack"

    if object_type == "car" and part in LIGHT_OR_MIRROR_PARTS:
        if issue in {"dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part"}:
            return "missing_part" if issue == "missing_part" else "broken_part"

    if object_type == "package":
        if part == "seal" and issue in {"crack", "broken_part", "scratch"}:
            return "torn_packaging"
        if part in PACKAGE_SURFACE_PARTS and issue == "broken_part":
            return "crushed_packaging"
        if part in {"contents", "item"} and issue in {"missing_part", "broken_part"}:
            return "missing_part"

    if object_type == "laptop":
        if part == "hinge" and issue in {"crack", "dent", "broken_part"}:
            return "broken_part"
        if part == "keyboard" and issue == "water_damage":
            return "stain"
        if part == "trackpad" and issue == "scratch":
            return "none"

    return issue
