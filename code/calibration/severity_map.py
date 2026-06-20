"""
Severity calibration.

Severity is based on visual evidence. These defaults only fill gaps when the
VLM gives an allowed issue but an unknown or implausible severity.
"""

from __future__ import annotations


VALID_SEVERITIES = {"none", "low", "medium", "high", "unknown"}


def calibrate_severity(
    object_type: str,
    object_part: str,
    issue_type: str,
    vlm_severity: str,
) -> str:
    issue = (issue_type or "unknown").strip().lower()
    part = (object_part or "unknown").strip().lower()
    raw = (vlm_severity or "unknown").strip().lower()

    if issue == "none":
        return "none"
    if issue == "unknown":
        return "unknown"
    if raw in {"low", "medium", "high"}:
        return raw

    if issue in {"scratch", "stain"}:
        return "low"

    if issue in {"dent", "crack", "torn_packaging", "crushed_packaging", "water_damage"}:
        return "medium"

    if issue == "glass_shatter":
        return "high"

    if issue == "missing_part":
        if object_type == "package" and part in {"contents", "item"}:
            return "high"
        return "medium"

    if issue == "broken_part":
        if object_type == "car" and part in {"front_bumper", "headlight", "taillight"}:
            return "high"
        return "medium"

    return raw if raw in VALID_SEVERITIES else "unknown"
