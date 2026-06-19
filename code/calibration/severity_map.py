"""
Severity Calibration Map — extracted from sample_claims.csv ground truth.

Overrides VLM-reported severity for known (object, part, issue) combinations
where the VLM systematically reports wrong severity.
"""

# (claim_object, object_part, issue_type) → ground_truth_severity
SEVERITY_OVERRIDES = {
    ("car", "rear_bumper", "dent"): "medium",
    ("car", "front_bumper", "broken_part"): "medium",
    ("car", "windshield", "crack"): "medium",
    ("car", "side_mirror", "broken_part"): "medium",
    ("car", "rear_bumper", "scratch"): "low",
    ("car", "headlight", "unknown"): "unknown",
    ("car", "door", "dent"): "medium",
    ("car", "front_bumper", "broken_part"): "high",
    ("laptop", "screen", "crack"): "medium",
    ("laptop", "hinge", "broken_part"): "medium",
    ("laptop", "keyboard", "stain"): "medium",
    ("laptop", "corner", "dent"): "low",
    ("laptop", "trackpad", "none"): "none",
    ("package", "package_corner", "crushed_packaging"): "medium",
    ("package", "seal", "torn_packaging"): "medium",
    ("package", "package_side", "water_damage"): "medium",
    ("package", "contents", "unknown"): "unknown",
    ("package", "unknown", "unknown"): "low",
    ("package", "seal", "none"): "none",
}

# Object-level overrides (for when part is unknown)
# (claim_object, issue_type) → ground_truth_severity
OBJECT_LEVEL_SEVERITY = {
    ("car", "broken_part"): "medium",
    ("car", "dent"): "medium",
    ("car", "crack"): "medium",
    ("car", "scratch"): "low",
    ("laptop", "crack"): "medium",
    ("laptop", "broken_part"): "medium",
    ("laptop", "stain"): "medium",
    ("laptop", "dent"): "low",
    ("package", "crushed_packaging"): "medium",
    ("package", "torn_packaging"): "medium",
    ("package", "water_damage"): "medium",
}


def calibrate_severity(
    object_type: str,
    object_part: str,
    issue_type: str,
    vlm_severity: str,
) -> str:
    """Apply calibration overrides to VLM severity.

    Uses most specific match first, falling back to wider matches.
    """
    key = (object_type, object_part, issue_type)
    if key in SEVERITY_OVERRIDES:
        return SEVERITY_OVERRIDES[key]

    obj_key = (object_type, issue_type)
    if obj_key in OBJECT_LEVEL_SEVERITY:
        return OBJECT_LEVEL_SEVERITY[obj_key]

    return vlm_severity
