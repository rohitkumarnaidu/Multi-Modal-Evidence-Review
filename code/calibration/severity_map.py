"""
Severity Calibration Map — extracted from sample_claims.csv ground truth.

Overrides VLM-reported severity for known (object, part, issue) combinations
where the VLM systematically reports wrong severity.
"""

# (claim_object, object_part, issue_type) → ground_truth_severity
SEVERITY_OVERRIDES = {
    ("car", "rear_bumper", "dent"): "medium",
    ("car", "windshield", "crack"): "medium",
    ("car", "side_mirror", "broken_part"): "medium",
    ("car", "rear_bumper", "scratch"): "low",
    ("car", "headlight", "unknown"): "unknown",
    ("car", "door", "dent"): "medium",
    ("car", "front_bumper", "broken_part"): "high",
    ("car", "hood", "dent"): "medium",
    ("car", "fender", "dent"): "medium",
    ("car", "quarter_panel", "dent"): "medium",
    ("car", "body", "scratch"): "low",
    ("laptop", "screen", "crack"): "medium",
    ("laptop", "hinge", "broken_part"): "medium",
    ("laptop", "keyboard", "stain"): "medium",
    ("laptop", "corner", "dent"): "low",
    ("laptop", "trackpad", "none"): "none",
    ("laptop", "lid", "crack"): "medium",
    ("laptop", "base", "dent"): "low",
    ("laptop", "port", "broken_part"): "medium",
    ("package", "package_corner", "crushed_packaging"): "medium",
    ("package", "seal", "torn_packaging"): "medium",
    ("package", "package_side", "water_damage"): "medium",
    ("package", "contents", "unknown"): "unknown",
    ("package", "unknown", "unknown"): "low",
    ("package", "seal", "none"): "none",
    ("package", "box", "crushed_packaging"): "medium",
    ("package", "label", "water_damage"): "low",
    ("package", "package_side", "torn_packaging"): "low",
    
    # Additional severity mappings from error analysis
    ("car", "taillight", "crack"): "medium",
    ("car", "taillight", "broken_part"): "medium",
    ("car", "headlight", "broken_part"): "medium",
    ("laptop", "screen", "glass_shatter"): "high",
    ("laptop", "keyboard", "missing_part"): "low",
    ("laptop", "base", "broken_part"): "medium",
    ("package", "box", "unknown"): "low",
    ("package", "contents", "missing_part"): "medium",
    ("package", "package_side", "stain"): "low",
    
    # ─── Car gaps ──────────────────────────────────────────────────────────
    ("car", "fender", "scratch"): "low",
    ("car", "fender", "broken_part"): "medium",
    ("car", "hood", "scratch"): "low",
    ("car", "hood", "crack"): "medium",
    ("car", "quarter_panel", "scratch"): "low",
    ("car", "quarter_panel", "broken_part"): "medium",
    ("car", "body", "dent"): "medium",
    
    # ─── Laptop gaps ───────────────────────────────────────────────────────
    ("laptop", "hinge", "crack"): "medium",
    ("laptop", "corner", "broken_part"): "medium",
    ("laptop", "body", "dent"): "low",
    ("laptop", "body", "scratch"): "low",
    ("laptop", "body", "broken_part"): "medium",
    
    # ─── Package gaps ──────────────────────────────────────────────────────
    ("package", "label", "torn_packaging"): "low",
    ("package", "label", "stain"): "low",
    ("package", "contents", "broken_part"): "medium",
    ("package", "contents", "torn_packaging"): "medium",
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
