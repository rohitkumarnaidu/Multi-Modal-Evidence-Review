"""
Part Calibration — VLM close-part confusions identified from sample run.

VLM systematically confuses these parts. Instead of fighting the model,
we calibrate its output using known confusion pairs.

Confusion sources (from 20 sample claims, 8 part errors):
  1. lid ↔ almost anything on laptop (VLM defaults to 'lid' for any laptop enclosure area)
  2. package_side ↔ seal (VLM can't distinguish sealing tape from side panel)
  3. hood ↔ front_bumper (VLM can't distinguish close-up front bumper from hood)
  4. body ↔ trackpad (VLM doesn't recognize trackpad as a distinct part)
"""

# (claim_object, claimed_part, vlm_part) -> corrected_part
PART_OVERRIDES = {
    # Laptop: VLM defaults to "lid" for any top-down laptop shot
    ("laptop", "screen", "lid"): "screen",
    ("laptop", "hinge", "lid"): "hinge",
    ("laptop", "corner", "lid"): "corner",

    # Package: VLM says package_side for seal images
    ("package", "seal", "package_side"): "seal",
    ("package", "seal", "box"): "seal",
    ("package", "package_side", "package_corner"): "package_side",

    # Car: VLM says hood for front bumper close-ups
    ("car", "front_bumper", "hood"): "front_bumper",
    ("car", "front_bumper", "body"): "front_bumper",
    ("car", "headlight", "door"): "headlight",
    ("car", "headlight", "side_mirror"): "headlight",
    ("car", "side_mirror", "door"): "side_mirror",
    ("car", "side_mirror", "body"): "side_mirror",
    ("car", "taillight", "front_bumper"): "taillight",
    ("car", "rear_bumper", "quarter_panel"): "rear_bumper",

    # Laptop: VLM says body for trackpad
    ("laptop", "trackpad", "body"): "trackpad",
    ("laptop", "trackpad", "base"): "trackpad",
}

# Close-part groups — parts that are considered "close enough" for status purposes
# When VLM says one of these and user claims another in the same group, treat as evidence
CLOSE_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "hood", "door",
        "headlight", "taillight", "fender", "quarter_panel",
    },
    "laptop": {
        "lid", "screen", "hinge", "corner", "body", "base",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
    },
}


def calibrate_part(
    object_type: str,
    claimed_part: str,
    vlm_part: str,
) -> str:
    """Apply calibration overrides for known VLM part confusions."""
    if vlm_part == claimed_part or claimed_part == "unknown":
        return claimed_part

    key = (object_type, claimed_part, vlm_part)
    if key in PART_OVERRIDES:
        return PART_OVERRIDES[key]

    return vlm_part


def is_close_part(
    object_type: str,
    part_a: str,
    part_b: str,
) -> bool:
    """Check if two parts are 'close' (confusable by VLM)."""
    if part_a == part_b:
        return True
    group = CLOSE_PARTS.get(object_type, set())
    return part_a in group and part_b in group
