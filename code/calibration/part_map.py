"""Part calibration for common adjacent-part visual confusions."""

from __future__ import annotations


PART_OVERRIDES = {
    ("laptop", "screen", "lid"): "screen",
    ("laptop", "hinge", "lid"): "hinge",
    ("laptop", "hinge", "screen"): "hinge",
    ("laptop", "corner", "lid"): "corner",
    ("laptop", "trackpad", "body"): "trackpad",
    ("laptop", "trackpad", "base"): "trackpad",
    ("package", "seal", "package_side"): "seal",
    ("package", "seal", "box"): "seal",
    ("package", "package_side", "package_corner"): "package_side",
    ("car", "front_bumper", "hood"): "front_bumper",
    ("car", "front_bumper", "body"): "front_bumper",
    ("car", "headlight", "side_mirror"): "headlight",
    ("car", "side_mirror", "door"): "side_mirror",
    ("car", "side_mirror", "body"): "side_mirror",
    ("car", "taillight", "front_bumper"): "taillight",
    ("car", "rear_bumper", "quarter_panel"): "rear_bumper",
}


CLOSE_PARTS = {
    "car": [
        {"front_bumper", "hood", "headlight", "fender"},
        {"rear_bumper", "taillight", "quarter_panel"},
        {"door", "side_mirror", "fender", "quarter_panel"},
    ],
    "laptop": [
        {"screen", "lid"},
        {"hinge", "lid", "corner"},
        {"trackpad", "keyboard", "base", "body"},
        {"port", "base", "body"},
    ],
    "package": [
        {"box", "package_corner", "package_side"},
        {"seal", "label", "package_side"},
        {"contents", "item"},
    ],
}


def calibrate_part(
    object_type: str,
    claimed_part: str,
    vlm_part: str,
) -> str:
    """Normalize specific VLM part confusions without inventing evidence."""
    if vlm_part == claimed_part or claimed_part == "unknown":
        return claimed_part
    return PART_OVERRIDES.get((object_type, claimed_part, vlm_part), vlm_part)


def is_close_part(
    object_type: str,
    part_a: str,
    part_b: str,
) -> bool:
    """Return true only for adjacent parts that can plausibly share evidence."""
    if part_a == part_b:
        return True
    return any(
        part_a in group and part_b in group
        for group in CLOSE_PARTS.get(object_type, [])
    )
