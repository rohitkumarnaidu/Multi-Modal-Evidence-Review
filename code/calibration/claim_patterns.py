"""
Claim Pattern Recognition — multi-part claims, evidence expectations.

Some claims involve multiple parts (e.g., "front bumper and left headlight").
This module detects such patterns and provides evidence expectations for each.
"""

MULTI_PART_PATTERNS = {
    "and": "conjunction",
    "&": "conjunction",
    "both": "conjunction",
    "as well as": "conjunction",
    "as wel as": "conjunction",
    "along with": "conjunction",
    "plus": "conjunction",
    "also": "also_suffix",
}

CAR_PARTS = {
    "front_bumper": ["front bumper", "front bumper"],
    "rear_bumper": ["rear bumper", "back bumper", "rear bumper"],
    "door": ["door"],
    "hood": ["hood", "bonnet"],
    "windshield": ["windshield", "windscreen", "wind screen", "wind shield"],
    "side_mirror": ["mirror", "side mirror", "side mirror"],
    "headlight": ["headlight", "head light", "headlight"],
    "taillight": ["taillight", "tail light", "tailight", "taillight"],
    "fender": ["fender"],
    "quarter_panel": ["quarter panel", "quarter panel"],
    "body": ["body"],
}

LAPTOP_PARTS = {
    "screen": ["screen", "display"],
    "keyboard": ["keyboard"],
    "trackpad": ["trackpad", "touchpad", "touch pad", "track pad"],
    "hinge": ["hinge"],
    "lid": ["lid", "cover"],
    "corner": ["corner"],
    "port": ["port", "usb", "charging port"],
    "base": ["base", "bottom"],
    "body": ["body", "casing", "chassis"],
}

PACKAGE_PARTS = {
    "box": ["box"],
    "package_corner": ["corner"],
    "package_side": ["side"],
    "seal": ["seal", "tape", "seal"],
    "label": ["label", "sticker"],
    "contents": ["contents", "content", "item", "items", "inside"],
    "item": ["item"],
}


def detect_multi_part(claim_text: str) -> tuple[bool, list[str]]:
    """Check if a claim mentions multiple parts.

    Returns (is_multi_part, secondary_parts_found).
    """
    text = claim_text.lower()

    if " only " in text or text.strip().endswith(" only"):
        return False, []

    for pattern in MULTI_PART_PATTERNS:
        if pattern in text:
            return True, []

    return False, []
