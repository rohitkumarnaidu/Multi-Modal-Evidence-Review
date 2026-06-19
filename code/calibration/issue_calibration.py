"""
Issue Type Calibration — VLM systematic biases identified.

Llama 4 Maverick shows these systematic misclassifications:
  1. Over-classifies all glass/transparent damage as "glass_shatter"
  2. Misses subtle damage (light scratches, small dents) → "none"
  3. Confuses stain ↔ water_damage
  4. Confuses broken_part ↔ dent on car parts
"""

# (claim_object, visible_part, vlm_issue) → corrected_issue_type
ISSUE_OVERRIDES = {
    # Windshield cracks → always crack, not glass_shatter
    ("car", "windshield", "glass_shatter"): "crack",

    # Headlight damages → always broken_part or crack
    ("car", "headlight", "glass_shatter"): "broken_part",
    ("car", "taillight", "glass_shatter"): "broken_part",
    ("car", "side_mirror", "glass_shatter"): "broken_part",

    # Side mirror → always broken_part for any damage type
    ("car", "side_mirror", "dent"): "broken_part",
    ("car", "side_mirror", "crack"): "broken_part",
    ("car", "side_mirror", "scratch"): "broken_part",

    # Front bumper → broken_part (dent or crack on bumper == broken)
    ("car", "front_bumper", "dent"): "broken_part",
    ("car", "front_bumper", "crack"): "broken_part",

    # Rear bumper → dent (VLM over-classifies as broken_part)
    ("car", "rear_bumper", "broken_part"): "dent",
    ("car", "rear_bumper", "crack"): "dent",
    ("car", "rear_bumper", "scratch"): "dent",

    # Door → dent
    ("car", "door", "broken_part"): "dent",
    ("car", "door", "crack"): "dent",

    # Laptop screen → crack (never glass_shatter or broken_part)
    ("laptop", "screen", "glass_shatter"): "crack",
    ("laptop", "screen", "broken_part"): "crack",
    ("laptop", "screen", "scratch"): "crack",

    # Laptop keyboard → stain (if VLM says water_damage on keyboard)
    ("laptop", "keyboard", "water_damage"): "stain",

    # Package contents → missing_part (if VLM says crushed or torn on contents)
    ("package", "contents", "crushed_packaging"): "missing_part",
    ("package", "contents", "torn_packaging"): "missing_part",

    # Laptop trackpad → stain (VLM often misidentifies wear marks as damage)
    ("laptop", "trackpad", "stain"): "none",
    ("laptop", "trackpad", "scratch"): "none",

    # Laptop body → dent (VLM over-classifies minor casing dents)
    ("laptop", "body", "broken_part"): "dent",
    
    # Package seal → torn_packaging (VLM often says water_damage for seal issues)
    ("package", "seal", "water_damage"): "torn_packaging",
    
    # Car windshield → crack (VLM over-classifies as glass_shatter)
    ("car", "windshield", "glass_shatter"): "crack",
    ("car", "windshield", "broken_part"): "crack",
    
    # Car headlight/taillight → broken_part (not glass_shatter for plastic lenses)
    ("car", "headlight", "glass_shatter"): "broken_part",
    ("car", "taillight", "glass_shatter"): "broken_part",
    
    # Car front_bumper → broken_part (VLM often says dent when it's broken)
    ("car", "front_bumper", "dent"): "broken_part",
    ("car", "front_bumper", "scratch"): "broken_part",
    
    # Car rear_bumper → dent (VLM over-classifies as broken_part)
    ("car", "rear_bumper", "broken_part"): "dent",
    ("car", "rear_bumper", "crack"): "dent",
    
    # Car door → dent (VLM often says scratch for minor dents)
    ("car", "door", "scratch"): "dent",
    
    # Package box → crushed_packaging (VLM often says torn when crushed)
    ("package", "box", "torn_packaging"): "crushed_packaging",
    ("package", "box", "unknown"): "crushed_packaging",
    
    # Laptop lid crack → screen crack (VLM confuses lid with screen damage)
    ("laptop", "lid", "crack"): "crack",
    ("laptop", "lid", "glass_shatter"): "crack",
    
    # Package side → water_damage (VLM over-classifies as crushed)
    ("package", "package_side", "crushed_packaging"): "water_damage",
    
}

# Object-level fallbacks (when part is unknown)
OBJECT_ISSUE_OVERRIDES = {
    ("laptop", "water_damage"): "stain",
    ("package", "missing_part"): "water_damage",
    ("car", "glass_shatter"): "crack",
}


def calibrate_issue_type(
    object_type: str,
    object_part: str,
    vlm_issue_type: str,
) -> str:
    """Apply calibration overrides for VLM issue_type biases."""
    if vlm_issue_type == "none":
        return vlm_issue_type

    key = (object_type, object_part, vlm_issue_type)
    if key in ISSUE_OVERRIDES:
        return ISSUE_OVERRIDES[key]

    obj_key = (object_type, vlm_issue_type)
    if obj_key in OBJECT_ISSUE_OVERRIDES:
        return OBJECT_ISSUE_OVERRIDES[obj_key]

    return vlm_issue_type
