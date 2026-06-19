"""
Engine 4: Image Quality Engine.

Aggregates per-image quality assessments into overall valid_image
and quality-related risk flags.

valid_image = false when:
  - All images have stock watermarks (non-original)
  - All images are completely unusable
  - Contents cannot be verified for missing item claims
"""

from __future__ import annotations

import logging

from models import ImageAnalysis

logger = logging.getLogger(__name__)


def assess_image_quality(
    image_analyses: list[ImageAnalysis],
) -> dict:
    """Assess overall image quality across all submitted images.
    
    Returns dict with:
      - valid_image: bool
      - quality_flags: list of risk flags from quality issues
    """
    if not image_analyses:
        return {
            "valid_image": False,
            "quality_flags": ["damage_not_visible"],
        }

    quality_flags = []

    # Check each image for quality issues
    any_usable = False
    any_has_watermark = False
    any_blurry = False
    any_cropped = False
    any_low_light = False
    any_wrong_angle = False
    any_text_instruction = False
    any_edited = False
    all_non_original = True

    for a in image_analyses:
        if a.is_usable:
            any_usable = True
        if a.is_blurry:
            any_blurry = True
        if a.is_cropped:
            any_cropped = True
        if a.is_low_light:
            any_low_light = True
        if a.has_wrong_angle:
            any_wrong_angle = True
        if a.has_watermark:
            any_has_watermark = True
        else:
            all_non_original = False
        if a.has_text_instruction:
            any_text_instruction = True
        if a.is_edited:
            any_edited = True

    # Build quality flags
    if any_blurry:
        quality_flags.append("blurry_image")
    if any_cropped:
        quality_flags.append("cropped_or_obstructed")
    if any_low_light:
        quality_flags.append("low_light_or_glare")
    if any_wrong_angle:
        quality_flags.append("wrong_angle")
    if any_text_instruction:
        quality_flags.append("text_instruction_present")
    if any_has_watermark and all_non_original:
        quality_flags.append("non_original_image")
    if any_edited and "possible_manipulation" not in quality_flags:
        quality_flags.append("possible_manipulation")

    # Determine valid_image
    valid_image = True
    if not any_usable:
        valid_image = False
    if all_non_original and any_has_watermark:
        valid_image = False

    return {
        "valid_image": valid_image,
        "quality_flags": quality_flags,
    }
