"""
Engine 2: Vision Evidence Engine.

Per-image VLM analysis — each image analyzed INDEPENDENTLY.
This is Call 2 of the 2-call design.

Key design decision: Images are analyzed independently so that:
  - Each image gets its own quality assessment
  - Vehicle identity can be cross-checked across images
  - Blurry img_1 doesn't pollute analysis of clear img_2
  - Supporting image selection is per-image
  - Text instructions in one image don't influence others
"""

from __future__ import annotations

import logging
from typing import Optional

from config import ISSUE_TYPES, OBJECT_PARTS_BY_TYPE, STOCK_IMAGE_MARKERS
from data_loader import get_image_mime_type, load_image_as_base64
from detectors.cv_quality import analyze_image_quality
from detectors.ela import analyze_ela
from detectors.exif_analyzer import analyze_exif
from detectors.yolo_detector import detect_objects
from engines.claim_engine import _fuzzy_match_part, _fuzzy_match_issue
from models import ClaimInput, ImageAnalysis

logger = logging.getLogger(__name__)


def analyze_single_image(
    claim: ClaimInput,
    image_path: str,
    image_id: str,
    llm_client,
) -> ImageAnalysis:
    """Analyze a single image with the VLM.
    
    Returns ImageAnalysis with all visual findings independent of claim.
    """
    from llm.prompts import build_image_analysis_prompt

    # Load image
    base64_data = load_image_as_base64(image_path)
    if base64_data is None:
        logger.error(f"Cannot load image: {image_path}")
        return ImageAnalysis(
            image_id=image_id,
            image_path=image_path,
            is_usable=False,
            damage_description="Image could not be loaded.",
            confidence=0.0,
        )

    # Run classical CV pre-check (fast, no API call)
    cv_quality = analyze_image_quality(image_path)
    exif_data = analyze_exif(image_path)
    ela_result = analyze_ela(image_path)

    # Run YOLO for deterministic object detection prior
    yolo_result = detect_objects(image_path)
    yolo_prior = ""
    if yolo_result and yolo_result["object_type"] != "unknown":
        obj = yolo_result["object_type"]
        conf = yolo_result["confidence"]
        if yolo_result["all_objects"]:
            labels = [d["label"] for d in yolo_result["all_objects"][:3]]
            yolo_prior = (
                f"YOLO object detection prior: detected a {obj} "
                f"(confidence={conf:.2f}, labels: {', '.join(labels)}). "
                f"Use this as a hint for visible_object_type."
            )
        else:
            yolo_prior = (
                f"YOLO object detection prior: detected a {obj} "
                f"(confidence={conf:.2f}). "
                f"Use this as a hint for visible_object_type."
            )

    # Build prompt and call VLM (ensemble mode when enabled)
    prompt = build_image_analysis_prompt(image_id, claim.claim_object, yolo_prior)
    mime_type = get_image_mime_type(image_path)

    if hasattr(llm_client, "call_vision_ensemble"):
        result = llm_client.call_vision_ensemble(
            prompt=prompt,
            image_data=[{"mime_type": mime_type, "data": base64_data}],
            image_paths=[image_path],
        )
    else:
        result = llm_client.call_vision(
            prompt=prompt,
            image_data=[{"mime_type": mime_type, "data": base64_data}],
            image_paths=[image_path],
        )

    if result is None:
        logger.error(f"VLM analysis failed for {image_id}")
        return ImageAnalysis(
            image_id=image_id,
            image_path=image_path,
            is_usable=False,
            damage_description="VLM analysis failed.",
            confidence=0.0,
        )

    # Parse VLM response into ImageAnalysis (with fuzzy matching for robustness)
    allowed_parts = OBJECT_PARTS_BY_TYPE.get(claim.claim_object, set())
    visible_part = result.get("visible_object_part", "unknown")
    if visible_part not in allowed_parts:
        visible_part = _fuzzy_match_part(visible_part, allowed_parts)

    visible_issue = result.get("visible_issue_type", "unknown")
    if visible_issue not in ISSUE_TYPES:
        visible_issue = _fuzzy_match_issue(visible_issue)

    # Parse CoT reasoning from VLM result
    vlm_reasoning = result.get("reasoning", "")

    # Parse visible_parts_list from VLM result
    raw_parts = result.get("visible_parts_list", [])
    if isinstance(raw_parts, list):
        visible_parts = [p for p in raw_parts if isinstance(p, str) and p in allowed_parts]
    else:
        visible_parts = [visible_part] if visible_part in allowed_parts else []

    # Merge YOLO with VLM object type (YOLO overrides when confident)
    yolo_obj = ""
    yolo_conf = 0.0
    if yolo_result:
        yolo_obj = yolo_result.get("object_type", "")
        yolo_conf = yolo_result.get("confidence", 0.0)
    if yolo_obj and yolo_conf >= 0.5 and yolo_obj in ("car", "laptop", "package"):
        obj_type = yolo_obj
    else:
        obj_type = result.get("visible_object_type", "unknown")

    # Merge CV quality with VLM results (CV overrides for objective metrics)
    analysis = ImageAnalysis(
        image_id=image_id,
        image_path=image_path,
        reasoning=vlm_reasoning,
        visible_object_type=obj_type,
        visible_object_part=visible_part,
        visible_parts_list=visible_parts,
        visible_issue_type=visible_issue,
        visible_severity=_normalize_severity(result.get("visible_severity", "unknown")),
        vehicle_color=result.get("vehicle_color", ""),
        yolo_object_type=yolo_obj,
        yolo_confidence=yolo_conf,
        is_blurry=cv_quality.get("is_blurry", False) or result.get("is_blurry", False),
        is_low_light=cv_quality.get("is_low_light", False) or result.get("is_low_light", False),
        is_cropped=cv_quality.get("is_cropped", False) or result.get("is_cropped", False),
        has_wrong_angle=cv_quality.get("has_wrong_angle", False) or result.get("has_wrong_angle", False),
        has_watermark=result.get("has_watermark", False),
        watermark_text=result.get("watermark_text", ""),
        has_exif=exif_data.get("has_exif", False),
        is_edited=exif_data.get("is_edited", False),
        exif_datetime=exif_data.get("datetime_original", ""),
        exif_camera_model=exif_data.get("camera_model", ""),
        ela_anomaly=ela_result.get("has_anomaly", False),
        ela_mean_diff=ela_result.get("ela_mean_diff", 0.0),
        has_text_instruction=result.get("has_text_instruction", False),
        text_instruction_content=result.get("text_instruction_content", ""),
        is_usable=result.get("is_usable", True),
        damage_description=result.get("damage_description", ""),
        confidence=min(1.0, max(0.0, result.get("confidence", 0.5))),
    )

    # Post-process: check for stock image markers in watermark text
    if analysis.watermark_text:
        wt_lower = analysis.watermark_text.lower()
        for marker in STOCK_IMAGE_MARKERS:
            if marker in wt_lower:
                analysis.has_watermark = True
                break

    logger.info(
        f"Image {image_id}: obj={analysis.visible_object_type}, "
        f"part={analysis.visible_object_part}, "
        f"issue={analysis.visible_issue_type}, "
        f"severity={analysis.visible_severity}, "
        f"usable={analysis.is_usable}, "
        f"text_instruction={analysis.has_text_instruction}"
    )
    return analysis


def analyze_all_images(
    claim: ClaimInput,
    llm_client,
) -> list[ImageAnalysis]:
    """Analyze all images for a claim, each independently.
    
    Returns list of ImageAnalysis, one per image.
    """
    analyses = []
    for path, img_id in zip(claim.image_path_list, claim.image_ids):
        analysis = analyze_single_image(claim, path, img_id, llm_client)
        analyses.append(analysis)
    return analyses


def select_supporting_images(
    analyses: list[ImageAnalysis],
    claimed_part: str,
    claimed_issue: str,
) -> list[str]:
    """Select image IDs that best support the claim decision.
    
    Rules (from sample labels):
      - Pick the image(s) that show the claimed part AND damage
      - If one is blurry and another is clear, prefer the clear one
      - If no image shows the claimed part, return "none"
      - For contradicted claims, still return images that show evidence 
        (since they support the CONTRADICTION finding)
    """
    # Tier 1: Images showing claimed part WITH damage (clear)
    tier1 = [
        a for a in analyses
        if a.shows_claimed_part and a.shows_claimed_damage
        and not a.is_blurry and a.is_usable
    ]

    # Tier 2: Images showing claimed part (may have damage, may be clear)
    tier2 = [
        a for a in analyses
        if a.shows_claimed_part and a.is_usable
    ]

    # Tier 3: Images showing the relevant damage type, even if not the exact part claimed
    tier3 = [
        a for a in analyses
        if a.visible_issue_type == claimed_issue
        and a.visible_issue_type != "none"
        and a.is_usable
    ]

    # Tier 4: Any usable image showing relevant object
    tier4 = [
        a for a in analyses
        if a.is_usable
        and a.visible_object_type != "other"
        and a.visible_object_type != "unknown"
    ]

    for tier in [tier1, tier2, tier3, tier4]:
        if tier:
            return [a.image_id for a in tier]

    return []


def _normalize_severity(raw: str) -> str:
    """Normalize severity value."""
    raw = raw.strip().lower()
    valid = {"none", "low", "medium", "high", "unknown"}
    if raw in valid:
        return raw
    # Map common variants
    mapping = {
        "minor": "low",
        "moderate": "medium",
        "severe": "high",
        "critical": "high",
        "minimal": "low",
        "significant": "medium",
    }
    return mapping.get(raw, "unknown")
