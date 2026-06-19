"""
Engine 5: Fraud Detection Engine.

Cross-references claim extraction vs. vision findings to detect:
  - wrong_object: image shows car but claim is laptop, or canned food vs box
  - wrong_object_part: right object, wrong part visible
  - claim_mismatch: damage type or severity doesn't match claim
  - possible_manipulation: suspicious patterns
  - non_original_image: stock watermarks
  - text_instruction_present: text in image telling system to approve
  - damage_not_visible: part is visible but no damage seen
  - vehicle identity issues: different cars in multi-image set

Key design rules (from user corrections):
  - wrong_object vs wrong_object_part are DISTINCT signals
  - damage_not_visible is different from issue_type=none
  - Vehicle COLOR matching is required for car claims
  - Text instruction detection must come from VLM (image analysis), not just regex
"""

from __future__ import annotations

import logging

from models import ClaimExtraction, ClaimInput, FraudSignals, ImageAnalysis

logger = logging.getLogger(__name__)


def detect_fraud(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analyses: list[ImageAnalysis],
) -> FraudSignals:
    """Run all fraud detection checks.
    
    Returns FraudSignals with all detected risk flags.
    """
    flags: list[str] = []
    fraud = FraudSignals()

    # ── 1. Wrong Object Detection ────────────────────────────────────────
    _check_wrong_object(claim, image_analyses, flags, fraud)

    # ── 2. Wrong Object Part Detection ───────────────────────────────────
    _check_wrong_part(extraction, image_analyses, flags, fraud)

    # ── 3. Claim Mismatch Detection ──────────────────────────────────────
    _check_claim_mismatch(extraction, image_analyses, flags, fraud)

    # ── 4. Prompt Injection in Conversation ──────────────────────────────
    if extraction.has_prompt_injection:
        fraud.has_prompt_injection_in_text = True
        # Don't add a flag here — it's captured in other flags
        logger.warning(f"Prompt injection in text: {extraction.prompt_injection_detail}")

    # ── 5. Text Instructions in Images ───────────────────────────────────
    _check_image_text_instructions(image_analyses, flags, fraud)

    # ── 6. Non-Original Images (Watermarks) ──────────────────────────────
    _check_non_original(image_analyses, flags, fraud)

    # ── 7. Vehicle Identity Cross-Check ──────────────────────────────────
    if claim.claim_object == "car":
        _check_vehicle_identity(claim, extraction, image_analyses, flags, fraud)

    # ── 8. Damage Not Visible ────────────────────────────────────────────
    _check_damage_visibility(extraction, image_analyses, flags, fraud)

    fraud.risk_flags = flags
    fraud.fraud_summary = "; ".join(flags) if flags else "No fraud signals detected"

    logger.info(f"Fraud check for {claim.user_id}: {fraud.fraud_summary}")
    return fraud


def _check_wrong_object(
    claim: ClaimInput,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check if images show wrong object type entirely."""
    for a in analyses:
        if (
            a.visible_object_type != claim.claim_object
            and a.visible_object_type not in ("unknown", "")
            and a.visible_object_type != "other"
            and a.is_usable
        ):
            fraud.has_wrong_object = True
            if "wrong_object" not in flags:
                flags.append("wrong_object")
            break

    # Also check "other" — e.g., canned food when claiming package
    for a in analyses:
        if a.visible_object_type == "other" and a.is_usable:
            fraud.has_wrong_object = True
            if "wrong_object" not in flags:
                flags.append("wrong_object")
            break


def _check_wrong_part(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check if images show the right object but wrong part."""
    if extraction.claimed_object_part == "unknown":
        return

    # Check if ANY usable image shows the claimed part
    claimed_part_visible = any(
        a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )

    if not claimed_part_visible:
        # Right object visible but wrong part
        right_object = any(
            a.visible_object_type not in ("unknown", "other", "")
            and a.is_usable
            for a in analyses
        )
        if right_object:
            fraud.has_wrong_object_part = True
            if "wrong_object_part" not in flags:
                # Only flag this if images show the object but not the claimed part
                # AND there's a clearly different part visible
                visible_parts = [
                    a.visible_object_part for a in analyses
                    if a.is_usable and a.visible_object_part != "unknown"
                ]
                if visible_parts:
                    flags.append("wrong_object_part")


def _check_claim_mismatch(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check if visible damage contradicts claimed damage type/severity."""
    if extraction.claimed_issue_type in ("unknown", "none"):
        return

    for a in analyses:
        if not a.is_usable:
            continue

        # Check damage type mismatch
        if (
            a.visible_issue_type not in ("none", "unknown")
            and a.visible_issue_type != extraction.claimed_issue_type
            and a.visible_object_part != "unknown"
        ):
            fraud.has_claim_mismatch = True
            if "claim_mismatch" not in flags:
                flags.append("claim_mismatch")

        # Check severity exaggeration
        # User claims severe but visual shows minor
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}
        hint = extraction.claimed_severity_hint.lower()
        if hint in ("severe", "bad", "pretty bad", "heavily", "badly"):
            if a.visible_severity in ("low", "none"):
                fraud.has_claim_mismatch = True
                if "claim_mismatch" not in flags:
                    flags.append("claim_mismatch")


def _check_image_text_instructions(
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check for text instructions found IN images by VLM."""
    for a in analyses:
        if a.has_text_instruction:
            fraud.has_prompt_injection_in_image = True
            if "text_instruction_present" not in flags:
                flags.append("text_instruction_present")
            logger.warning(
                f"Text instruction in image {a.image_id}: "
                f"{a.text_instruction_content}"
            )


def _check_non_original(
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check for stock image watermarks."""
    all_watermarked = all(a.has_watermark for a in analyses) if analyses else False
    any_watermarked = any(a.has_watermark for a in analyses)

    if any_watermarked:
        fraud.has_non_original_image = True
        if "non_original_image" not in flags:
            flags.append("non_original_image")


def _check_vehicle_identity(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Cross-check vehicle identity across multiple images.
    
    Checks:
      - Vehicle color consistency
      - Vehicle type consistency
      - Claimed color vs visible color (e.g., "my blue car")
    """
    if len(analyses) < 2:
        return

    # Collect vehicle colors
    colors = []
    for a in analyses:
        if a.vehicle_color and a.is_usable:
            colors.append(a.vehicle_color.lower().strip())
            fraud.vehicle_colors_found.append(a.vehicle_color.lower().strip())

    # Check color consistency across images
    if len(set(colors)) > 1:
        fraud.has_vehicle_identity_issue = True
        logger.warning(f"Vehicle color mismatch: {colors}")

    # Check claimed color from conversation
    conv_lower = claim.user_claim.lower()
    claimed_color = ""
    for color in ["blue", "black", "white", "red", "silver", "grey", "gray", "green"]:
        if f"my {color} car" in conv_lower or f"{color} car" in conv_lower:
            claimed_color = color
            break

    if claimed_color and colors:
        # Normalize grey/gray
        normalized_colors = [c.replace("grey", "gray") for c in colors]
        claimed_color = claimed_color.replace("grey", "gray")
        if not any(claimed_color in c for c in normalized_colors):
            fraud.has_vehicle_identity_issue = True
            if "wrong_object" not in flags:
                flags.append("wrong_object")
            logger.warning(
                f"Claimed color '{claimed_color}' not found in images: {colors}"
            )


def _check_damage_visibility(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check if the claimed part is visible but damage is NOT visible."""
    if extraction.claimed_issue_type in ("none", "unknown"):
        return

    for a in analyses:
        if not a.is_usable:
            continue
        # Part is visible but no damage seen
        if (
            a.visible_object_part == extraction.claimed_object_part
            and a.visible_issue_type == "none"
        ):
            fraud.damage_not_visible = True
            if "damage_not_visible" not in flags:
                flags.append("damage_not_visible")
            break
