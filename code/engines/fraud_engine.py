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
"""

from __future__ import annotations

import logging

from models import ClaimExtraction, ClaimInput, FraudSignals, ImageAnalysis

logger = logging.getLogger(__name__)


def detect_fraud(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analyses: list[ImageAnalysis],
    llm_client=None,
) -> FraudSignals:
    flags: list[str] = []
    fraud = FraudSignals()

    _check_wrong_object(claim, image_analyses, flags, fraud)
    _check_wrong_part(extraction, image_analyses, flags, fraud)
    _check_claim_mismatch(extraction, image_analyses, flags, fraud)

    if extraction.has_prompt_injection:
        fraud.has_prompt_injection_in_text = True
        logger.warning(f"Prompt injection in text: {extraction.prompt_injection_detail}")

    _check_image_text_instructions(image_analyses, flags, fraud)
    _check_non_original(image_analyses, flags, fraud)

    if claim.claim_object == "car":
        _check_vehicle_identity(claim, extraction, image_analyses, flags, fraud, llm_client)

    _check_unknown_object(claim, extraction, image_analyses, flags, fraud)
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

    other_count = sum(1 for a in analyses if a.visible_object_type == "other" and a.is_usable)
    usable_count = sum(1 for a in analyses if a.is_usable)
    if usable_count > 0 and other_count >= usable_count:
        fraud.has_wrong_object = True
        if "wrong_object" not in flags:
            flags.append("wrong_object")


def _check_wrong_part(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    if extraction.claimed_object_part == "unknown":
        return

    claimed_part_visible = any(
        a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )

    if not claimed_part_visible:
        right_object = any(
            a.visible_object_type not in ("unknown", "other", "")
            and a.is_usable
            for a in analyses
        )
        if right_object:
            fraud.has_wrong_object_part = True
            if "wrong_object_part" not in flags:
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
    if extraction.claimed_issue_type in ("unknown", "none"):
        return

    for a in analyses:
        if not a.is_usable:
            continue

        if (
            a.visible_issue_type not in ("none", "unknown")
            and a.visible_issue_type != extraction.claimed_issue_type
            and a.visible_object_part != "unknown"
        ):
            fraud.has_claim_mismatch = True
            if "claim_mismatch" not in flags:
                flags.append("claim_mismatch")

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
    all_watermarked = all(a.has_watermark for a in analyses) if analyses else False
    any_watermarked = any(a.has_watermark for a in analyses)

    if any_watermarked:
        fraud.has_non_original_image = True
        if "non_original_image" not in flags:
            flags.append("non_original_image")


def _check_unknown_object(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """When VLM says object_type=unknown but identifies a specific part,
    the visible object likely isn't the claimed type → wrong_object."""
    for a in analyses:
        if (
            a.visible_object_type in ("unknown", "other")
            and a.is_usable
            and a.visible_object_part not in ("unknown", "")
            and a.visible_object_part != extraction.claimed_object_part
        ):
            fraud.has_wrong_object = True
            if "wrong_object" not in flags:
                flags.append("wrong_object")
            logger.info(
                f"Unknown object detected: VLM sees part={a.visible_object_part} "
                f"but claim object={claim.claim_object}, claimed part={extraction.claimed_object_part}"
            )
            break


def _check_vehicle_identity(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
    llm_client=None,
):
    """Multi-factor vehicle identity cross-check.

    Instead of flagging on ANY color difference, compute an identity score
    from multiple signals and only flag when confidence is high that images
    show different vehicles.

    Factors:
      1. Color consistency (primary + secondary descriptions)
      2. Vehicle type consistency (sedan vs SUV vs truck)
      3. Claimed color vs visible color match
      4. VLM cross-image identity check (when available)
    """
    if len(analyses) < 2:
        return

    usable = [a for a in analyses if a.is_usable]
    if len(usable) < 2:
        return

    # ── Factor 1: Color consistency ──────────────────────────────────────
    colors = []
    for a in usable:
        if a.vehicle_color:
            c = a.vehicle_color.lower().strip()
            if c:
                colors.append(c)
                fraud.vehicle_colors_found.append(c)

    known_colors = {"blue", "black", "white", "red", "silver", "grey", "gray", "green", "yellow", "orange", "brown", "gold", "navy", "charcoal", "beige", "cream", "maroon", "purple"}

    def normalize_color(c):
        c = c.replace("grey", "gray").replace("colored", "").strip()
        for known in known_colors:
            if known in c:
                return known
        return c

    normalized_colors = [normalize_color(c) for c in colors if c]
    distinct_colors = set(normalized_colors)

    color_score = 1.0
    if len(distinct_colors) > 1:
        valid_distinct = distinct_colors - {"unknown", ""}
        if len(valid_distinct) > 1:
            color_score = 0.15
        else:
            color_score = 0.8

    # ── Factor 2: Vehicle type consistency ───────────────────────────────
    types = []
    for a in usable:
        vt = getattr(a, 'vehicle_type', '') or ''
        if vt:
            types.append(vt.lower().strip())

    type_score = 1.0
    if len(set(types)) > 1:
        known_types = {"sedan", "suv", "truck", "hatchback", "coupe", "convertible", "van", "wagon"}
        valid_types = [t for t in types if any(k in t for k in known_types)]
        if len(set(valid_types)) > 1:
            type_score = 0.2

    # ── Factor 3: Claimed color match ────────────────────────────────────
    conv_lower = claim.user_claim.lower()
    claimed_color = ""
    for color in ["blue", "black", "white", "red", "silver", "grey", "gray", "green"]:
        if f"my {color} car" in conv_lower or f"{color} car" in conv_lower:
            claimed_color = color
            break

    claimed_color_match = True
    if claimed_color and colors:
        nc = normalize_color(claimed_color)
        if not any(nc == normalize_color(c) for c in colors if c):
            claimed_color_match = False

    # ── Identity score ───────────────────────────────────────────────────
    identity_score = (color_score + type_score) / 2.0

    if identity_score < 0.3:
        fraud.has_vehicle_identity_issue = True
        logger.warning(f"Vehicle identity issue: colors={colors}, types={types}, score={identity_score:.2f}")
    elif not claimed_color_match and claimed_color:
        fraud.has_vehicle_identity_issue = True
        logger.warning(f"Vehicle identity issue: claimed color '{claimed_color}' not visible in images ({colors})")
    elif not claimed_color_match:
        logger.info(f"Claimed color '{claimed_color}' differs from visual colors: {colors}")

    # ── Factor 4: VLM Cross-Image Identity Check ─────────────────────────
    # When color/type matching is inconclusive, ask the VLM directly
    if not fraud.has_vehicle_identity_issue and llm_client is not None:
        _check_vehicle_identity_vlm(claim, usable, flags, fraud, llm_client)


def _check_vehicle_identity_vlm(
    claim: ClaimInput,
    usable_analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
    llm_client,
):
    """Use VLM to cross-check vehicle identity across images.
    
    Builds summaries from per-image analyses and asks the VLM to determine
    if images show the same vehicle.
    """
    from llm.prompts import build_vehicle_identity_prompt

    # Convert ImageAnalysis to dicts for prompt
    image_summaries = []
    for a in usable_analyses:
        image_summaries.append({
            "image_id": a.image_id,
            "visible_object_type": a.visible_object_type,
            "vehicle_color": a.vehicle_color,
            "visible_object_part": a.visible_object_part,
            "visible_issue_type": a.visible_issue_type,
        })

    conv_lower = claim.user_claim.lower()
    claimed_description = ""
    for color in ["blue", "black", "white", "red", "silver", "grey", "gray", "green"]:
        if f"my {color} car" in conv_lower:
            claimed_description = f"{color} car"
            break

    prompt = build_vehicle_identity_prompt(image_summaries, claimed_description)
    result = llm_client.call_text(prompt)

    if result is None:
        logger.warning("VLM vehicle identity check failed, using color-only result")
        return

    if not result.get("same_vehicle", True):
        fraud.has_vehicle_identity_issue = True
        fraud.has_wrong_object = True
        reason = result.get("consistency_reason", "Images appear to show different vehicles")
        logger.warning(f"Vehicle identity issue (VLM): {reason}")
        if "wrong_object" not in flags:
            flags.append("wrong_object")


def _check_damage_visibility(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    if extraction.claimed_issue_type in ("none", "unknown"):
        return

    for a in analyses:
        if not a.is_usable:
            continue
        if (
            a.visible_object_part == extraction.claimed_object_part
            and a.visible_issue_type == "none"
        ):
            fraud.damage_not_visible = True
            if "damage_not_visible" not in flags:
                flags.append("damage_not_visible")
            break
