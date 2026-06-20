"""
Engine 5: Fraud Detection Engine.

Cross-references claim extraction vs. vision findings to detect:
  - wrong_object: image shows car but claim is laptop
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
from typing import Optional

from models import ClaimExtraction, ClaimInput, FraudSignals, ImageAnalysis

logger = logging.getLogger(__name__)

# Weight map for risk scoring (higher = more severe signal)
RISK_WEIGHTS: dict[str, float] = {
    "wrong_object": 0.35,
    "wrong_object_part": 0.20,
    "claim_mismatch": 0.15,
    "possible_manipulation": 0.25,
    "non_original_image": 0.30,
    "text_instruction_present": 0.30,
    "damage_not_visible": 0.10,
    "blurry_image": 0.05,
    "cropped_or_obstructed": 0.05,
    "low_light_or_glare": 0.05,
    "wrong_angle": 0.05,
}


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
    _check_image_integrity(image_analyses, claim, flags, fraud)
    _check_sequential_images(image_analyses, flags, fraud)

    fraud.risk_flags = flags
    fraud.fraud_score = _compute_risk_score(flags)
    fraud.fraud_summary = "; ".join(flags) if flags else "No fraud signals detected"
    logger.info(
        f"Fraud check for {claim.user_id}: score={fraud.fraud_score:.2f}, "
        f"flags={fraud.fraud_summary}"
    )
    return fraud


def _majority_usable(analyses: list[ImageAnalysis]) -> int:
    """Return count of usable images; 0 if none."""
    return sum(1 for a in analyses if a.is_usable)


def _check_wrong_object(
    claim: ClaimInput,
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    usable_count = _majority_usable(analyses)
    if usable_count == 0:
        return

    wrong_count = sum(
        1 for a in analyses
        if a.is_usable
        and a.visible_object_type != claim.claim_object
        and a.visible_object_type not in ("unknown", "")
        and a.visible_object_type != "other"
    )

    if wrong_count / usable_count >= 0.51:
        fraud.has_wrong_object = True
        if "wrong_object" not in flags:
            flags.append("wrong_object")

    other_count = sum(1 for a in analyses if a.visible_object_type == "other" and a.is_usable)
    if usable_count > 0 and other_count / usable_count >= 0.51:
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

    usable = [a for a in analyses if a.is_usable]
    if not usable:
        return

    part_not_visible_count = sum(
        1 for a in usable
        if a.visible_object_part != extraction.claimed_object_part
        and extraction.claimed_object_part not in a.visible_parts_list
    )

    part_mismatch_ratio = part_not_visible_count / len(usable)
    if part_mismatch_ratio >= 0.51:
        right_object = any(
            a.visible_object_type not in ("unknown", "other", "")
            and a.is_usable
            for a in analyses
        )
        if right_object:
            fraud.has_wrong_object_part = True
            if "wrong_object_part" not in flags:
                flags.append("wrong_object_part")


def _same_evidence_family(issue_a: str, issue_b: str) -> bool:
    """Check if two issue types belong to the same evidence requirement family."""
    a_lower = issue_a.lower()
    b_lower = issue_b.lower()

    families = [
        {"dent", "scratch"},
        {"crack", "broken", "missing"},
        {"crushed", "torn", "seal"},
        {"water", "stain", "label"},
        {"screen", "keyboard", "trackpad"},
        {"hinge", "lid", "corner", "body", "port"},
    ]

    for family in families:
        if any(kw in a_lower for kw in family) and any(kw in b_lower for kw in family):
            return True
    return False


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
            and extraction.claimed_issue_type not in ("none", "unknown")
            and a.visible_issue_type != extraction.claimed_issue_type
            and a.visible_object_part != "unknown"
        ):
            if _same_evidence_family(a.visible_issue_type, extraction.claimed_issue_type):
                continue
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
    """Multi-factor vehicle identity cross-check."""
    if len(analyses) < 2:
        return

    usable = [a for a in analyses if a.is_usable]
    if len(usable) < 2:
        return

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

    identity_score = (color_score + type_score) / 2.0

    if identity_score < 0.3:
        fraud.has_vehicle_identity_issue = True
        logger.warning(f"Vehicle identity issue: colors={colors}, types={types}, score={identity_score:.2f}")
    elif not claimed_color_match and claimed_color:
        fraud.has_vehicle_identity_issue = True
        logger.warning(f"Vehicle identity issue: claimed color '{claimed_color}' not visible in images ({colors})")
    elif not claimed_color_match:
        logger.info(f"Claimed color '{claimed_color}' differs from visual colors: {colors}")

    if not fraud.has_vehicle_identity_issue and llm_client is not None:
        _check_vehicle_identity_vlm(claim, usable, flags, fraud, llm_client, identity_score)


def _check_vehicle_identity_vlm(
    claim: ClaimInput,
    usable_analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
    llm_client,
    identity_score: float = 0.0,
):
    from llm.prompts import build_vehicle_identity_prompt

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
        if identity_score >= 0.6:
            # VLM flagged identity but color/type analysis says same vehicle
            # Likely just different angles confusing the VLM
            logger.info(f"VLM flagged identity but color analysis (score={identity_score:.2f}) says same vehicle — trusting deterministic check")
            return

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

    clear_damage_visible = any(
        a.is_usable
        and a.visible_issue_type not in ("none", "unknown", "")
        and (
            a.visible_object_part == extraction.claimed_object_part
            or extraction.claimed_object_part in a.visible_parts_list
            or extraction.claimed_object_part in getattr(a, "damaged_parts", [])
        )
        for a in analyses
    )
    if clear_damage_visible:
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


def _check_image_integrity(
    analyses: list[ImageAnalysis],
    claim: ClaimInput,
    flags: list[str],
    fraud: FraudSignals,
):
    """Cross-image integrity checks: EXIF manipulation, near-duplicate detection, ELA."""
    usable = [a for a in analyses if a.is_usable]
    if not usable:
        return

    # EXIF manipulation flag
    any_edited = any(a.is_edited for a in usable)
    if any_edited and "possible_manipulation" not in flags:
        flags.append("possible_manipulation")

    # ELA anomaly detection (splicing/tampering)
    any_ela_anomaly = any(getattr(a, 'ela_anomaly', False) for a in usable)
    if any_ela_anomaly and "possible_manipulation" not in flags:
        flags.append("possible_manipulation")
        ela_diffs = [getattr(a, 'ela_mean_diff', 0) for a in usable if getattr(a, 'ela_anomaly', False)]
        logger.warning(f"ELA anomalies detected: mean_diffs={ela_diffs}")

    # Camera model consistency check
    camera_models = [getattr(a, 'exif_camera_model', '') for a in usable if getattr(a, 'exif_camera_model', '')]
    if len(set(camera_models)) > 1:
        logger.warning(
            f"Multiple camera models across images: {set(camera_models)} "
            f"— possible composite evidence"
        )
        if "possible_manipulation" not in flags:
            flags.append("possible_manipulation")

    image_paths = [a.image_path for a in usable if a.image_path]
    if len(image_paths) >= 2:
        from detectors.perceptual_hash import find_duplicates, max_phash_distance

        duplicates = find_duplicates(image_paths)
        if duplicates:
            logger.warning(
                f"Near-duplicate images detected: {len(duplicates)} pair(s)"
            )
            # Duplicate uploads within a claim are review context, not proof
            # of fraud and not a valid output-vocabulary flag on their own.
            if "manual_review_required" not in flags:
                flags.append("manual_review_required")

        if claim.claim_object == "car":
            max_dist = max_phash_distance(image_paths)
            if max_dist is not None and max_dist > 20:
                logger.warning(
                    f"Large perceptual distance ({max_dist}) between images "
                    f"— possible different vehicles"
                )


def _check_sequential_images(
    analyses: list[ImageAnalysis],
    flags: list[str],
    fraud: FraudSignals,
):
    """Check EXIF timestamps for suspicious patterns.
    
    Flags:
    - Sequential timestamps too close (< 1 second apart): possible repackaging
    - Timestamps too far apart (> 7 days): possible composite evidence
    - Missing EXIF across all images: possible sanitization
    """
    usable = [a for a in analyses if a.is_usable]
    if len(usable) < 2:
        return

    datetimes = []
    for a in usable:
        dt_str = getattr(a, 'exif_datetime', '') or ''
        if dt_str:
            datetimes.append(dt_str)

    if not datetimes:
        return

    all_no_exif = all(
        not getattr(a, 'has_exif', False) for a in usable
    )
    if all_no_exif and "possible_manipulation" not in flags:
        logger.warning("No EXIF data across all images — possible sanitization")
        return

    parsed = []
    import re
    for dt in datetimes:
        try:
            from datetime import datetime as dt_parse
            parsed.append(dt_parse.strptime(dt.strip(), "%Y:%m:%d %H:%M:%S"))
        except (ValueError, TypeError):
            pass

    if len(parsed) < 2:
        return

    from datetime import timedelta
    diffs = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            diffs.append(abs((parsed[i] - parsed[j]).total_seconds()))

    if not diffs:
        return

    min_diff = min(diffs)
    max_diff = max(diffs)

    if min_diff < 1.0:
        logger.warning(
            f"Sequential timestamps suspicious: {min_diff:.1f}s apart "
            f"({datetimes}) — possible repackaging"
        )
        if "possible_manipulation" not in flags:
            flags.append("possible_manipulation")

    if max_diff > 7 * 86400:
        logger.warning(
            f"Large timestamp gap: {max_diff / 86400:.1f} days "
            f"({datetimes}) — possible composite evidence"
        )
        if "possible_manipulation" not in flags:
            flags.append("possible_manipulation")


def _compute_risk_score(flags: list[str]) -> float:
    """Compute weighted fraud risk score from active flags.
    
    Uses predefined weights per flag. Score ranges 0.0 (clean) to 1.0 (fraud).
    Multiple flags combine additively, capped at 1.0.
    """
    score = 0.0
    for flag in flags:
        score += RISK_WEIGHTS.get(flag, 0.1)
    return min(1.0, score)
