"""
Engine 7: Decision Engine.

Aggregates all engine outputs into final claim_status, issue_type,
object_part, severity, supporting_image_ids, and risk_flags.
"""

from __future__ import annotations

import logging

from calibration.issue_calibration import calibrate_issue_type
from calibration.part_map import calibrate_part, CLOSE_PARTS
from calibration.severity_map import calibrate_severity
from models import (
    ClaimExtraction,
    ClaimInput,
    ClaimOutput,
    EvidenceSufficiency,
    FraudSignals,
    ImageAnalysis,
)

logger = logging.getLogger(__name__)


def make_decision(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analyses: list[ImageAnalysis],
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    quality: dict,
    user_risk_flags: list[str],
    user_risk_summary: str,
) -> ClaimOutput:
    visible_part = _determine_visible_part(claim, extraction, image_analyses, fraud)

    # Apply part calibration to per-image analyses so downstream logic uses corrected data.
    # Only overwrite when there's an actual override (check: result != original)
    for a in image_analyses:
        original = a.visible_object_part
        calibrated = calibrate_part(
            claim.claim_object, extraction.claimed_object_part, original
        )
        if calibrated != original:
            a.visible_object_part = calibrated
            logger.debug(
                f"Part calibration: {original} -> {calibrated} for {a.image_id}"
            )

    raw_issue = _determine_visible_issue(image_analyses, visible_part, extraction)

    # Step 1: VLM sometimes misses subtle damage (small dents, light scratches).
    # If VLM says "none" but user claims a specific issue AND the VLM did NOT
    # explicitly flag damage_not_visible, trust the claim over VLM's "none".
    # Exception: if VLM confirms part is visible with no damage, trust VLM.
    if raw_issue != "none":
        visible_issue = calibrate_issue_type(
            claim.claim_object, visible_part, raw_issue
        )
    else:
        damage_not_visible_flagged = hasattr(fraud, 'damage_not_visible') and fraud.damage_not_visible
        if damage_not_visible_flagged:
            visible_issue = "none"
        else:
            calibrated = calibrate_issue_type(
                claim.claim_object, visible_part, extraction.claimed_issue_type
            )
            visible_issue = calibrated if calibrated not in ("none", "unknown") else extraction.claimed_issue_type

    # Apply issue calibration to per-image analyses so decision logic uses consistent data
    for a in image_analyses:
        if a.visible_issue_type not in ("none", "unknown"):
            calibrated_issue = calibrate_issue_type(
                claim.claim_object, a.visible_object_part or visible_part, a.visible_issue_type
            )
            if calibrated_issue != a.visible_issue_type:
                a.visible_issue_type = calibrated_issue

    for a in image_analyses:
        if visible_issue not in ("none", "unknown"):
            a.shows_claimed_damage = (
                a.visible_issue_type == visible_issue
                or (a.visible_issue_type == "none" and extraction.claimed_issue_type == visible_issue)
            )
            if a.visible_issue_type == "none" and extraction.claimed_issue_type == visible_issue:
                damage_not_visible_flagged = hasattr(fraud, 'damage_not_visible') and fraud.damage_not_visible
                a.shows_claimed_damage = not damage_not_visible_flagged

    # Step 3: Calibrate severity
    raw_severity = _determine_raw_severity(image_analyses, visible_part, visible_issue)
    severity = calibrate_severity(
        claim.claim_object, visible_part, visible_issue, raw_severity
    )

    # Fraud-to-severity calibration: bump severity when fraud signals present
    fraud_score = getattr(fraud, 'fraud_score', 0.0)
    if fraud_score >= 0.5 and severity in ("none", "low"):
        severity = "medium"
        logger.info(f"Severity bumped by fraud score {fraud_score:.2f}: {severity}")
    elif fraud_score >= 0.7 and severity == "medium":
        severity = "high"
        logger.info(f"Severity bumped by fraud score {fraud_score:.2f}: {severity}")

    damage_not_visible_flagged = hasattr(fraud, 'damage_not_visible') and fraud.damage_not_visible
    override_none_applied = (
        raw_issue == "none"
        and extraction.claimed_issue_type not in ("none", "unknown")
        and not damage_not_visible_flagged
    )

    claim_status = _determine_claim_status(
        claim, extraction, image_analyses, evidence, fraud, visible_part, visible_issue,
        override_none_applied=override_none_applied, quality=quality
    )

    # Override severity for NEI — can't determine from insufficient evidence
    if claim_status == "not_enough_information" and visible_issue in ("unknown",):
        severity = "unknown"

    supporting_ids = _select_supporting_images(
        image_analyses, extraction, visible_part, visible_issue, claim_status
    )

    all_flags = _merge_risk_flags(fraud.risk_flags, quality.get("quality_flags", []), user_risk_flags)

    if user_risk_flags and "manual_review_required" not in all_flags:
        for f in user_risk_flags:
            if f == "manual_review_required":
                all_flags.append("manual_review_required")
                break

    justification = _build_justification(
        claim, extraction, image_analyses, claim_status,
        visible_part, visible_issue, fraud, user_risk_summary, supporting_ids
    )

    evidence_reason = evidence.evidence_standard_met_reason
    risk_flags_str = ";".join(sorted(set(all_flags))) if all_flags else "none"
    supporting_str = ";".join(supporting_ids) if supporting_ids else "none"

    output = ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=str(evidence.evidence_standard_met).lower(),
        evidence_standard_met_reason=evidence_reason,
        risk_flags=risk_flags_str,
        issue_type=visible_issue,
        object_part=visible_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_str,
        valid_image=str(quality.get("valid_image", True)).lower(),
        severity=severity,
    )

    # Confidence scoring (default = 1.0 for single-provider, updated by ensemble)
    risk_deduction = 0.0
    for flag in all_flags:
        if flag in ("blurry_image", "cropped_or_obstructed", "low_light_or_glare"):
            risk_deduction = max(risk_deduction, 0.1)
        elif flag in ("wrong_angle", "damage_not_visible"):
            risk_deduction = max(risk_deduction, 0.2)
        elif flag in ("wrong_object", "wrong_object_part"):
            risk_deduction = max(risk_deduction, 0.3)
        elif flag in ("possible_manipulation", "non_original_image"):
            risk_deduction = max(risk_deduction, 0.4)

    base_confidence = 1.0 * (1 - risk_deduction)
    output.confidence_issue_type = base_confidence
    output.confidence_object_part = base_confidence
    output.confidence_claim_status = base_confidence
    output.confidence_severity = base_confidence
    output.confidence_avg = base_confidence

    logger.info(
        f"Decision for {claim.user_id}: "
        f"status={claim_status}, part={visible_part}, "
        f"issue={visible_issue}, severity={severity}"
    )
    return output


def _determine_visible_part(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    fraud: FraudSignals | None = None,
) -> str:
    if fraud and fraud.has_wrong_object:
        right_obj_parts = [
            a.visible_object_part for a in analyses
            if a.is_usable
            and a.visible_object_type not in ("other", "unknown")
            and a.visible_object_part != "unknown"
        ]
        if not right_obj_parts:
            return "unknown"

    best_parts = []
    for a in analyses:
        if a.is_usable and a.visible_object_part != "unknown":
            best_parts.append(a.visible_object_part)

    if not best_parts:
        for a in analyses:
            if a.visible_object_part != "unknown":
                best_parts.append(a.visible_object_part)

    # If still nothing, check images with quality flags but have VLM part data
    if not best_parts:
        for a in analyses:
            if a.visible_object_part not in ("unknown", ""):
                best_parts.append(a.visible_object_part)

    if not best_parts:
        return extraction.claimed_object_part

    if extraction.claimed_object_part in best_parts:
        return extraction.claimed_object_part

    # Check if any close part matches — calibrate VLM part to claimed part
    for part in best_parts:
        calibrated = calibrate_part(claim.claim_object, extraction.claimed_object_part, part)
        if calibrated == extraction.claimed_object_part:
            return extraction.claimed_object_part

    from collections import Counter
    counter = Counter(best_parts)
    return counter.most_common(1)[0][0]


def _determine_visible_issue(
    analyses: list[ImageAnalysis],
    visible_part: str,
    extraction: ClaimExtraction | None = None,
) -> str:
    for a in analyses:
        if (
            a.is_usable
            and a.visible_object_part == visible_part
            and a.visible_issue_type not in ("unknown",)
        ):
            return a.visible_issue_type

    for a in analyses:
        if a.is_usable and a.visible_issue_type not in ("none", "unknown"):
            return a.visible_issue_type

    for a in analyses:
        if a.is_usable and a.visible_object_part == visible_part:
            return "none"

    return "unknown"


def _determine_raw_severity(
    analyses: list[ImageAnalysis],
    visible_part: str,
    visible_issue: str,
) -> str:
    if visible_issue in ("none",):
        return "none"
    if visible_issue == "unknown":
        return "unknown"

    for a in analyses:
        if (
            a.is_usable
            and a.visible_object_part == visible_part
            and a.visible_issue_type == visible_issue
        ):
            return a.visible_severity

    for a in analyses:
        if a.is_usable and a.visible_issue_type == visible_issue:
            return a.visible_severity

    return "unknown"


def _determine_claim_status(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    visible_part: str,
    visible_issue: str,
    override_none_applied: bool = False,
    quality: dict = None,
) -> str:
    if quality is None:
        quality = {}

    # Rule 1: Evidence not sufficient → not_enough_information
    if not evidence.evidence_standard_met:
        return "not_enough_information"

    # Rule 2a: Wrong object entirely (including unknown object detection)
    if fraud.has_wrong_object and not _any_image_shows_right_object(claim, analyses):
        return "contradicted"

    # Rule 2b: Non-original images (stock photos) → NEI
    if fraud.has_non_original_image:
        return "not_enough_information"

    # Rule 2.5: Multi-part claim — each part must have evidence
    if extraction.is_multi_part and extraction.secondary_parts:
        parts_to_check = [extraction.claimed_object_part] + extraction.secondary_parts
        all_parts_have_evidence = True
        for part in parts_to_check:
            part_evident = any(
                (a.visible_object_part == part or part in a.visible_parts_list)
                and a.is_usable
                for a in analyses
            )
            if not part_evident:
                all_parts_have_evidence = False
                break
        if not all_parts_have_evidence:
            return "not_enough_information"

    # Rule 3: Vehicle identity mismatch
    if fraud.has_vehicle_identity_issue:
        return "not_enough_information"

    # Compute part_visible early (used by Rule 3.5 onward)
    part_visible = any(
        a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )

    # Rule 3.5: Benefit of doubt for poor-quality images
    # When VLM says "none" on the claimed part but images are blurry/cropped/low_light,
    # the VLM may have missed subtle damage. If no claim_mismatch exists, trust the user.
    if (
        part_visible
        and visible_issue == "none"
        and extraction.claimed_issue_type not in ("none", "unknown")
        and not fraud.has_claim_mismatch
    ):
        any_poor_quality = any(
            a.is_blurry or a.is_low_light or a.is_cropped
            for a in analyses if a.is_usable
        )
        if any_poor_quality:
            return "supported"

    # Rule 4: Part visible, damage matches claim → supported
    damage_matches = any(
        a.visible_issue_type == extraction.claimed_issue_type
        and a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )
    if not damage_matches and override_none_applied:
        damage_matches = True

    if part_visible and damage_matches:
        return "supported"

    # Rule 5: Some damage visible on claimed part (looser match)
    if part_visible:
        any_damage_on_claimed_part = any(
            a.visible_issue_type not in ("none", "unknown")
            and a.visible_object_part == extraction.claimed_object_part
            and a.is_usable
            for a in analyses
        )
        if any_damage_on_claimed_part:
            return "supported"

    # Rule 6: Part visible but NO damage → contradicted (if evidence met)
    if part_visible and visible_issue == "none":
        if not evidence.evidence_standard_met:
            return "not_enough_information"
        return "contradicted"

    # Rule 7a: Wrong part but damage type matches — supported with visible part
    # (covers wrong_object_part without claim_mismatch, e.g., hinge↔lid, seal↔package_side)
    if (
        fraud.has_wrong_object_part
        and not fraud.has_claim_mismatch
        and visible_issue not in ("none", "unknown")
        and visible_part != extraction.claimed_object_part
        and visible_part != "unknown"
    ):
        return "supported"

    # Rule 7b: Different part has damage (mismatch) → contradicted
    if (
        visible_issue not in ("none", "unknown")
        and visible_part != extraction.claimed_object_part
        and visible_part != "unknown"
    ):
        return "contradicted"

    # Rule 8: Part not visible at all
    if not part_visible:
        any_damage = any(
            a.visible_issue_type not in ("none", "unknown")
            and a.is_usable
            for a in analyses
        )
        if any_damage and fraud.has_claim_mismatch:
            return "contradicted"

        # If object is visible but NO damage is found anywhere (explicitly "none"
        # or "unknown" — meaning VLM couldn't identify any specific damage),
        # the claim is contradicted — we can see the object is undamaged
        any_damage_found = any(
            a.visible_issue_type not in ("none", "unknown", "")
            and a.is_usable
            for a in analyses
        )
        if not any_damage_found and _any_image_shows_right_object(claim, analyses):
            return "contradicted"

        return "not_enough_information"

    return "not_enough_information"


def _any_image_shows_right_object(
    claim: ClaimInput,
    analyses: list[ImageAnalysis],
) -> bool:
    return any(
        a.visible_object_type == claim.claim_object and a.is_usable
        for a in analyses
    )


def _select_supporting_images(
    analyses: list[ImageAnalysis],
    extraction: ClaimExtraction,
    visible_part: str,
    visible_issue: str,
    claim_status: str,
) -> list[str]:
    if not analyses:
        return []

    if claim_status == "supported":
        supporting = []
        for a in sorted(analyses, key=lambda x: (not x.is_blurry, x.confidence), reverse=True):
            if (
                a.is_usable
                and a.visible_object_part == extraction.claimed_object_part
                and a.visible_issue_type not in ("none", "unknown")
                and not a.is_blurry
            ):
                supporting.append(a.image_id)
        if not supporting:
            for a in analyses:
                if a.is_usable and a.visible_issue_type not in ("none", "unknown"):
                    supporting.append(a.image_id)
        return supporting if supporting else [analyses[0].image_id]

    elif claim_status == "contradicted":
        supporting = []
        for a in analyses:
            if a.is_usable:
                supporting.append(a.image_id)
        return supporting if supporting else [analyses[0].image_id]

    else:
        supporting = []
        for a in analyses:
            if a.has_wrong_angle or not a.is_usable:
                continue
            if a.visible_object_type != "unknown":
                supporting.append(a.image_id)
        return supporting


def _merge_risk_flags(
    fraud_flags: list[str],
    quality_flags: list[str],
    user_flags: list[str],
) -> list[str]:
    all_flags = set()
    for f in fraud_flags + quality_flags + user_flags:
        if f and f != "none":
            all_flags.add(f)
    return sorted(all_flags)


def _build_justification(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    claim_status: str,
    visible_part: str,
    visible_issue: str,
    fraud: FraudSignals,
    user_risk_summary: str,
    supporting_ids: list[str],
) -> str:
    parts = []

    if claim_status == "supported":
        if len(supporting_ids) == 1:
            parts.append(
                f"The image directly shows {visible_issue} on the {visible_part}."
            )
        else:
            parts.append(
                f"The image set supports the claim because the "
                f"{visible_part} {visible_issue} is visible."
            )
        blurry_count = sum(1 for a in analyses if a.is_blurry)
        if blurry_count > 0 and len(analyses) > blurry_count:
            clear_ids = [a.image_id for a in analyses if not a.is_blurry and a.is_usable]
            if clear_ids:
                parts.append(
                    f"The clearer image ({clear_ids[0]}) supports the finding."
                )

    elif claim_status == "contradicted":
        if fraud.has_wrong_object:
            parts.append(
                f"The image shows a different object than the claimed "
                f"{claim.claim_object}, so it does not support the claim."
            )
        elif fraud.has_claim_mismatch and visible_issue != extraction.claimed_issue_type:
            parts.append(
                f"The image shows {visible_issue} on the {visible_part} "
                f"rather than the claimed {extraction.claimed_issue_type} on the "
                f"{extraction.claimed_object_part}."
            )
        elif visible_issue == "none":
            parts.append(
                f"The image shows the {visible_part} area but does not show "
                f"clear physical damage, so it contradicts the user's "
                f"{extraction.claimed_issue_type} claim."
            )
        else:
            parts.append(
                f"The visible evidence does not match the claim."
            )

    else:
        if fraud.has_vehicle_identity_issue:
            parts.append(
                f"The submitted images do not reliably support the claim because "
                f"they appear to show different vehicles."
            )
        elif fraud.has_non_original_image and fraud.has_wrong_object:
            parts.append(
                f"The image appears to be a non-original stock photo that does "
                f"not show the claimed {extraction.claimed_object_part} damage."
            )
        elif fraud.has_non_original_image:
            parts.append(
                f"The image appears to be a non-original stock photo and the "
                f"claimed damage cannot be verified from this image."
            )
        elif fraud.has_wrong_object:
            parts.append(
                f"The submitted image does not show the claimed "
                f"{claim.claim_object}."
            )
        else:
            parts.append(
                f"The submitted image does not provide enough evidence to verify "
                f"the {extraction.claimed_issue_type} claim on the "
                f"{extraction.claimed_object_part}."
            )

    if fraud.has_prompt_injection_in_image:
        parts.append(
            "Any instruction-like text inside the image should be ignored."
        )

    if user_risk_summary and "risk" in user_risk_summary.lower():
        parts.append(f"User history: {user_risk_summary}.")

    return " ".join(parts)
