"""Engine 7: evidence-first final decision aggregation."""

from __future__ import annotations

import logging
from collections import Counter

from calibration.issue_calibration import calibrate_issue_type
from calibration.part_map import calibrate_part, is_close_part
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


ISSUE_FAMILIES = [
    {"dent", "scratch"},
    {"crack", "glass_shatter", "broken_part", "missing_part"},
    {"torn_packaging", "crushed_packaging"},
    {"water_damage", "stain"},
]


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
    _calibrate_analyses(claim, extraction, image_analyses)

    supporting_for_support = _supporting_damage_images(claim, extraction, image_analyses)
    claimed_part_visible = _claimed_part_visible(claim, extraction, image_analyses)
    right_object_visible = _any_image_shows_right_object(claim, image_analyses)
    conflicting_damage = _conflicting_damage_images(claim, extraction, image_analyses)
    undamaged_claimed_part = _undamaged_claimed_part_visible(claim, extraction, image_analyses)

    visible_part = _determine_visible_part(claim, extraction, image_analyses, supporting_for_support, conflicting_damage)
    visible_issue = _determine_visible_issue(
        extraction, supporting_for_support, conflicting_damage, undamaged_claimed_part, image_analyses
    )

    claim_status = _determine_claim_status(
        claim=claim,
        extraction=extraction,
        evidence=evidence,
        fraud=fraud,
        quality=quality,
        right_object_visible=right_object_visible,
        supporting_for_support=supporting_for_support,
        conflicting_damage=conflicting_damage,
        undamaged_claimed_part=undamaged_claimed_part,
        claimed_part_visible=claimed_part_visible,
    )

    if claim_status == "not_enough_information" and not evidence.evidence_standard_met:
        if not conflicting_damage and not right_object_visible:
            visible_part = "unknown"
            visible_issue = "unknown"
        elif extraction.claimed_object_part in ("contents", "item"):
            visible_part = "unknown"
            visible_issue = "unknown"
    if claim_status == "contradicted" and fraud.has_wrong_object:
        visible_part = "unknown"
        visible_issue = "unknown"

    raw_severity = _determine_raw_severity(image_analyses, visible_part, visible_issue)
    severity = calibrate_severity(claim.claim_object, visible_part, visible_issue, raw_severity)
    if claim_status == "not_enough_information" and visible_issue == "unknown":
        severity = "unknown"
    if visible_issue == "none":
        severity = "none"

    supporting_ids = _select_supporting_images(
        image_analyses=image_analyses,
        support_images=supporting_for_support,
        conflict_images=conflicting_damage,
        claim_status=claim_status,
        evidence=evidence,
        fraud=fraud,
        visible_issue=visible_issue,
    )

    all_flags = _merge_risk_flags(
        fraud.risk_flags,
        quality.get("quality_flags", []),
        user_risk_flags,
        evidence,
        fraud,
        claim_status,
    )

    justification = _build_justification(
        claim, extraction, claim_status, visible_part, visible_issue,
        evidence, fraud, user_risk_summary, supporting_ids
    )
    valid_image = quality.get("valid_image", True)
    if (
        extraction.claimed_object_part in {"contents", "item"}
        and not evidence.evidence_standard_met
    ):
        valid_image = False

    output = ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=str(evidence.evidence_standard_met).lower(),
        evidence_standard_met_reason=evidence.evidence_standard_met_reason,
        risk_flags=";".join(sorted(set(all_flags))) if all_flags else "none",
        issue_type=visible_issue,
        object_part=visible_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=str(valid_image).lower(),
        severity=severity,
    )

    base_confidence = _confidence_from_flags(all_flags)
    output.confidence_issue_type = base_confidence
    output.confidence_object_part = base_confidence
    output.confidence_claim_status = base_confidence
    output.confidence_severity = base_confidence
    output.confidence_avg = base_confidence

    logger.info(
        "Decision for %s: status=%s, part=%s, issue=%s, severity=%s",
        claim.user_id, claim_status, visible_part, visible_issue, severity,
    )
    return output


def _calibrate_analyses(claim: ClaimInput, extraction: ClaimExtraction, analyses: list[ImageAnalysis]) -> None:
    for a in analyses:
        a.visible_object_part = calibrate_part(
            claim.claim_object, extraction.claimed_object_part, a.visible_object_part
        )
        a.visible_issue_type = calibrate_issue_type(
            claim.claim_object, a.visible_object_part, a.visible_issue_type
        )
        if a.has_text_instruction:
            a.visible_issue_type = "none"
    if any(a.has_text_instruction for a in analyses):
        for a in analyses:
            a.visible_issue_type = "none"


def _issue_compatible(visible_issue: str, claimed_issue: str) -> bool:
    if visible_issue == claimed_issue:
        return True
    if visible_issue in ("none", "unknown") or claimed_issue in ("none", "unknown"):
        return False
    return any(visible_issue in family and claimed_issue in family for family in ISSUE_FAMILIES)


def _part_compatible(claim: ClaimInput, visible_part: str, claimed_part: str, visible_parts: list[str]) -> bool:
    return (
        visible_part == claimed_part
        or claimed_part in visible_parts
        or is_close_part(claim.claim_object, visible_part, claimed_part)
    )


def _has_clear_damage(a: ImageAnalysis) -> bool:
    if not a.is_usable or a.visible_issue_type in ("none", "unknown", ""):
        return False
    return getattr(a, "damage_evidence_level", "clear") in ("clear", "partial")


def _supporting_damage_images(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
) -> list[ImageAnalysis]:
    return [
        a for a in analyses
        if a.visible_object_type == claim.claim_object
        and _has_clear_damage(a)
        and _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
        and _issue_compatible(a.visible_issue_type, extraction.claimed_issue_type)
    ]


def _conflicting_damage_images(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
) -> list[ImageAnalysis]:
    conflicts = []
    for a in analyses:
        if a.visible_object_type != claim.claim_object or not _has_clear_damage(a):
            continue
        part_match = _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
        issue_match = _issue_compatible(a.visible_issue_type, extraction.claimed_issue_type)
        if not part_match or not issue_match:
            conflicts.append(a)
    return conflicts


def _claimed_part_visible(claim: ClaimInput, extraction: ClaimExtraction, analyses: list[ImageAnalysis]) -> bool:
    return any(
        a.is_usable
        and a.visible_object_type == claim.claim_object
        and not a.has_wrong_angle
        and _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
        for a in analyses
    )


def _undamaged_claimed_part_visible(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
) -> bool:
    return any(
        a.is_usable
        and a.visible_object_type == claim.claim_object
        and _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
        and a.visible_issue_type == "none"
        and getattr(a, "damage_evidence_level", "not_visible") in ("not_visible", "clear", "partial")
        for a in analyses
    )


def _determine_claim_status(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    quality: dict,
    right_object_visible: bool,
    supporting_for_support: list[ImageAnalysis],
    conflicting_damage: list[ImageAnalysis],
    undamaged_claimed_part: bool,
    claimed_part_visible: bool,
) -> str:
    if fraud.has_wrong_object:
        return "contradicted"

    if fraud.has_vehicle_identity_issue:
        return "not_enough_information"

    if fraud.has_non_original_image:
        if fraud.has_wrong_object or fraud.has_claim_mismatch or fraud.has_prompt_injection_in_image:
            return "contradicted"
        if not evidence.evidence_standard_met:
            return "not_enough_information"

    if not evidence.evidence_standard_met or not quality.get("valid_image", True):
        if conflicting_damage and not claimed_part_visible:
            return "contradicted"
        return "not_enough_information"

    if supporting_for_support:
        return "supported"

    if undamaged_claimed_part:
        return "contradicted"

    if conflicting_damage:
        part_conflict = any(
            not _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
            for a in conflicting_damage
        )
        if part_conflict:
            return "contradicted"
        if claimed_part_visible:
            return "contradicted"
        return "not_enough_information"

    return "not_enough_information"


def _determine_visible_part(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    support_images: list[ImageAnalysis],
    conflict_images: list[ImageAnalysis],
) -> str:
    for pool in (support_images, conflict_images):
        if pool:
            return pool[0].visible_object_part

    for a in analyses:
        if (
            a.is_usable
            and a.visible_object_type == claim.claim_object
            and _part_compatible(claim, a.visible_object_part, extraction.claimed_object_part, a.visible_parts_list)
        ):
            return extraction.claimed_object_part

    usable_parts = [
        a.visible_object_part for a in analyses
        if a.is_usable and a.visible_object_type == claim.claim_object and a.visible_object_part != "unknown"
    ]
    if usable_parts:
        return Counter(usable_parts).most_common(1)[0][0]
    return "unknown"


def _determine_visible_issue(
    extraction: ClaimExtraction,
    support_images: list[ImageAnalysis],
    conflict_images: list[ImageAnalysis],
    undamaged_claimed_part: bool,
    analyses: list[ImageAnalysis],
) -> str:
    if support_images:
        return support_images[0].visible_issue_type
    if conflict_images:
        return conflict_images[0].visible_issue_type
    if undamaged_claimed_part:
        return "none"
    for a in analyses:
        if a.is_usable and a.visible_issue_type not in ("unknown", ""):
            return a.visible_issue_type
    return "unknown"


def _determine_raw_severity(analyses: list[ImageAnalysis], visible_part: str, visible_issue: str) -> str:
    if visible_issue == "none":
        return "none"
    if visible_issue == "unknown":
        return "unknown"
    for a in analyses:
        if (
            a.is_usable
            and a.visible_object_part == visible_part
            and a.visible_issue_type == visible_issue
            and a.visible_severity in ("low", "medium", "high")
        ):
            return a.visible_severity
    for a in analyses:
        if a.is_usable and a.visible_issue_type == visible_issue and a.visible_severity in ("low", "medium", "high"):
            return a.visible_severity
    return "unknown"


def _any_image_shows_right_object(claim: ClaimInput, analyses: list[ImageAnalysis]) -> bool:
    return any(a.visible_object_type == claim.claim_object and a.is_usable for a in analyses)


def _select_supporting_images(
    image_analyses: list[ImageAnalysis],
    support_images: list[ImageAnalysis],
    conflict_images: list[ImageAnalysis],
    claim_status: str,
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    visible_issue: str,
) -> list[str]:
    if claim_status == "supported":
        return [a.image_id for a in support_images if not a.is_blurry] or [a.image_id for a in support_images]

    if claim_status == "contradicted":
        if conflict_images:
            return [a.image_id for a in conflict_images]
        usable = [a.image_id for a in image_analyses if a.is_usable and not a.has_wrong_angle]
        return usable[:2]

    if fraud.has_vehicle_identity_issue:
        return [a.image_id for a in image_analyses if a.is_usable]
    if not evidence.evidence_standard_met or visible_issue == "unknown":
        return []
    return [a.image_id for a in image_analyses if a.is_usable and not a.has_wrong_angle][:1]


def _merge_risk_flags(
    fraud_flags: list[str],
    quality_flags: list[str],
    user_flags: list[str],
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    claim_status: str,
) -> list[str]:
    flags = {f for f in fraud_flags + quality_flags + user_flags if f and f != "none"}
    if claim_status == "not_enough_information" and not evidence.evidence_standard_met:
        reason = evidence.evidence_standard_met_reason.lower()
        if "angle" in reason:
            flags.add("wrong_angle")
        if "cannot be verified" in reason or "not enough" in reason:
            flags.add("manual_review_required")
    if fraud.has_vehicle_identity_issue or fraud.has_non_original_image:
        flags.add("manual_review_required")
    return sorted(flags)


def _build_justification(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    claim_status: str,
    visible_part: str,
    visible_issue: str,
    evidence: EvidenceSufficiency,
    fraud: FraudSignals,
    user_risk_summary: str,
    supporting_ids: list[str],
) -> str:
    if claim_status == "supported":
        text = f"The image evidence shows {visible_issue} on the {visible_part}, matching the claim."
    elif claim_status == "contradicted":
        if fraud.has_wrong_object:
            text = f"The submitted image evidence shows a different object than the claimed {claim.claim_object}."
        elif visible_issue == "none":
            text = f"The claimed {extraction.claimed_object_part} area is visible, but no physical damage is visible."
        elif visible_part != extraction.claimed_object_part:
            text = f"The visible damage is {visible_issue} on the {visible_part}, not the claimed {extraction.claimed_object_part}."
        else:
            text = f"The visible evidence does not match the claimed {extraction.claimed_issue_type} on the {extraction.claimed_object_part}."
    else:
        if fraud.has_vehicle_identity_issue:
            text = "The images do not reliably show the same vehicle, so the claim cannot be verified."
        elif fraud.has_non_original_image:
            text = "The image appears non-original, so the claimed damage cannot be verified from it."
        else:
            text = evidence.evidence_standard_met_reason or (
                f"The submitted images do not provide enough visual evidence to verify the "
                f"{extraction.claimed_issue_type} claim on the {extraction.claimed_object_part}."
            )

    if supporting_ids:
        text += f" Relevant image IDs: {';'.join(supporting_ids)}."
    if fraud.has_prompt_injection_in_image:
        text += " Instruction-like text in the image was ignored."
    if user_risk_summary and "risk" in user_risk_summary.lower():
        text += f" User history: {user_risk_summary}."
    return text


def _confidence_from_flags(flags: list[str]) -> float:
    risk_deduction = 0.0
    for flag in flags:
        if flag in ("blurry_image", "cropped_or_obstructed", "low_light_or_glare"):
            risk_deduction = max(risk_deduction, 0.1)
        elif flag in ("wrong_angle", "damage_not_visible"):
            risk_deduction = max(risk_deduction, 0.2)
        elif flag in ("wrong_object", "wrong_object_part", "claim_mismatch"):
            risk_deduction = max(risk_deduction, 0.3)
        elif flag in ("possible_manipulation", "non_original_image"):
            risk_deduction = max(risk_deduction, 0.4)
    return 1.0 * (1 - risk_deduction)
