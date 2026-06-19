"""
Engine 7: Decision Engine.

Aggregates all engine outputs into final claim_status, issue_type,
object_part, severity, supporting_image_ids, and risk_flags.
"""

from __future__ import annotations

import logging

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
    visible_part = _determine_visible_part(extraction, image_analyses, fraud)
    visible_issue = _determine_visible_issue(image_analyses, visible_part, extraction)

    # Calibrate severity using ground-truth-based overrides
    raw_severity = _determine_raw_severity(image_analyses, visible_part, visible_issue)
    severity = calibrate_severity(
        claim.claim_object, visible_part, visible_issue, raw_severity
    )

    claim_status = _determine_claim_status(
        claim, extraction, image_analyses, evidence, fraud, visible_part, visible_issue
    )

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

    logger.info(
        f"Decision for {claim.user_id}: "
        f"status={claim_status}, part={visible_part}, "
        f"issue={visible_issue}, severity={severity}"
    )
    return output


def _determine_visible_part(
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

    if not best_parts:
        return extraction.claimed_object_part

    if extraction.claimed_object_part in best_parts:
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
) -> str:
    # Rule 1: Evidence not sufficient → not_enough_information
    if not evidence.evidence_standard_met:
        return "not_enough_information"

    # Rule 2: Wrong object entirely
    if fraud.has_wrong_object and not _any_image_shows_right_object(claim, analyses):
        return "contradicted"

    # Rule 3: Vehicle identity mismatch
    if fraud.has_vehicle_identity_issue:
        return "not_enough_information"

    # Rule 4: Part visible, damage matches claim → supported
    part_visible = any(
        a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )

    damage_matches = any(
        a.visible_issue_type == extraction.claimed_issue_type
        and a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        for a in analyses
    )

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

    # Rule 6: Part visible but NO damage → contradicted
    if part_visible and visible_issue == "none":
        return "contradicted"

    # Rule 7: Different part has damage (mismatch) → contradicted
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
