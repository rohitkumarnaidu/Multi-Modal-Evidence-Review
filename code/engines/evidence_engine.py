"""
Engine 3: Evidence Sufficiency Engine.

Deterministic check: do the submitted images meet the minimum evidence
requirements from evidence_requirements.csv?

Supports multi-part claims where extraction.is_multi_part is True.
"""

from __future__ import annotations

import logging

from models import (
    ClaimExtraction,
    ClaimInput,
    EvidenceRequirement,
    EvidenceSufficiency,
    ImageAnalysis,
)

logger = logging.getLogger(__name__)


ISSUE_TO_REQUIREMENT_FAMILY = {
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "crack, broken, or missing part",
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
}

PART_TO_REQUIREMENT_FAMILY = {
    "screen": "screen, keyboard, or trackpad",
    "keyboard": "screen, keyboard, or trackpad",
    "trackpad": "screen, keyboard, or trackpad",
    "hinge": "hinge, lid, corner, body, or port",
    "lid": "hinge, lid, corner, body, or port",
    "corner": "hinge, lid, corner, body, or port",
    "body": "hinge, lid, corner, body, or port",
    "base": "hinge, lid, corner, body, or port",
    "port": "hinge, lid, corner, body, or port",
    "contents": "contents or inner item",
    "item": "contents or inner item",
    "label": "water, stain, or label damage",
    "seal": "crushed, torn, or seal damage",
    "package_corner": "crushed, torn, or seal damage",
    "package_side": "crushed, torn, or seal damage",
    "box": "crushed, torn, or seal damage",
}


def check_evidence_sufficiency(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analyses: list[ImageAnalysis],
    requirements: list[EvidenceRequirement],
) -> EvidenceSufficiency:
    evidence = _check_part_evidence(claim, extraction.claimed_object_part, extraction.claimed_issue_type, extraction, image_analyses, requirements)

    if extraction.is_multi_part and extraction.secondary_parts:
        for secondary_part in extraction.secondary_parts:
            secondary_evidence = _check_part_evidence(claim, secondary_part, extraction.claimed_issue_type, extraction, image_analyses, requirements)
            if not secondary_evidence.evidence_standard_met:
                return secondary_evidence

    return evidence


def _check_part_evidence(
    claim: ClaimInput,
    check_part: str,
    check_issue: str,
    extraction: ClaimExtraction,
    image_analyses: list[ImageAnalysis],
    requirements: list[EvidenceRequirement],
) -> EvidenceSufficiency:
    applicable_reqs = _find_applicable_requirements(
        claim.claim_object, check_issue, check_part, requirements,
    )

    claimed_part_visible = any(
        (a.visible_object_part == check_part or check_part in a.visible_parts_list)
        and a.is_usable
        and not a.has_wrong_angle
        for a in image_analyses
    )

    right_object_visible = any(
        a.visible_object_type == claim.claim_object
        and a.is_usable
        for a in image_analyses
    )
    if not right_object_visible:
        right_object_visible = any(
            a.visible_object_type == claim.claim_object
            for a in image_analyses
        )

    all_blurry = all(a.is_blurry for a in image_analyses) if image_analyses else True
    all_unusable = all(not a.is_usable for a in image_analyses) if image_analyses else True
    all_wrong_object = all(
        a.visible_object_type != claim.claim_object
        and a.visible_object_type != "unknown"
        for a in image_analyses
    ) if image_analyses else True

    has_identity_issue = False

    if not image_analyses:
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason="No images were submitted.",
        )

    if all_unusable:
        # Even if all images have quality issues, we may still have valid
        # object/part/issue data from VLM. Only reject if truly no data.
        any_object_info = any(
            a.visible_object_type != "unknown" for a in image_analyses
        )
        if not any_object_info:
            return EvidenceSufficiency(
                evidence_standard_met=False,
                evidence_standard_met_reason="None of the submitted images are usable for review.",
            )

    if all_wrong_object:
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"The submitted images do not show a {claim.claim_object}, "
                f"so the claim cannot be evaluated."
            ),
        )

    if has_identity_issue:
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"The images appear to show different {claim.claim_object}s, "
                f"so the image set does not satisfy evidence requirements."
            ),
        )

    if check_issue == "missing_part" and claimed_part_visible:
        part_visible_with_damage = any(
            (a.visible_object_part == check_part or check_part in a.visible_parts_list)
            and a.is_usable
            and a.visible_issue_type not in ("none", "unknown", "missing_part")
            for a in image_analyses
        )
        if part_visible_with_damage:
            return EvidenceSufficiency(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    f"The image shows the {check_part} with visible damage, "
                    f"which contradicts the claim that it is missing."
                ),
            )
        return EvidenceSufficiency(
            evidence_standard_met=True,
            evidence_standard_met_reason=_build_met_reason(
                check_part, check_issue, image_analyses, all_blurry
            ),
            matched_requirements=[r.requirement_id for r in applicable_reqs],
        )

    if check_part in ("contents", "item"):
        contents_visible = any(
            a.visible_object_part in ("contents", "item")
            or "contents" in a.visible_parts_list
            or "item" in a.visible_parts_list
            for a in image_analyses
            if a.is_usable and not a.has_wrong_angle
        )
        if not contents_visible:
            return EvidenceSufficiency(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    "The images do not clearly show the package contents "
                    "or an opened package view, so the missing-item claim cannot be verified."
                ),
            )

    if claimed_part_visible:
        return EvidenceSufficiency(
            evidence_standard_met=True,
            evidence_standard_met_reason=_build_met_reason(
                check_part, check_issue, image_analyses, all_blurry
            ),
            matched_requirements=[r.requirement_id for r in applicable_reqs],
        )

    if right_object_visible:
        conflicting_damage = any(
            a.is_usable
            and a.visible_object_type == claim.claim_object
            and a.visible_issue_type not in ("none", "unknown", "")
            and a.visible_object_part not in ("unknown", check_part)
            and check_part not in a.visible_parts_list
            for a in image_analyses
        )
        broad_undamaged_view = any(
            a.is_usable
            and a.visible_object_type == claim.claim_object
            and a.visible_issue_type == "none"
            and len(a.visible_parts_list) >= 3
            for a in image_analyses
        )
        if conflicting_damage or broad_undamaged_view:
            return EvidenceSufficiency(
                evidence_standard_met=True,
                evidence_standard_met_reason=(
                    f"The claimed {check_part} is not directly shown, but the submitted "
                    f"{claim.claim_object} view provides enough visible context to evaluate a conflict."
                ),
                matched_requirements=[r.requirement_id for r in applicable_reqs],
            )
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"The submitted images show a {claim.claim_object}, but not enough of the "
                f"claimed {check_part} area to verify the {check_issue} claim."
            ),
        )

    return EvidenceSufficiency(
        evidence_standard_met=True,
        evidence_standard_met_reason=_build_met_reason(
            check_part, check_issue, image_analyses, all_blurry
        ),
        matched_requirements=[r.requirement_id for r in applicable_reqs],
    )


def _find_applicable_requirements(
    claim_object: str,
    issue_type: str,
    object_part: str,
    requirements: list[EvidenceRequirement],
) -> list[EvidenceRequirement]:
    applicable = []
    for req in requirements:
        if req.claim_object not in (claim_object, "all"):
            continue
        if req.applies_to in ("general claim review", "multi-image rows", "reviewability"):
            applicable.append(req)
            continue
        issue_family = ISSUE_TO_REQUIREMENT_FAMILY.get(issue_type, "")
        if issue_family and issue_family in req.applies_to:
            applicable.append(req)
            continue
        part_family = PART_TO_REQUIREMENT_FAMILY.get(object_part, "")
        if part_family and part_family in req.applies_to:
            applicable.append(req)
            continue
        if "identity" in req.applies_to and claim_object == "car":
            applicable.append(req)
    return applicable


def _build_met_reason(
    part: str,
    issue: str,
    analyses: list[ImageAnalysis],
    all_blurry: bool,
) -> str:
    best = None
    for a in analyses:
        if (a.visible_object_part == part or part in a.visible_parts_list) and a.is_usable and not a.is_blurry:
            best = a
            break
    if not best:
        for a in analyses:
            if a.is_usable:
                best = a
                break

    has_any_blurry = any(a.is_blurry for a in analyses)
    if has_any_blurry and len(analyses) > 1:
        clear_ones = [a for a in analyses if not a.is_blurry]
        if clear_ones:
            return (
                f"Some images are blurry, but at least one image clearly "
                f"shows the {part} {issue}."
            )

    if best and best.visible_issue_type not in ("none", "unknown"):
        return (
            f"The {part} is visible and the {issue} can be verified "
            f"from the submitted image."
        )

    return (
        f"The {part} is visible enough to evaluate the claimed condition."
    )
