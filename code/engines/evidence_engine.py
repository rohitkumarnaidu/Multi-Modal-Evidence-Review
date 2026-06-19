"""
Engine 3: Evidence Sufficiency Engine.

Deterministic check: do the submitted images meet the minimum evidence
requirements from evidence_requirements.csv?

Key rule: evidence_standard_met is about whether the CLAIMED PART is
visible in at least one image — not just any part of the object.
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


# Map issue types to requirement families
ISSUE_TO_REQUIREMENT_FAMILY = {
    # Car
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "crack, broken, or missing part",
    # Laptop
    # (laptop uses part-based requirements)
    # Package
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
}

# Map parts to requirement families (for laptops)
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
    # Package
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
    """Check if submitted images meet evidence requirements.
    
    Logic:
      1. Find applicable requirements for this claim_object + issue family
      2. Check if ANY image shows the claimed part clearly enough
      3. Handle edge cases: wrong object, all blurry, wrong angle
    """
    # Step 1: Identify relevant requirements
    applicable_reqs = _find_applicable_requirements(
        claim.claim_object,
        extraction.claimed_issue_type,
        extraction.claimed_object_part,
        requirements,
    )

    # Step 2: Check if claimed part is visible in any image
    claimed_part_visible = any(
        a.visible_object_part == extraction.claimed_object_part
        and a.is_usable
        and not a.has_wrong_angle
        for a in image_analyses
    )

    # Also check if at least one image shows the right object type
    right_object_visible = any(
        a.visible_object_type == claim.claim_object
        and a.is_usable
        for a in image_analyses
    )

    # Step 3: Check for disqualifying conditions
    all_blurry = all(a.is_blurry for a in image_analyses) if image_analyses else True
    all_unusable = all(not a.is_usable for a in image_analyses) if image_analyses else True
    all_wrong_object = all(
        a.visible_object_type != claim.claim_object
        and a.visible_object_type != "unknown"
        for a in image_analyses
    ) if image_analyses else True

    # For vehicle identity issues
    has_identity_issue = False
    if claim.claim_object == "car" and len(image_analyses) > 1:
        colors = [a.vehicle_color.lower().strip() for a in image_analyses if a.vehicle_color]
        if len(set(colors)) > 1 and "" not in [c for c in colors]:
            has_identity_issue = True

    # Step 4: Determine evidence sufficiency
    if not image_analyses:
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason="No images were submitted.",
        )

    if all_unusable:
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
                "The submitted images appear to show different vehicles, "
                "so the image set does not satisfy vehicle identity evidence."
            ),
            matched_requirements=["REQ_CAR_IDENTITY_OR_SIDE"],
        )

    if claimed_part_visible:
        return EvidenceSufficiency(
            evidence_standard_met=True,
            evidence_standard_met_reason=_build_met_reason(
                extraction, image_analyses, all_blurry
            ),
            matched_requirements=[r.requirement_id for r in applicable_reqs],
        )

    # Claimed part not visible but object is
    if right_object_visible:
        # Special case: if damage IS visible but on a different part
        # Evidence is sufficient to evaluate (may lead to contradicted)
        any_damage_visible = any(
            a.visible_issue_type not in ("none", "unknown")
            and a.is_usable
            for a in image_analyses
        )
        if any_damage_visible:
            return EvidenceSufficiency(
                evidence_standard_met=True,
                evidence_standard_met_reason=(
                    f"The submitted image is sufficient to see that the visible "
                    f"damage does not match the claimed {extraction.claimed_object_part} "
                    f"{extraction.claimed_issue_type}."
                ),
                matched_requirements=[r.requirement_id for r in applicable_reqs],
            )

        # Part not visible at all
        return EvidenceSufficiency(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"The image does not show the {extraction.claimed_object_part}, "
                f"so the claimed {extraction.claimed_issue_type} cannot be verified."
            ),
        )

    # Fallback: Check for specific content-related issues
    if extraction.claimed_object_part in ("contents", "item"):
        contents_visible = any(
            a.visible_object_part in ("contents", "item")
            and a.is_usable
            for a in image_analyses
        )
        if not contents_visible:
            return EvidenceSufficiency(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    "The images do not clearly show the expected contents "
                    "or enough of the opened package to verify whether anything is missing."
                ),
            )

    return EvidenceSufficiency(
        evidence_standard_met=True,
        evidence_standard_met_reason=_build_met_reason(
            extraction, image_analyses, all_blurry
        ),
        matched_requirements=[r.requirement_id for r in applicable_reqs],
    )


def _find_applicable_requirements(
    claim_object: str,
    issue_type: str,
    object_part: str,
    requirements: list[EvidenceRequirement],
) -> list[EvidenceRequirement]:
    """Find evidence requirements applicable to this claim."""
    applicable = []
    for req in requirements:
        # Match by object
        if req.claim_object not in (claim_object, "all"):
            continue
        # General requirements always apply
        if req.applies_to in ("general claim review", "multi-image rows", "reviewability"):
            applicable.append(req)
            continue
        # Match by issue family
        issue_family = ISSUE_TO_REQUIREMENT_FAMILY.get(issue_type, "")
        if issue_family and issue_family in req.applies_to:
            applicable.append(req)
            continue
        # Match by part family
        part_family = PART_TO_REQUIREMENT_FAMILY.get(object_part, "")
        if part_family and part_family in req.applies_to:
            applicable.append(req)
            continue
        # Vehicle identity requirement
        if "identity" in req.applies_to and claim_object == "car":
            applicable.append(req)
    return applicable


def _build_met_reason(
    extraction: ClaimExtraction,
    analyses: list[ImageAnalysis],
    all_blurry: bool,
) -> str:
    """Build evidence_standard_met_reason when standard IS met."""
    part = extraction.claimed_object_part
    issue = extraction.claimed_issue_type

    # Find the best image
    best = None
    for a in analyses:
        if a.visible_object_part == part and a.is_usable and not a.is_blurry:
            best = a
            break
    if not best:
        for a in analyses:
            if a.is_usable:
                best = a
                break

    if all_blurry and len(analyses) > 1:
        clear_ones = [a for a in analyses if not a.is_blurry]
        if clear_ones:
            return (
                f"One image is blurry, but the second image clearly "
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
