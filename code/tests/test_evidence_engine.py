from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.evidence_engine import check_evidence_sufficiency
from models import (
    ClaimExtraction, ClaimInput, EvidenceRequirement, ImageAnalysis,
)


def make_req(req_id: str, obj: str, applies: str, min_ev: str = "1 clear image") -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=req_id, claim_object=obj,
        applies_to=applies, minimum_image_evidence=min_ev,
    )


REQUIREMENTS = [
    make_req("REQ_CAR_IDENTITY_OR_SIDE", "car", "general claim review"),
    make_req("REQ_DENT_SCRATCH", "car", "dent or scratch"),
    make_req("REQ_WINDSHIELD", "car", "crack, broken, or missing part"),
    make_req("REQ_LAPTOP_GENERAL", "laptop", "screen, keyboard, or trackpad"),
    make_req("REQ_LAPTOP_STRUCTURAL", "laptop", "hinge, lid, corner, body, or port"),
    make_req("REQ_PACKAGE_EXTERNAL", "package", "crushed, torn, or seal damage"),
]


class TestEvidenceSufficiency:
    def test_basic_supported(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="car", visible_object_part="door",
                          visible_issue_type="dent"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is True

    def test_no_images(self):
        claim = ClaimInput(user_id="u1", image_paths="", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        result = check_evidence_sufficiency(claim, extraction, [], REQUIREMENTS)
        assert result.evidence_standard_met is False
        assert "No images" in result.evidence_standard_met_reason

    def test_all_unusable_no_object_info(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=False,
                          visible_object_type="unknown", visible_object_part="unknown"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is False

    def test_all_wrong_object(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="laptop", visible_object_part="screen"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is False
        assert "do not show a car" in result.evidence_standard_met_reason

    def test_right_object_wrong_part(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="hood")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="car", visible_object_part="front_bumper",
                          visible_issue_type="dent"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is True

    def test_missing_part_claim_with_visible_part(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="missing", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="missing_part", claimed_object_part="side_mirror")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="car", visible_object_part="side_mirror",
                          visible_issue_type="broken_part"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is False

    def test_blurry_image_with_clear_backup(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True, is_blurry=True,
                          visible_object_type="car", visible_object_part="door"),
            ImageAnalysis(image_id="img_2", image_path="b.jpg", is_usable=True, is_blurry=False,
                          visible_object_type="car", visible_object_part="door",
                          visible_issue_type="dent"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is True

    def test_multi_part_claim(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(
            claimed_issue_type="dent", claimed_object_part="door",
            is_multi_part=True, secondary_parts=["hood"],
        )
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="car", visible_object_part="door",
                          visible_issue_type="dent"),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is True

    def test_watermark_does_not_block_evidence(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="dent", claim_object="car")
        extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          visible_object_type="car", visible_object_part="door",
                          visible_issue_type="dent", has_watermark=True),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is True

    def test_missing_contents_requires_contents_view(self):
        claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="item missing", claim_object="package")
        extraction = ClaimExtraction(claimed_issue_type="missing_part", claimed_object_part="contents")
        analyses = [
            ImageAnalysis(
                image_id="img_1", image_path="a.jpg", is_usable=True,
                visible_object_type="package", visible_object_part="box",
                visible_issue_type="none",
            ),
        ]
        result = check_evidence_sufficiency(claim, extraction, analyses, REQUIREMENTS)
        assert result.evidence_standard_met is False
        assert "contents" in result.evidence_standard_met_reason
