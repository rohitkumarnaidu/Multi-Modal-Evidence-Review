from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.decision_engine import make_decision
from models import (
    ClaimExtraction, ClaimInput, ClaimOutput, EvidenceSufficiency,
    FraudSignals, ImageAnalysis,
)


def test_supported_claim():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="door dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent", visible_severity="medium"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals()
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "supported"
    assert output.object_part == "door"
    assert output.issue_type == "dent"


def test_contradicted_wrong_part():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="hood scratch", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="scratch", claimed_object_part="hood")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="front_bumper",
                      visible_issue_type="broken_part", visible_severity="high"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals(has_claim_mismatch=True, risk_flags=["claim_mismatch"])
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "contradicted"
    assert output.object_part == "front_bumper"
    assert output.issue_type == "broken_part"


def test_contradicted_no_damage():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="door dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="none", visible_severity="none"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals(damage_not_visible=True, risk_flags=["damage_not_visible"])
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "contradicted"
    assert output.issue_type == "none"


def test_not_enough_information_evidence_not_met():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="door dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=False,
                      visible_object_type="unknown", visible_object_part="unknown"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=False, evidence_standard_met_reason="no usable images")
    fraud = FraudSignals()
    quality = {"valid_image": False, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "not_enough_information"


def test_wrong_object_contradicted():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="laptop dent", claim_object="laptop")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="body")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals(has_wrong_object=True, risk_flags=["wrong_object"])
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "contradicted"


def test_vehicle_identity_not_enough_info():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="car dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent", vehicle_color="blue"),
        ImageAnalysis(image_id="img_2", image_path="b.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent", vehicle_color="red"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals(has_vehicle_identity_issue=True, risk_flags=["claim_mismatch"])
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "not_enough_information"


def test_user_risk_flags_propagate():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="door dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent", visible_severity="medium"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals()
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality,
                           ["user_history_risk", "manual_review_required"], "flagged history")
    assert "user_history_risk" in output.risk_flags
    assert "manual_review_required" in output.risk_flags


def test_supporting_images_for_supported():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="door dent", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="dent", claimed_object_part="door")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True, is_blurry=False,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent"),
        ImageAnalysis(image_id="img_2", image_path="b.jpg", is_usable=True, is_blurry=False,
                      visible_object_type="car", visible_object_part="door",
                      visible_issue_type="dent"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals()
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.claim_status == "supported"
    assert "img_1" in output.supporting_image_ids or "img_2" in output.supporting_image_ids


def test_override_none_with_calibration():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="windshield crack", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="crack", claimed_object_part="windshield")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="windshield",
                      visible_issue_type="none", visible_severity="none"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals()
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.issue_type == "crack", f"Expected crack, got {output.issue_type}"
    assert output.claim_status == "supported"


def test_calibration_glass_shatter_to_crack():
    claim = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="windshield crack", claim_object="car")
    extraction = ClaimExtraction(claimed_issue_type="crack", claimed_object_part="windshield")
    analyses = [
        ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                      visible_object_type="car", visible_object_part="windshield",
                      visible_issue_type="glass_shatter", visible_severity="high"),
    ]
    evidence = EvidenceSufficiency(evidence_standard_met=True, evidence_standard_met_reason="ok")
    fraud = FraudSignals()
    quality = {"valid_image": True, "quality_flags": []}
    output = make_decision(claim, extraction, analyses, evidence, fraud, quality, [], "")
    assert output.issue_type == "crack", f"Expected crack, got {output.issue_type}"
