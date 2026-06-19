from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ClaimInput, ClaimOutput, ImageAnalysis, UserHistory


class TestClaimInput:
    def test_valid_claim_object(self):
        c = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="test", claim_object="car")
        assert c.claim_object == "car"

    def test_image_path_list(self):
        c = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="test", claim_object="car")
        assert c.image_path_list == ["a.jpg", "b.jpg"]

    def test_image_path_list_single(self):
        c = ClaimInput(user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car")
        assert c.image_path_list == ["a.jpg"]

    def test_image_ids(self):
        c = ClaimInput(user_id="u1", image_paths="a.jpg;b.jpg", user_claim="test", claim_object="car")
        assert c.image_ids == ["a", "b"]

    def test_image_ids_with_path(self):
        c = ClaimInput(user_id="u1", image_paths="images/test/case_001/img_1.jpg", user_claim="test", claim_object="car")
        assert c.image_ids == ["img_1"]


class TestClaimOutput:
    def test_to_csv_row_has_all_columns(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        row = o.to_csv_row()
        expected = {
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason",
            "risk_flags", "issue_type", "object_part", "claim_status",
            "claim_status_justification", "supporting_image_ids",
            "valid_image", "severity",
        }
        assert set(row.keys()) == expected

    def test_invalid_claim_status_normalized(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="INVALID", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        assert o.claim_status == "not_enough_information"

    def test_invalid_issue_type_normalized(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="overload_damage", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        assert o.issue_type == "unknown"

    def test_risk_flags_normalization(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="blurry_image;manual_review_required;user_history_risk",
            issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        assert o.normalize_risk_flags() == "blurry_image;manual_review_required;user_history_risk"

    def test_risk_flags_deduplicated(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="blurry_image;blurry_image;manual_review_required",
            issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        result = o.normalize_risk_flags()
        assert result.count("blurry_image") == 1

    def test_risk_flags_invalid_dropped(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="blurry_image;made_up_flag;invalid",
            issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        assert o.normalize_risk_flags() == "blurry_image"

    def test_risk_flags_none(self):
        o = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        assert o.normalize_risk_flags() == "none"


class TestImageAnalysis:
    def test_shows_claimed_part(self):
        a = ImageAnalysis(
            image_id="img_1", image_path="t.jpg",
            visible_object_part="door",
        )
        a.shows_claimed_part = (a.visible_object_part == "door")
        assert a.shows_claimed_part is True

    def test_shows_claimed_damage(self):
        a = ImageAnalysis(
            image_id="img_1", image_path="t.jpg",
            visible_issue_type="dent",
        )
        a.shows_claimed_damage = (
            a.visible_issue_type == "dent"
            and a.visible_issue_type not in ("none", "unknown")
        )
        assert a.shows_claimed_damage is True


class TestUserHistory:
    def test_history_flag_list(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=3,
            history_flags="user_history_risk;manual_review_required",
        )
        assert "user_history_risk" in uh.history_flag_list
        assert "manual_review_required" in uh.history_flag_list

    def test_history_flag_list_none(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=0, history_flags="none",
        )
        assert uh.history_flag_list == []

    def test_rejection_ratio(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=10, accept_claim=3,
            manual_review_claim=2, rejected_claim=5, last_90_days_claim_count=2,
            history_flags="none",
        )
        assert uh.rejection_ratio == 0.5

    def test_rejection_ratio_zero_claims(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=0, history_flags="none",
        )
        assert uh.rejection_ratio == 0.0

    def test_is_high_frequency(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=5,
            history_flags="none",
        )
        assert uh.is_high_frequency is True

    def test_not_high_frequency(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=2,
            history_flags="none",
        )
        assert uh.is_high_frequency is False

    def test_has_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=0, history_flags="user_history_risk",
        )
        assert uh.has_risk is True

    def test_needs_manual_review(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=0, history_flags="manual_review_required",
        )
        assert uh.needs_manual_review is True
