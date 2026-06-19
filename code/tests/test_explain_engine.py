from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.explain_engine import polish_output
from models import ClaimOutput


class TestExplainEngine:
    def test_supported_claim_unchanged(self):
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="Part visible",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported", claim_status_justification="The image shows dent on door.",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        result = polish_output(out)
        assert result.claim_status == "supported"
        assert result.severity == "medium"

    def test_nei_sets_severity_unknown(self):
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="false", evidence_standard_met_reason="No images",
            risk_flags="none", issue_type="unknown", object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="No evidence.",
            supporting_image_ids="none", valid_image="false", severity="medium",
        )
        result = polish_output(out)
        assert result.severity == "unknown"

    def test_supported_with_unknown_severity_fixed(self):
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="Part visible",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported",
            claim_status_justification="Shows dent on door.",
            supporting_image_ids="img_1", valid_image="true", severity="unknown",
        )
        result = polish_output(out)
        assert result.severity == "medium"

    def test_inconsistency_issue_none_status_supported(self):
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="none", object_part="door",
            claim_status="supported",
            claim_status_justification="test",
            supporting_image_ids="img_1", valid_image="true", severity="none",
        )
        result = polish_output(out)
        assert result.claim_status == "contradicted"

    def test_long_justification_truncated(self):
        long_text = "Sentence one. " * 100
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason="ok",
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported",
            claim_status_justification=long_text,
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        result = polish_output(out)
        assert len(result.claim_status_justification) <= 500

    def test_long_evidence_reason_truncated(self):
        long_text = "Sentence one. " * 100
        out = ClaimOutput(
            user_id="u1", image_paths="a.jpg", user_claim="test", claim_object="car",
            evidence_standard_met="true", evidence_standard_met_reason=long_text,
            risk_flags="none", issue_type="dent", object_part="door",
            claim_status="supported",
            claim_status_justification="Shows dent.",
            supporting_image_ids="img_1", valid_image="true", severity="medium",
        )
        result = polish_output(out)
        assert len(result.evidence_standard_met_reason) <= 300
