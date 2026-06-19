from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.risk_engine import get_user_risk_flags, get_risk_summary
from models import UserHistory


class TestUserRisk:
    def test_no_history(self):
        flags = get_user_risk_flags(None)
        assert flags == []

    def test_no_risk_flags_in_history(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=1, accept_claim=1,
            manual_review_claim=0, rejected_claim=0, last_90_days_claim_count=0,
            history_flags="none", history_summary="clean record",
        )
        flags = get_user_risk_flags(uh)
        assert flags == []

    def test_propagates_user_history_risk_flag(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=3,
            history_flags="user_history_risk", history_summary="flagged for review",
        )
        flags = get_user_risk_flags(uh)
        assert "user_history_risk" in flags

    def test_propagates_manual_review_required(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=3,
            history_flags="manual_review_required", history_summary="needs manual review",
        )
        flags = get_user_risk_flags(uh)
        assert "manual_review_required" in flags

    def test_propagates_multiple_flags(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=3,
            history_flags="user_history_risk;manual_review_required",
            history_summary="high risk user with prior manual reviews",
        )
        flags = get_user_risk_flags(uh)
        assert "user_history_risk" in flags
        assert "manual_review_required" in flags

    def test_high_rejection_ratio_triggers_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=10, accept_claim=2,
            manual_review_claim=2, rejected_claim=6, last_90_days_claim_count=2,
            history_flags="none", history_summary="",
        )
        flags = get_user_risk_flags(uh)
        assert "user_history_risk" in flags

    def test_high_frequency_triggers_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=10, accept_claim=5,
            manual_review_claim=2, rejected_claim=3, last_90_days_claim_count=6,
            history_flags="none", history_summary="",
        )
        flags = get_user_risk_flags(uh)
        assert "user_history_risk" in flags

    def test_low_frequency_no_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=2, accept_claim=2,
            manual_review_claim=0, rejected_claim=0, last_90_days_claim_count=1,
            history_flags="none", history_summary="",
        )
        flags = get_user_risk_flags(uh)
        assert flags == []

    def test_high_manual_review_triggers_flag(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=1,
            manual_review_claim=3, rejected_claim=1, last_90_days_claim_count=2,
            history_flags="none", history_summary="",
        )
        flags = get_user_risk_flags(uh)
        assert "manual_review_required" in flags

    def test_history_flag_trumps_auto_detection(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=10, accept_claim=2,
            manual_review_claim=2, rejected_claim=6, last_90_days_claim_count=6,
            history_flags="user_history_risk", history_summary="already flagged",
        )
        flags = get_user_risk_flags(uh)
        assert flags.count("user_history_risk") == 1


class TestRiskSummary:
    def test_no_history(self):
        assert get_risk_summary(None) == ""

    def test_no_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=1, accept_claim=1,
            manual_review_claim=0, rejected_claim=0, last_90_days_claim_count=0,
            history_flags="none", history_summary="clean",
        )
        assert "does not add risk" in get_risk_summary(uh)

    def test_has_risk(self):
        uh = UserHistory(
            user_id="u1", past_claim_count=5, accept_claim=2,
            manual_review_claim=1, rejected_claim=2, last_90_days_claim_count=3,
            history_flags="user_history_risk", history_summary="prior suspicious claims",
        )
        summary = get_risk_summary(uh)
        assert "prior suspicious claims" in summary
