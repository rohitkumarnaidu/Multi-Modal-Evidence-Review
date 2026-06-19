"""
Engine 6: User Risk Engine.

Loads user_history.csv and propagates risk flags.

CRITICAL RULE (from user correction #6):
  History flags are ALWAYS added to output, even on supported claims.
  History NEVER overrides visual evidence — only adds risk context.
  
  Example: case_031 (sample) — claim is supported but user has 
  user_history_risk → output still gets user_history_risk;manual_review_required.
"""

from __future__ import annotations

import logging

from models import UserHistory

logger = logging.getLogger(__name__)


def get_user_risk_flags(
    user_history: UserHistory | None,
) -> list[str]:
    """Get risk flags from user history.
    
    ALWAYS propagates history flags — never conditional on claim_status.
    
    Returns list of risk flag strings to ADD to the claim's risk_flags.
    """
    if user_history is None:
        logger.debug("No user history found")
        return []

    flags = []

    # Propagate history_flags directly
    for flag in user_history.history_flag_list:
        if flag and flag != "none":
            flags.append(flag)

    # Additional risk signals from metrics (only if not already flagged)
    if "user_history_risk" not in flags:
        # High rejection ratio
        if user_history.rejection_ratio >= 0.4 and user_history.past_claim_count >= 3:
            flags.append("user_history_risk")
            logger.info(
                f"Auto-flagged {user_history.user_id}: "
                f"rejection_ratio={user_history.rejection_ratio:.2f}"
            )
        # Very high frequency
        elif user_history.is_high_frequency and user_history.last_90_days_claim_count >= 5:
            flags.append("user_history_risk")
            logger.info(
                f"Auto-flagged {user_history.user_id}: "
                f"high frequency ({user_history.last_90_days_claim_count} in 90 days)"
            )

    if "manual_review_required" not in flags:
        # Many manual reviews historically
        if (
            user_history.manual_review_claim >= 2
            and user_history.past_claim_count >= 4
        ):
            flags.append("manual_review_required")

    return flags


def get_risk_summary(user_history: UserHistory | None) -> str:
    """Get human-readable risk summary for justification text."""
    if user_history is None:
        return ""

    if not user_history.has_risk and not user_history.needs_manual_review:
        return "the user history does not add risk"

    parts = []
    if user_history.has_risk:
        parts.append(user_history.history_summary)
    if user_history.needs_manual_review:
        parts.append("prior claims required manual review")
    if user_history.rejected_claim > 0:
        parts.append(f"{user_history.rejected_claim} rejected claims")

    return "; ".join(parts)
