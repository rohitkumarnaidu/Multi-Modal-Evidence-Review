"""Pipeline — single-claim processing wrapper for API and reuse."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EVIDENCE_REQUIREMENTS_CSV
from data_loader import load_evidence_requirements, load_user_history
from engines.claim_engine import extract_claim_with_llm
from engines.decision_engine import make_decision
from engines.evidence_engine import check_evidence_sufficiency
from engines.explain_engine import polish_output
from engines.fraud_engine import detect_fraud
from engines.quality_engine import assess_image_quality
from engines.risk_engine import get_risk_summary, get_user_risk_flags
from engines.vision_engine import analyze_all_images
from llm.multi_provider_client import MultiProviderClient
from models import ClaimInput, ClaimOutput

logger = logging.getLogger(__name__)

_EVIDENCE_REQS = None
_USER_HISTORY = None


def _ensure_data():
    global _EVIDENCE_REQS, _USER_HISTORY
    if _EVIDENCE_REQS is None:
        _EVIDENCE_REQS = load_evidence_requirements()
    if _USER_HISTORY is None:
        _USER_HISTORY = load_user_history()
    return _EVIDENCE_REQS, _USER_HISTORY


def process_claim(
    claim: ClaimInput,
    llm_client: MultiProviderClient | None = None,
) -> ClaimOutput:
    evidence_reqs, user_history = _ensure_data()
    if llm_client is None:
        llm_client = MultiProviderClient()

    try:
        extraction = extract_claim_with_llm(claim, llm_client)
        image_analyses = analyze_all_images(claim, llm_client)

        for a in image_analyses:
            a.shows_claimed_part = a.visible_object_part == extraction.claimed_object_part
            a.shows_claimed_damage = (
                a.visible_issue_type == extraction.claimed_issue_type
                and a.visible_issue_type not in ("none", "unknown")
            )

        evidence = check_evidence_sufficiency(claim, extraction, image_analyses, evidence_reqs)
        quality = assess_image_quality(image_analyses)
        fraud = detect_fraud(claim, extraction, image_analyses, llm_client)

        user = user_history.get(claim.user_id)
        user_risk_flags = get_user_risk_flags(user)
        user_risk_summary = get_risk_summary(user)

        output = make_decision(
            claim=claim,
            extraction=extraction,
            image_analyses=image_analyses,
            evidence=evidence,
            fraud=fraud,
            quality=quality,
            user_risk_flags=user_risk_flags,
            user_risk_summary=user_risk_summary,
        )

        return polish_output(output)

    except Exception as e:
        logger.error(f"Error processing claim {claim.user_id}: {e}", exc_info=True)
        return ClaimOutput(
            user_id=claim.user_id,
            image_paths=claim.image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met="false",
            evidence_standard_met_reason=f"Processing error: {str(e)[:100]}",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="The claim could not be processed automatically.",
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
        )
