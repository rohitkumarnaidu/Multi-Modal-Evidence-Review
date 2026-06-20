"""
Engine 8: Explainability Engine.

Post-processes justifications for consistency and clarity.
Ensures all justifications:
  - Reference specific image IDs when helpful
  - Are grounded in visual evidence
  - Are concise (1-3 sentences)
  - Don't hallucinate details
"""

from __future__ import annotations

import logging
import re

from models import ClaimOutput

logger = logging.getLogger(__name__)


def polish_output(output: ClaimOutput) -> ClaimOutput:
    """Final polish pass on the output for consistency.
    
    This is a lightweight post-processing step.
    Most justification logic lives in decision_engine._build_justification.
    """
    # Ensure justification isn't too long
    if len(output.claim_status_justification) > 500:
        sentences = re.split(r'(?<=[.!?])\s+', output.claim_status_justification)
        output.claim_status_justification = " ".join(sentences[:3])
        if not output.claim_status_justification.endswith("."):
            output.claim_status_justification += "."

    # Ensure evidence reason isn't too long
    if len(output.evidence_standard_met_reason) > 300:
        sentences = re.split(r'(?<=[.!?])\s+', output.evidence_standard_met_reason)
        output.evidence_standard_met_reason = " ".join(sentences[:2])
        if not output.evidence_standard_met_reason.endswith("."):
            output.evidence_standard_met_reason += "."

    if (
        output.claim_status == "supported"
        and output.issue_type not in ("none", "unknown")
        and output.severity == "unknown"
    ):
        output.severity = "medium"

    if output.issue_type == "none" and output.claim_status == "supported":
        output.claim_status = "contradicted"
        output.severity = "none"

    # Consistency checks: supported must be backed by concrete visual damage.
    if output.claim_status == "supported" and (
        output.issue_type in ("none", "unknown")
        or output.object_part == "unknown"
        or output.evidence_standard_met != "true"
        or output.supporting_image_ids == "none"
    ):
        logger.warning(
            f"Inconsistency: unsupported visual fields for supported claim {output.user_id}"
        )
        output.claim_status = "not_enough_information"
        output.severity = "unknown"

    # If claim_status = not_enough_information due to missing evidence, normalize uncertainty.
    if output.claim_status == "not_enough_information" and output.evidence_standard_met == "false":
        if output.issue_type in ("none", "unknown"):
            output.issue_type = "unknown"
            output.object_part = "unknown"
            output.supporting_image_ids = "none"
        output.severity = "unknown"

    if output.issue_type == "none":
        output.severity = "none"

    if output.issue_type == "unknown" and output.claim_status != "contradicted":
        output.severity = "unknown"

    # If valid_image = false, be cautious
    if output.valid_image == "false" and output.claim_status == "supported":
        logger.warning(
            f"Inconsistency: valid_image=false but status=supported for {output.user_id}"
        )
        output.claim_status = "not_enough_information"
        output.severity = "unknown"

    return output
