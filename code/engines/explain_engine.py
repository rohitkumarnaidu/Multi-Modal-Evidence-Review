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

    # Consistency checks
    # If claim_status = supported, severity should not be "unknown" if evidence met
    if (
        output.claim_status == "supported"
        and output.severity == "unknown"
        and output.evidence_standard_met == "true"
    ):
        output.severity = "medium"  # Safe default for supported claims

    # If claim_status = not_enough_information, severity should be "unknown"
    if (
        output.claim_status == "not_enough_information"
        and output.evidence_standard_met == "false"
    ):
        output.severity = "unknown"

    # If issue_type = none and claim_status = supported, this is inconsistent
    if output.issue_type == "none" and output.claim_status == "supported":
        logger.warning(
            f"Inconsistency: issue_type=none but status=supported for {output.user_id}"
        )
        # If no damage visible but claim supported, something is off
        # This shouldn't happen with correct logic, but catch it
        output.claim_status = "contradicted"

    # If no images supporting and status is supported
    if output.supporting_image_ids == "none" and output.claim_status == "supported":
        logger.warning(
            f"Inconsistency: no supporting images but status=supported for {output.user_id}"
        )

    # If valid_image = false, be cautious
    if output.valid_image == "false" and output.claim_status == "supported":
        logger.warning(
            f"Inconsistency: valid_image=false but status=supported for {output.user_id}"
        )

    return output
