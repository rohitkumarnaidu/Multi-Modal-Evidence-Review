"""
Engine 1: Claim Understanding Engine.

Extracts the actual damage claim from conversation transcript.
Text-only LLM call — no images.

Handles:
  - Multi-language (Hindi, Spanish, Chinese)
  - Verbose/distracted users
  - Prompt injection in conversation text
  - Multi-part claims
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from config import (
    ISSUE_TYPES,
    OBJECT_PARTS_BY_TYPE,
    PROMPT_INJECTION_KEYWORDS,
)
from models import ClaimExtraction, ClaimInput

logger = logging.getLogger(__name__)


def extract_claim_text_only(claim: ClaimInput) -> ClaimExtraction:
    """Pre-scan conversation for deterministic signals before LLM call."""
    conversation = claim.user_claim.lower()

    # Detect prompt injection via regex
    has_injection = False
    injection_detail = ""
    for pattern in PROMPT_INJECTION_KEYWORDS:
        if re.search(pattern, conversation, re.IGNORECASE):
            has_injection = True
            injection_detail = f"Detected pattern: {pattern}"
            logger.warning(f"Prompt injection detected in {claim.user_id}: {pattern}")
            break

    return ClaimExtraction(
        has_prompt_injection=has_injection,
        prompt_injection_detail=injection_detail,
    )


def extract_claim_with_llm(
    claim: ClaimInput,
    llm_client,
) -> ClaimExtraction:
    """Full claim extraction using LLM (Call 1 of 2-call design).
    
    Combines deterministic pre-scan with LLM reasoning.
    """
    from llm.prompts import build_claim_extraction_prompt

    # Step 1: Deterministic pre-scan
    pre_scan = extract_claim_text_only(claim)

    # Step 2: LLM extraction (ensemble with self-consistency when enabled)
    prompt = build_claim_extraction_prompt(claim.user_claim, claim.claim_object)
    if hasattr(llm_client, "call_text_ensemble"):
        result = llm_client.call_text_ensemble(prompt)
    else:
        result = llm_client.call_text(prompt)

    if result is None:
        logger.error(f"LLM claim extraction failed for {claim.user_id}")
        return pre_scan

    # Step 3: Merge results
    allowed_parts = OBJECT_PARTS_BY_TYPE.get(claim.claim_object, set())

    claimed_part = result.get("claimed_object_part", "unknown")
    if claimed_part not in allowed_parts:
        claimed_part = _fuzzy_match_part(claimed_part, allowed_parts)

    claimed_issue = result.get("claimed_issue_type", "unknown")
    if claimed_issue not in ISSUE_TYPES:
        claimed_issue = _fuzzy_match_issue(claimed_issue)

    extraction = ClaimExtraction(
        reasoning=result.get("reasoning", ""),
        claimed_issue_type=claimed_issue,
        claimed_object_part=claimed_part,
        claimed_severity_hint=result.get("claimed_severity_hint", "unknown"),
        claim_summary=result.get("claim_summary", ""),
        has_prompt_injection=pre_scan.has_prompt_injection or result.get("has_prompt_injection", False),
        prompt_injection_detail=pre_scan.prompt_injection_detail or result.get("prompt_injection_detail", ""),
        is_multi_part=result.get("is_multi_part", False),
        secondary_parts=[
            p for p in result.get("secondary_parts", [])
            if p in allowed_parts
        ],
    )

    logger.info(
        f"Claim extracted for {claim.user_id}: "
        f"part={extraction.claimed_object_part}, "
        f"issue={extraction.claimed_issue_type}, "
        f"multi_part={extraction.is_multi_part}"
    )
    return extraction


def _fuzzy_match_part(raw: str, allowed: set[str]) -> str:
    """Best-effort normalization of object_part to allowed values."""
    raw_lower = raw.lower().strip().replace(" ", "_")
    if raw_lower in allowed:
        return raw_lower

    # Common aliases
    aliases = {
        "bumper_front": "front_bumper",
        "bumper_rear": "rear_bumper",
        "rear_bumper_area": "rear_bumper",
        "front_bumper_area": "front_bumper",
        "mirror": "side_mirror",
        "left_mirror": "side_mirror",
        "right_mirror": "side_mirror",
        "head_light": "headlight",
        "tail_light": "taillight",
        "back_light": "taillight",
        "front_glass": "windshield",
        "windscreen": "windshield",
        "panel": "body",
        "body_panel": "body",
        "side_panel": "body",
        "quarter": "quarter_panel",
        "display": "screen",
        "monitor": "screen",
        "lcd": "screen",
        "palm_rest": "trackpad",
        "touchpad": "trackpad",
        "keys": "keyboard",
        "keycap": "keyboard",
        "shell": "body",
        "casing": "body",
        "cover": "lid",
        "top_cover": "lid",
        "box_corner": "package_corner",
        "flap": "seal",
        "tape": "seal",
        "wrapping": "box",
        "product": "item",
        "inside_item": "item",
        "inner_item": "item",
        # Hindi/regional car parts
        "gaadi": "car",
        "bonnet": "hood",
        "darwaza": "door",
        "shisha": "windshield",
        "dikky": "body",
        "pahiya": "body",
        "bumper": "front_bumper",
        "headlight": "headlight",
        "tailight": "taillight",
        # Hindi/regional laptop parts
        "parda": "screen",
        "chavi": "keyboard",
        # Hindi/regional package parts
        "dabba": "box",
        "sthaniya": "package_side",
    }
    if raw_lower in aliases and aliases[raw_lower] in allowed:
        return aliases[raw_lower]

    # Partial match
    for part in allowed:
        if part in raw_lower or raw_lower in part:
            return part

    return "unknown"


def _fuzzy_match_issue(raw: str) -> str:
    """Best-effort normalization of issue_type."""
    raw_lower = raw.lower().strip().replace(" ", "_")
    if raw_lower in ISSUE_TYPES:
        return raw_lower

    aliases = {
        "dented": "dent",
        "scratched": "scratch",
        "cracked": "crack",
        "shattered": "glass_shatter",
        "broken": "broken_part",
        "missing": "missing_part",
        "torn": "torn_packaging",
        "crushed": "crushed_packaging",
        "wet": "water_damage",
        "liquid_damage": "water_damage",
        "stained": "stain",
        "hail_damage": "dent",
        "hail_dent": "dent",
        # Hindi/regional
        "kharoch": "scratch",
        "kharochna": "scratch",
        "dent": "dent",
        "darar": "crack",
        "tuta": "broken_part",
        "tuta_hua": "broken_part",
        "gayab": "missing_part",
        "fatna": "torn_packaging",
        "pani": "water_damage",
        "gilaa": "water_damage",
        "daag": "stain",
    }
    if raw_lower in aliases:
        return aliases[raw_lower]

    return "unknown"
