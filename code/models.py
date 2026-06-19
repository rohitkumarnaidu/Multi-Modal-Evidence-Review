"""
Pydantic Data Models for Multi-Modal Evidence Review Platform.

Strict validation ensures outputs always conform to allowed values.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from config import (
    CLAIM_OBJECTS,
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS_BY_TYPE,
    RISK_FLAGS,
    SEVERITIES,
)

logger = logging.getLogger(__name__)


# ─── Input Models ────────────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """One row from claims.csv — input only."""
    user_id: str
    image_paths: str          # semicolon-separated
    user_claim: str           # chat transcript
    claim_object: str         # car | laptop | package

    @property
    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_ids(self) -> list[str]:
        """Extract image IDs (filename without extension)."""
        from pathlib import Path
        return [Path(p).stem for p in self.image_path_list]

    @field_validator("claim_object")
    @classmethod
    def validate_claim_object(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CLAIM_OBJECTS:
            raise ValueError(f"Invalid claim_object: {v}")
        return v


class UserHistory(BaseModel):
    """One row from user_history.csv."""
    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = "none"
    history_summary: str = ""

    @property
    def history_flag_list(self) -> list[str]:
        if self.history_flags.strip().lower() == "none":
            return []
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]

    @property
    def rejection_ratio(self) -> float:
        if self.past_claim_count == 0:
            return 0.0
        return self.rejected_claim / self.past_claim_count

    @property
    def is_high_frequency(self) -> bool:
        return self.last_90_days_claim_count >= 4

    @property
    def has_risk(self) -> bool:
        return "user_history_risk" in self.history_flag_list

    @property
    def needs_manual_review(self) -> bool:
        return "manual_review_required" in self.history_flag_list


class EvidenceRequirement(BaseModel):
    """One row from evidence_requirements.csv."""
    requirement_id: str
    claim_object: str         # car | laptop | package | all
    applies_to: str           # issue family
    minimum_image_evidence: str


# ─── Intermediate Models ─────────────────────────────────────────────────────

class ClaimExtraction(BaseModel):
    """Output of Engine 1: what the user is actually claiming."""
    claimed_issue_type: str = "unknown"
    claimed_object_part: str = "unknown"
    claimed_severity_hint: str = "unknown"  # user's words, NOT final severity
    claim_summary: str = ""
    has_prompt_injection: bool = False
    prompt_injection_detail: str = ""
    is_multi_part: bool = False
    secondary_parts: list[str] = Field(default_factory=list)


class ImageAnalysis(BaseModel):
    """Output of Engine 2 (per-image): what the VLM sees in ONE image."""
    image_id: str
    image_path: str

    # What's visible
    visible_object_type: str = "unknown"     # car, laptop, package, other
    visible_object_part: str = "unknown"     # specific part
    visible_parts_list: list[str] = Field(default_factory=list)  # ALL visible parts
    visible_issue_type: str = "none"         # damage type visible
    visible_severity: str = "unknown"        # from visual evidence
    vehicle_color: str = ""                  # for car identity matching
    vehicle_type: str = ""                   # sedan, SUV, truck, etc.

    # YOLO priors (deterministic object detection)
    yolo_object_type: str = ""               # car/laptop/package/other/unknown
    yolo_confidence: float = 0.0

    # Quality & trust
    is_blurry: bool = False
    is_low_light: bool = False
    is_cropped: bool = False
    has_wrong_angle: bool = False
    has_watermark: bool = False
    watermark_text: str = ""
    has_text_instruction: bool = False
    text_instruction_content: str = ""
    is_usable: bool = True

    # EXIF / metadata
    has_exif: bool = False
    is_edited: bool = False

    # Relevance to claim
    shows_claimed_part: bool = False
    shows_claimed_damage: bool = False
    damage_description: str = ""

    # Raw confidence
    confidence: float = 0.5


class EvidenceSufficiency(BaseModel):
    """Output of Engine 3: are the images enough?"""
    evidence_standard_met: bool = False
    evidence_standard_met_reason: str = ""
    matched_requirements: list[str] = Field(default_factory=list)


class FraudSignals(BaseModel):
    """Output of Engine 5: fraud detection results."""
    risk_flags: list[str] = Field(default_factory=list)
    has_prompt_injection_in_text: bool = False
    has_prompt_injection_in_image: bool = False
    has_wrong_object: bool = False
    has_wrong_object_part: bool = False
    has_claim_mismatch: bool = False
    has_non_original_image: bool = False
    has_vehicle_identity_issue: bool = False
    vehicle_colors_found: list[str] = Field(default_factory=list)
    damage_not_visible: bool = False
    fraud_summary: str = ""


# ─── Output Model ────────────────────────────────────────────────────────────

class ClaimOutput(BaseModel):
    """Final output row for output.csv — strictly validated."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str        # "true" or "false"
    evidence_standard_met_reason: str
    risk_flags: str                   # semicolon-separated or "none"
    issue_type: str
    object_part: str
    claim_status: str                 # supported | contradicted | not_enough_information
    claim_status_justification: str
    supporting_image_ids: str         # semicolon-separated or "none"
    valid_image: str                  # "true" or "false"
    severity: str                     # none | low | medium | high | unknown

    # Confidence tracking (for ensemble/metadata, not in CSV output)
    confidence_issue_type: float = 1.0
    confidence_object_part: float = 1.0
    confidence_claim_status: float = 1.0
    confidence_severity: float = 1.0
    confidence_avg: float = 1.0

    @field_validator("claim_status")
    @classmethod
    def validate_claim_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CLAIM_STATUSES:
            logger.warning(f"Normalizing invalid claim_status '{v}' → 'not_enough_information'")
            return "not_enough_information"
        return v

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ISSUE_TYPES:
            logger.warning(f"Normalizing invalid issue_type '{v}' → 'unknown'")
            return "unknown"
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SEVERITIES:
            logger.warning(f"Normalizing invalid severity '{v}' → 'unknown'")
            return "unknown"
        return v

    @field_validator("evidence_standard_met", "valid_image")
    @classmethod
    def validate_boolean_str(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("true", "false"):
            return "false"
        return v

    def validate_object_part(self, claim_object: str) -> str:
        """Validate object_part against claim_object's allowed values."""
        allowed = OBJECT_PARTS_BY_TYPE.get(claim_object, set())
        if self.object_part not in allowed:
            logger.warning(
                f"Normalizing invalid object_part '{self.object_part}' "
                f"for {claim_object} → 'unknown'"
            )
            return "unknown"
        return self.object_part

    def normalize_risk_flags(self) -> str:
        """Ensure all risk flags are valid, deduplicate, sort."""
        if self.risk_flags.strip().lower() == "none":
            return "none"
        flags = [f.strip() for f in self.risk_flags.split(";") if f.strip()]
        valid_flags = []
        for f in flags:
            if f in RISK_FLAGS and f != "none":
                valid_flags.append(f)
            else:
                logger.warning(f"Dropping invalid risk_flag: '{f}'")
        if not valid_flags:
            return "none"
        return ";".join(sorted(set(valid_flags)))

    def to_csv_row(self) -> dict:
        """Return dict with all fields validated and normalized for CSV."""
        self.object_part = self.validate_object_part(self.claim_object)
        self.risk_flags = self.normalize_risk_flags()
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": self.evidence_standard_met,
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": self.valid_image,
            "severity": self.severity,
        }
