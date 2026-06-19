"""
Data Loader — CSV parsing, image loading, and base64 encoding.

Handles all I/O for the platform.
"""

from __future__ import annotations

import base64
import csv
import logging
from pathlib import Path
from typing import Optional

from config import (
    CLAIMS_CSV,
    DATASET_DIR,
    EVIDENCE_REQUIREMENTS_CSV,
    SAMPLE_CLAIMS_CSV,
    USER_HISTORY_CSV,
)
from models import ClaimInput, EvidenceRequirement, UserHistory

logger = logging.getLogger(__name__)


def load_claims(csv_path: Path | None = None) -> list[ClaimInput]:
    """Load claims from CSV file."""
    path = csv_path or CLAIMS_CSV
    claims = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                claim = ClaimInput(
                    user_id=row["user_id"].strip(),
                    image_paths=row["image_paths"].strip(),
                    user_claim=row["user_claim"].strip(),
                    claim_object=row["claim_object"].strip(),
                )
                claims.append(claim)
            except Exception as e:
                logger.error(f"Error parsing claim row: {e}, row={row}")
    logger.info(f"Loaded {len(claims)} claims from {path}")
    return claims


def load_sample_claims(csv_path: Path | None = None) -> list[dict]:
    """Load sample claims with labels (ground truth)."""
    path = csv_path or SAMPLE_CLAIMS_CSV
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip all values
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)
    logger.info(f"Loaded {len(rows)} sample claims from {path}")
    return rows


def load_user_history(csv_path: Path | None = None) -> dict[str, UserHistory]:
    """Load user history, indexed by user_id."""
    path = csv_path or USER_HISTORY_CSV
    history: dict[str, UserHistory] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                uh = UserHistory(
                    user_id=row["user_id"].strip(),
                    past_claim_count=int(row["past_claim_count"]),
                    accept_claim=int(row["accept_claim"]),
                    manual_review_claim=int(row["manual_review_claim"]),
                    rejected_claim=int(row["rejected_claim"]),
                    last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                    history_flags=row["history_flags"].strip(),
                    history_summary=row["history_summary"].strip(),
                )
                history[uh.user_id] = uh
            except Exception as e:
                logger.error(f"Error parsing user_history row: {e}, row={row}")
    logger.info(f"Loaded {len(history)} user history records from {path}")
    return history


def load_evidence_requirements(csv_path: Path | None = None) -> list[EvidenceRequirement]:
    """Load evidence requirements."""
    path = csv_path or EVIDENCE_REQUIREMENTS_CSV
    reqs = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                req = EvidenceRequirement(
                    requirement_id=row["requirement_id"].strip(),
                    claim_object=row["claim_object"].strip(),
                    applies_to=row["applies_to"].strip(),
                    minimum_image_evidence=row["minimum_image_evidence"].strip(),
                )
                reqs.append(req)
            except Exception as e:
                logger.error(f"Error parsing evidence requirement: {e}, row={row}")
    logger.info(f"Loaded {len(reqs)} evidence requirements from {path}")
    return reqs


def load_image_as_base64(image_path: str) -> Optional[str]:
    """Load an image file and return its base64-encoded string.
    
    image_path is relative to the dataset/ directory.
    """
    # Resolve the full path
    full_path = DATASET_DIR / image_path.replace("/", "\\")
    if not full_path.exists():
        # Try forward slashes
        full_path = DATASET_DIR / image_path
    if not full_path.exists():
        logger.error(f"Image not found: {full_path}")
        return None

    try:
        with open(full_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        logger.debug(f"Loaded image: {full_path} ({len(data)} bytes)")
        return encoded
    except Exception as e:
        logger.error(f"Error loading image {full_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def write_output_csv(rows: list[dict], output_path: Path) -> None:
    """Write output rows to CSV with exact required column order."""
    columns = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    logger.info(f"Wrote {len(rows)} rows to {output_path}")
