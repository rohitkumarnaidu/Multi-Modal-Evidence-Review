"""
Ensemble Engine — Multi-Provider Consensus Voting.

Runs all available providers independently and uses majority voting
to select the final output for each claim field.

Confidence is calculated from agreement rate across providers.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from config import CODE_DIR, DATASET_DIR, OUTPUT_CSV
from data_loader import write_output_csv

logger = logging.getLogger(__name__)

# Fields that benefit from ensemble voting
VOTING_FIELDS = [
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
    "evidence_standard_met",
    "valid_image",
]


def run_ensemble(
    output_files: dict[str, Path],
    ensemble_output: Path = OUTPUT_CSV,
    sample_ground_truth: Optional[list[dict]] = None,
) -> list[dict]:
    """Run ensemble voting across multiple provider outputs.

    Args:
        output_files: Dict mapping provider name to its output CSV path.
        ensemble_output: Path for the ensemble output CSV.
        sample_ground_truth: Optional ground truth for tracking per-provider accuracy.

    Returns:
        List of ensemble rows (dicts).
    """
    # Load all provider outputs
    provider_rows: dict[str, list[dict]] = {}
    for provider, path in output_files.items():
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
            provider_rows[provider] = rows
            logger.info(f"Loaded {len(rows)} rows from {provider}: {path}")

    if not provider_rows:
        raise ValueError("No provider outputs to ensemble")

    # Verify all providers have the same number of rows
    row_counts = {p: len(rows) for p, rows in provider_rows.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Row count mismatch: {row_counts}")

    num_rows = list(provider_rows.values())[0].__len__()
    providers = list(provider_rows.keys())

    # For each row, vote across providers
    ensemble_rows = []
    for i in range(num_rows):
        ensemble_row = dict(provider_rows[providers[0]][i])  # copy base fields

        votes = {}
        for field in VOTING_FIELDS:
            field_votes = Counter()
            for p in providers:
                val = provider_rows[p][i].get(field, "unknown").strip().lower()
                if val:
                    field_votes[val] += 1
            votes[field] = field_votes

            # Majority vote
            if field_votes:
                winner = field_votes.most_common(1)[0][0]
                ensemble_row[field] = winner
            else:
                ensemble_row[field] = "unknown"

        # Calculate confidence per field
        total_providers = len(providers)
        confidences = {}
        for field, field_votes in votes.items():
            if field_votes:
                top_votes = field_votes.most_common(1)[0][1]
                confidences[field] = round(top_votes / total_providers, 2)
            else:
                confidences[field] = 0.0

        # Add confidence as risk flag if low
        avg_confidence = sum(confidences.values()) / max(1, len(confidences))
        ensemble_row["risk_flags"] = _add_confidence_to_risk(
            ensemble_row.get("risk_flags", "none"),
            confidences,
            avg_confidence,
        )

        # Add confidence note to justification
        if avg_confidence < 0.75:
            justification = ensemble_row.get("claim_status_justification", "")
            confidence_note = f"[Confidence: {avg_confidence:.0%} across {total_providers} providers]"
            if confidence_note not in justification:
                if justification:
                    justification += f" {confidence_note}"
                else:
                    justification = confidence_note
                ensemble_row["claim_status_justification"] = justification

        # Store per-field confidence for tracking
        ensemble_row["_confidence_issue_type"] = confidences.get("issue_type", 0.0)
        ensemble_row["_confidence_object_part"] = confidences.get("object_part", 0.0)
        ensemble_row["_confidence_claim_status"] = confidences.get("claim_status", 0.0)
        ensemble_row["_confidence_severity"] = confidences.get("severity", 0.0)
        ensemble_row["_confidence_avg"] = avg_confidence

        ensemble_rows.append(ensemble_row)

    # Write ensemble output
    write_output_csv(ensemble_rows, ensemble_output)
    logger.info(f"Ensemble output written to {ensemble_output}")

    # Log confidence summary
    low_conf = [r for r in ensemble_rows if r.get("_confidence_avg", 1.0) < 0.75]
    if low_conf:
        logger.warning(
            f"{len(low_conf)}/{len(ensemble_rows)} claims have low ensemble confidence: "
            f"{[r['user_id'] for r in low_conf]}"
        )
    else:
        logger.info(f"All {len(ensemble_rows)} claims have high ensemble confidence")

    return ensemble_rows


def _add_confidence_to_risk(
    existing_flags: str,
    confidences: dict[str, float],
    avg_confidence: float,
) -> str:
    """Add confidence-related risk flags based on ensemble agreement."""
    flags = set()
    if existing_flags.strip().lower() != "none":
        flags = {f.strip() for f in existing_flags.split(";") if f.strip() and f.strip() != "none"}

    # Flag fields with low confidence
    for field, conf in confidences.items():
        if conf < 0.5:
            flags.add(f"low_confidence_{field}")

    if avg_confidence < 0.5:
        flags.add("manual_review_required")

    return ";".join(sorted(flags)) if flags else "none"


def analyze_ensemble_agreement(
    provider_rows: dict[str, list[dict]],
) -> dict:
    """Analyze how well providers agree with each other."""
    providers = list(provider_rows.keys())
    num_rows = len(list(provider_rows.values())[0])

    # Pairwise agreement per field
    pairwise = {}
    for field in VOTING_FIELDS:
        pairwise[field] = {}
        for p1 in providers:
            for p2 in providers:
                if p1 < p2:
                    agreements = 0
                    total = 0
                    for i in range(num_rows):
                        v1 = provider_rows[p1][i].get(field, "").strip().lower()
                        v2 = provider_rows[p2][i].get(field, "").strip().lower()
                        if v1 and v2:
                            total += 1
                            if v1 == v2:
                                agreements += 1
                    pairwise[field][f"{p1}_vs_{p2}"] = round(agreements / max(1, total), 4)

    # Per-provider majority agreement
    majority_agreement = {p: {} for p in providers}
    for field in VOTING_FIELDS:
        for p in providers:
            matches = 0
            total = 0
            for i in range(num_rows):
                my_val = provider_rows[p][i].get(field, "").strip().lower()
                all_vals = [provider_rows[p2][i].get(field, "").strip().lower() for p2 in providers]
                majority = Counter(all_vals).most_common(1)[0][0]
                if my_val and majority:
                    total += 1
                    if my_val == majority:
                        matches += 1
            majority_agreement[p][field] = round(matches / max(1, total), 4)

    return {
        "pairwise_agreement": pairwise,
        "majority_agreement": majority_agreement,
    }
