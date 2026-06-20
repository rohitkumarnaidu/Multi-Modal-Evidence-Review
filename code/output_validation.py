"""Reusable CSV schema and evidence-consistency checks."""

from __future__ import annotations

import csv
from pathlib import Path

from config import CLAIM_STATUSES, ISSUE_TYPES, OBJECT_PARTS_BY_TYPE, RISK_FLAGS, SEVERITIES


OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


def validate_output_rows(rows: list[dict], expected_rows: int | None = None) -> list[str]:
    """Return human-readable violations without mutating prediction rows."""
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(rows)}.")

    for index, row in enumerate(rows, start=1):
        prefix = f"Row {index} ({row.get('user_id', '?')})"
        claim_object = row.get("claim_object", "").strip().lower()
        issue = row.get("issue_type", "").strip().lower()
        part = row.get("object_part", "").strip().lower()
        status = row.get("claim_status", "").strip().lower()
        severity = row.get("severity", "").strip().lower()

        if claim_object not in OBJECT_PARTS_BY_TYPE:
            errors.append(f"{prefix}: invalid claim_object={claim_object!r}.")
        if row.get("evidence_standard_met", "").strip().lower() not in {"true", "false"}:
            errors.append(f"{prefix}: evidence_standard_met must be true/false.")
        if row.get("valid_image", "").strip().lower() not in {"true", "false"}:
            errors.append(f"{prefix}: valid_image must be true/false.")
        if issue not in ISSUE_TYPES:
            errors.append(f"{prefix}: invalid issue_type={issue!r}.")
        if severity not in SEVERITIES:
            errors.append(f"{prefix}: invalid severity={severity!r}.")
        if status not in CLAIM_STATUSES:
            errors.append(f"{prefix}: invalid claim_status={status!r}.")
        if claim_object in OBJECT_PARTS_BY_TYPE and part not in OBJECT_PARTS_BY_TYPE[claim_object]:
            errors.append(f"{prefix}: invalid object_part={part!r} for {claim_object}.")

        flags = _parse_flags(row.get("risk_flags", "none"))
        invalid_flags = flags - (set(RISK_FLAGS) - {"none"})
        if invalid_flags:
            errors.append(f"{prefix}: invalid risk flags={sorted(invalid_flags)}.")

        if status == "supported":
            if issue in {"none", "unknown"}:
                errors.append(f"{prefix}: supported claim has issue_type={issue}.")
            if part == "unknown":
                errors.append(f"{prefix}: supported claim has object_part=unknown.")
            if row.get("evidence_standard_met", "").strip().lower() != "true":
                errors.append(f"{prefix}: supported claim does not meet evidence standard.")
            if row.get("supporting_image_ids", "").strip().lower() in {"", "none"}:
                errors.append(f"{prefix}: supported claim has no supporting image.")

        if issue == "none" and severity != "none":
            errors.append(f"{prefix}: issue_type=none requires severity=none.")
        if issue == "unknown" and status != "contradicted" and severity != "unknown":
            errors.append(f"{prefix}: issue_type=unknown requires severity=unknown.")
        if (
            status == "not_enough_information"
            and row.get("evidence_standard_met", "").strip().lower() == "false"
            and row.get("supporting_image_ids", "").strip().lower() not in {"", "none"}
        ):
            errors.append(f"{prefix}: insufficient evidence should not cite supporting images.")

    return errors


def load_and_validate_output(path: Path, expected_rows: int | None = None) -> tuple[list[dict], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []
    errors = []
    if columns != OUTPUT_COLUMNS:
        errors.append(f"Output columns do not match required schema: {columns}")
    errors.extend(validate_output_rows(rows, expected_rows=expected_rows))
    return rows, errors


def _parse_flags(raw: str) -> set[str]:
    value = (raw or "").strip().lower()
    if value in {"", "none"}:
        return set()
    return {flag.strip() for flag in value.split(";") if flag.strip()}
