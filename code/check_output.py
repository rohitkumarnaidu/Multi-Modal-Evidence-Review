"""Validate output.csv schema, invariants, and status distribution."""

from __future__ import annotations

from pathlib import Path

from config import CLAIMS_CSV, OUTPUT_CSV
from output_validation import load_and_validate_output


rows, errors = load_and_validate_output(
    OUTPUT_CSV,
    expected_rows=sum(1 for _ in open(CLAIMS_CSV, encoding="utf-8-sig")) - 1,
)
print(f"Total rows: {len(rows)}")
statuses = [r["claim_status"] for r in rows]
for s in ["supported", "contradicted", "not_enough_information"]:
    print(f"  {s}: {statuses.count(s)}")

failed = [r["user_id"] for r in rows if r.get("issue_type") == "unknown" and r.get("object_part") == "unknown"]
print(f"\nFully unknown (API failed): {len(failed)}")
if failed:
    print(f"  Users: {', '.join(failed)}")

print(f"\nConsistency violations: {len(errors)}")
for error in errors:
    print(f"  [FAIL] {error}")

if errors:
    raise SystemExit(1)
print("  [OK] Schema and evidence consistency checks passed")
