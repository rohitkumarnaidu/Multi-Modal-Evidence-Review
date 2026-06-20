"""Print sample prediction errors grouped by output field."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR


FIELDS = [
    "claim_status", "evidence_standard_met", "risk_flags", "issue_type",
    "object_part", "severity", "supporting_image_ids", "valid_image",
]


def main() -> None:
    with open(DATASET_DIR / "sample_claims.csv", encoding="utf-8-sig", newline="") as file:
        expected = list(csv.DictReader(file))
    with open(DATASET_DIR / "sample_output.csv", encoding="utf-8-sig", newline="") as file:
        predicted = list(csv.DictReader(file))

    grouped: dict[str, list[str]] = defaultdict(list)
    for index, (actual, prediction) in enumerate(zip(expected, predicted), start=1):
        for field in FIELDS:
            got = prediction.get(field, "").strip().lower()
            want = actual.get(field, "").strip().lower()
            if got != want:
                grouped[field].append(
                    f"row={index} user={actual['user_id']} predicted={got!r} expected={want!r}"
                )

    for field in FIELDS:
        errors = grouped[field]
        print(f"\n{field}: {len(errors)} errors")
        for error in errors:
            print(f"  {error}")


if __name__ == "__main__":
    main()
