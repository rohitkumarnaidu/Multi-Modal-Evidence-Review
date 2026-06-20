import csv, sys
from config import DATASET_DIR
path = DATASET_DIR / "sample_claims.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 1):
        print(f"Row {i}: {r['user_id']} case={r.get('case_id','?')}")
