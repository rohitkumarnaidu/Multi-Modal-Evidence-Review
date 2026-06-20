import csv
from config import DATASET_DIR, SAMPLE_CLAIMS_CSV

# Load predictions
with open(DATASET_DIR / "sample_output.csv", newline="", encoding="utf-8-sig") as f:
    preds = list(csv.DictReader(f))

# Load ground truth
with open(SAMPLE_CLAIMS_CSV, newline="", encoding="utf-8-sig") as f:
    gt = list(csv.DictReader(f))

print(f"Predictions: {len(preds)}, Ground truth: {len(gt)}")
print()

field = "claim_status"
for i, (p, g) in enumerate(zip(preds, gt)):
    pv = str(p.get(field, "")).strip().lower()
    gv = str(g.get(field, "")).strip().lower()
    match = "OK" if pv == gv else "XX"
    print(f"Row {i+1}: {p['user_id']:12s} pred={pv:20s} expected={gv:20s} {match}")
