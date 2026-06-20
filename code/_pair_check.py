import csv
from config import DATASET_DIR

gt = {}
with open(DATASET_DIR / "sample_ground_truth.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        gt[r["user_id"]] = r

preds = {}
with open(DATASET_DIR / "sample_output.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        preds[r["user_id"]] = r

print(f"{'Row':<8} {'user_id':<12} {'predicted':<18} {'expected':<18} {'match'}")
print("-"*70)
correct = 0
total = 0
for i, uid in enumerate(gt.keys(), 1):
    if uid in preds:
        p = preds[uid]["claim_status"]
        e = gt[uid]["claim_status"]
        match = "✓" if p == e else "✗"
        print(f"{i:<8} {uid:<12} {p:<18} {e:<18} {match}")
        if p == e:
            correct += 1
        total += 1
print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
