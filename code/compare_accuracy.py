"""Compare sample output against ground truth."""
import csv
import os

dataset = r"c:\Hackathons\Hackerrank\Multi-Modal Evidence Review\hackerrank-orchestrate-june26\dataset"

with open(os.path.join(dataset, "sample_output.csv"), "r", encoding="utf-8-sig") as f:
    gt = {r["user_id"]: r for r in csv.DictReader(f)}

with open(os.path.join(dataset, "sample_output_nvidia_latest.csv"), "r", encoding="utf-8-sig") as f:
    pred = {r["user_id"]: r for r in csv.DictReader(f)}

fields = ["claim_status", "evidence_standard_met", "issue_type", "object_part", "severity", "valid_image"]

print("=" * 70)
print("ACCURACY COMPARISON: Latest Code vs Ground Truth")
print("=" * 70)

for field in fields:
    correct = 0
    total = 0
    mismatches = []
    for uid in sorted(gt.keys()):
        if uid in pred:
            total += 1
            g = gt[uid].get(field, "").strip().lower()
            p = pred[uid].get(field, "").strip().lower()
            if g == p:
                correct += 1
            else:
                mismatches.append((uid, g, p))
    pct = 100 * correct / total if total > 0 else 0
    status = "OK" if pct >= 70 else "NEEDS WORK"
    print(f"\n  {field}: {correct}/{total} ({pct:.0f}%) [{status}]")
    for uid, g, p in mismatches:
        co = gt[uid].get("claim_object", "?")
        print(f"    WRONG {uid} ({co}): expected={g} got={p}")

# Risk flags F1
print(f"\n  --- Risk Flags Analysis ---")
tp = fp = fn = 0
for uid in sorted(gt.keys()):
    if uid not in pred:
        continue
    gt_flags = set(f.strip() for f in gt[uid].get("risk_flags", "").split(";") if f.strip() and f.strip() != "none")
    pred_flags = set(f.strip() for f in pred[uid].get("risk_flags", "").split(";") if f.strip() and f.strip() != "none")
    tp += len(gt_flags & pred_flags)
    fp += len(pred_flags - gt_flags)
    fn += len(gt_flags - pred_flags)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  risk_flags: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

print(f"\n{'=' * 70}")
print(f"OVERALL: {sum(1 for f in fields if sum(1 for uid in gt if uid in pred and gt[uid].get(f,'').strip().lower() == pred[uid].get(f,'').strip().lower()) / max(1, sum(1 for uid in gt if uid in pred)) >= 0.7)}/{len(fields)} fields at 70%+")
print(f"{'=' * 70}")
