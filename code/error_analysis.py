"""Detailed error analysis: compare output vs ground truth."""
import csv
import os

dataset = r"c:\Hackathons\Hackerrank\Multi-Modal Evidence Review\hackerrank-orchestrate-june26\dataset"

with open(os.path.join(dataset, "sample_output.csv"), "r", encoding="utf-8-sig") as f:
    gt = {r["user_id"]: r for r in csv.DictReader(f)}

with open(os.path.join(dataset, "output_nvidia.csv"), "r", encoding="utf-8-sig") as f:
    pred = {r["user_id"]: r for r in csv.DictReader(f)}

with open(os.path.join(dataset, "sample_claims.csv"), "r", encoding="utf-8-sig") as f:
    sample_claims = {r["user_id"]: r for r in csv.DictReader(f)}

print(f"Ground truth users ({len(gt)}): {sorted(gt.keys())}")
print(f"Prediction users ({len(pred)}): {len(pred)} total")
print(f"Sample claim users ({len(sample_claims)}): {sorted(sample_claims.keys())}")
print()

# Show duplicate user_ids in output_nvidia.csv
with open(os.path.join(dataset, "output_nvidia.csv"), "r", encoding="utf-8-sig") as f:
    all_rows = list(csv.DictReader(f))
uid_counts = {}
for r in all_rows:
    uid = r["user_id"]
    uid_counts[uid] = uid_counts.get(uid, 0) + 1
dupes = {k: v for k, v in uid_counts.items() if v > 1}
print(f"Duplicate user_ids in output: {dupes}")
print()

# Missing from predictions
missing = [u for u in gt if u not in pred]
print(f"Missing from pred: {missing}")
print()

# GT columns
first_gt = list(gt.values())[0]
print(f"GT columns: {list(first_gt.keys())}")
print()

# Detailed comparison for each sample user
print("=" * 100)
print("DETAILED PER-CLAIM COMPARISON")
print("=" * 100)

for uid in sorted(gt.keys()):
    g = gt[uid]
    p = pred.get(uid, None)
    co = g.get("claim_object", "?")
    
    print(f"\n--- {uid} ({co}) ---")
    if p is None:
        print("  NOT IN PREDICTIONS!")
        continue
    
    fields = ["claim_status", "object_part", "issue_type", "severity", "evidence_standard_met"]
    for field in fields:
        gv = g.get(field, "").strip().lower()
        pv = p.get(field, "").strip().lower()
        match = "OK" if gv == pv else "WRONG"
        if match == "WRONG":
            print(f"  WRONG {field}: expected={gv}  got={pv}")
        else:
            print(f"  OK    {field}: {gv}")
    
    # Show risk flags too
    g_risk = g.get("risk_flags", "").strip()
    p_risk = p.get("risk_flags", "").strip()
    if g_risk != p_risk:
        print(f"  RISK  expected={g_risk}  got={p_risk}")

    # Show justification snippet
    p_just = p.get("claim_status_justification", "")[:150]
    print(f"  JUSTIFICATION: {p_just}...")
