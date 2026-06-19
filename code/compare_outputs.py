"""Compare predicted vs ground truth per-case."""
import csv
import sys
sys.path.insert(0, ".")

with open("../dataset/sample_output_nvidia_v4.csv", "r", encoding="utf-8-sig") as f:
    preds = {r["user_id"]: r for r in csv.DictReader(f)}
with open("../dataset/sample_claims.csv", "r", encoding="utf-8-sig") as f:
    truth = {r["user_id"]: r for r in csv.DictReader(f)}

fields = ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met"]
exact = 0
total = 0
for uid in sorted(truth.keys()):
    p = preds[uid]
    t = truth[uid]
    total += 1
    mismatches = []
    for f in fields:
        if p[f] != t[f]:
            mismatches.append(f + ":" + p[f] + "!=" + t[f])
    if mismatches:
        print(uid + ": " + " | ".join(mismatches))
    else:
        print(uid + ": ALL OK")
        exact += 1

print("\n=== SUMMARY ===")
print("Exact 4-field match: " + str(exact) + "/" + str(total))
print("Claim status correct: " + str(sum(1 for u in truth if preds[u]["claim_status"] == truth[u]["claim_status"])) + "/" + str(total))
print("Issue type correct: " + str(sum(1 for u in truth if preds[u]["issue_type"] == truth[u]["issue_type"])) + "/" + str(total))
print("Object part correct: " + str(sum(1 for u in truth if preds[u]["object_part"] == truth[u]["object_part"])) + "/" + str(total))
print("Severity correct: " + str(sum(1 for u in truth if preds[u]["severity"] == truth[u]["severity"])) + "/" + str(total))
