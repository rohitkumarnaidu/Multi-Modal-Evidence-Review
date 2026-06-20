"""Study the ground truth for hard cases."""
import csv
import os

dataset = r"c:\Hackathons\Hackerrank\Multi-Modal Evidence Review\hackerrank-orchestrate-june26\dataset"

with open(os.path.join(dataset, "sample_claims.csv"), "r", encoding="utf-8-sig") as f:
    claims = {r["user_id"]: r for r in csv.DictReader(f)}

with open(os.path.join(dataset, "sample_output.csv"), "r", encoding="utf-8-sig") as f:
    gt = {r["user_id"]: r for r in csv.DictReader(f)}

focus = ["user_002","user_003","user_004","user_005","user_007","user_008","user_032","user_033","user_034","user_018"]

for uid in focus:
    c = claims[uid]
    g = gt[uid]
    co = c["claim_object"]
    imgs = c["image_paths"]
    claim_text = c["user_claim"][-250:]
    
    print(f"=== {uid} ({co}) ===")
    print(f"  Images: {imgs}")
    print(f"  Claim (tail): ...{claim_text}")
    print(f"  GT: status={g['claim_status']}, part={g['object_part']}, issue={g['issue_type']}, sev={g['severity']}")
    print(f"  GT evidence_met: {g['evidence_standard_met']}")
    print(f"  GT risk: {g['risk_flags']}")
    just = g["claim_status_justification"][:300]
    print(f"  GT justification: {just}")
    print()
