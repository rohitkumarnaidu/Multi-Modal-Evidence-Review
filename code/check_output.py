"""Check output.csv stats."""
import csv, sys
rows = list(csv.DictReader(open(
    r"c:\Hackathons\Hackerrank\Multi-Modal Evidence Review\hackerrank-orchestrate-june26\dataset\output.csv",
    "r", encoding="utf-8-sig"
)))
print(f"Total rows: {len(rows)}")
statuses = [r["claim_status"] for r in rows]
for s in ["supported", "contradicted", "not_enough_information"]:
    print(f"  {s}: {statuses.count(s)}")

failed = [r["user_id"] for r in rows if r.get("issue_type") == "unknown" and r.get("object_part") == "unknown"]
print(f"\nFully unknown (API failed): {len(failed)}")
if failed:
    print(f"  Users: {', '.join(failed)}")
