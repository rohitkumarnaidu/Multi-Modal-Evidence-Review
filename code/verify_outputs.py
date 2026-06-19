"""Quick verification of all output files."""
import csv
import os

dataset_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
)
os.chdir(dataset_dir)

valid_status = {"supported", "contradicted", "not_enough_information"}
valid_severity = {"none", "low", "medium", "high", "unknown"}

files = [
    "output.csv",
    "output_nvidia.csv",
    "output_openrouter.csv",
    "output_groq.csv",
    "output_gemini.csv",
]

for fname in files:
    if not os.path.exists(fname):
        print(f"❌ {fname} — NOT FOUND")
        continue
    
    with open(fname, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
    
    cols = list(reader[0].keys()) if reader else []
    unknowns = len([r for r in reader if r.get("issue_type") == "unknown" and r.get("object_part") == "unknown"])
    
    statuses = {}
    for r in reader:
        s = r["claim_status"]
        statuses[s] = statuses.get(s, 0) + 1
    
    bad_status = [r["user_id"] for r in reader if r["claim_status"] not in valid_status]
    bad_sev = [r["user_id"] for r in reader if r.get("severity", "") not in valid_severity]
    
    ok = "✅" if not bad_status and not bad_sev and len(reader) == 44 else "⚠️"
    
    print(f"\n{ok} {fname}")
    print(f"   Rows: {len(reader)} | Cols: {len(cols)}")
    print(f"   Status: {statuses}")
    print(f"   Unknowns (issue+part both unknown): {unknowns}")
    if bad_status:
        print(f"   ❌ Invalid statuses: {bad_status}")
    if bad_sev:
        print(f"   ❌ Invalid severity: {bad_sev}")

# Column check
print("\n=== COLUMN CHECK (output.csv) ===")
with open("output.csv", "r", encoding="utf-8-sig") as f:
    reader = list(csv.DictReader(f))
    cols = list(reader[0].keys())

required = [
    "user_id", "claim_object", "claim_status",
    "issue_type", "object_part", "severity",
    "evidence_standard_met", "valid_image", "supporting_image_ids",
    "risk_flags", "claim_status_justification"
]

for c in required:
    found = "✅" if c in cols else "❌ MISSING"
    print(f"   {found} {c}")

print(f"\n   All columns ({len(cols)}): {cols}")

# Sample rows
print("\n=== SAMPLE OUTPUT (first 5) ===")
for r in reader[:5]:
    print(f"   {r['user_id']}: {r['claim_status']} | {r['object_part']}/{r['issue_type']} | sev={r['severity']}")
