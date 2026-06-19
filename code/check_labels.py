"""Print all sample claim labels for analysis."""
import sys
sys.path.insert(0, ".")
from data_loader import load_sample_claims
sc = load_sample_claims()
for i, r in enumerate(sc):
    print(f"Case {i+1:02d} ({r['user_id']}): "
          f"status={r['claim_status']}, issue={r['issue_type']}, "
          f"part={r['object_part']}, severity={r['severity']}, "
          f"evidence={r['evidence_standard_met']}, valid={r['valid_image']}, "
          f"support={r['supporting_image_ids']}, risk={r['risk_flags']}")
