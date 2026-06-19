"""
Clear failed cache entries.

Scans .cache/ directory and deletes any cached response where the LLM
returned unknown/unknown (indicating the API call failed and we cached
a bad fallback result). This ensures re-runs will actually re-call the
API for previously failed images.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR


def clear_failed_cache():
    """Delete cache entries that contain failed/unknown results."""
    if not CACHE_DIR.exists():
        print("No cache directory found.")
        return

    cache_files = list(CACHE_DIR.glob("*.json"))
    print(f"Scanning {len(cache_files)} cache files...")

    deleted = 0
    kept = 0

    for f in cache_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            response = data.get("response", {})

            # Check if this is a failed vision response
            is_failed_vision = (
                response.get("visible_object_part") == "unknown"
                and response.get("visible_issue_type") == "unknown"
            )

            # Check if this is a failed claim extraction
            is_failed_claim = (
                response.get("claimed_object_part") == "unknown"
                and response.get("claimed_issue_type") == "unknown"
            )

            if is_failed_vision or is_failed_claim:
                os.remove(f)
                deleted += 1
            else:
                kept += 1

        except (json.JSONDecodeError, Exception) as e:
            # Corrupted cache file — delete it
            os.remove(f)
            deleted += 1

    print(f"Done: deleted {deleted} failed entries, kept {kept} good entries.")


if __name__ == "__main__":
    clear_failed_cache()
