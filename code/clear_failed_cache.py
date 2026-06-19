import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"

def main():
    if not CACHE_DIR.is_dir():
        print("No cache directory found.")
        return

    removed = 0
    for f in sorted(CACHE_DIR.iterdir()):
        if f.suffix != ".json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            f.unlink()
            removed += 1
            print(f"  Removed (corrupt): {f.name}")
            continue

        text = json.dumps(data).lower()
        if "unknown" in text and "unknown" in text:
            f.unlink()
            removed += 1
            print(f"  Removed (unknowns): {f.name}")

    print(f"\nDone. Removed {removed} cache entries.")


if __name__ == "__main__":
    main()
