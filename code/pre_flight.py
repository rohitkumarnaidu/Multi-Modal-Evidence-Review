import json
import os
import shutil
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent
ROOT = CODE_DIR.parent
DATASET = ROOT / "dataset"

REQUIRED_FILES = [
    "main.py",
    "config.py",
    "models.py",
    "pipeline.py",
    "engines/__init__.py",
    "engines/claim_engine.py",
    "engines/vision_engine.py",
    "engines/evidence_engine.py",
    "engines/quality_engine.py",
    "engines/fraud_engine.py",
    "engines/risk_engine.py",
    "engines/decision_engine.py",
    "engines/explain_engine.py",
    "llm/__init__.py",
    "llm/cache.py",
    "llm/prompts.py",
    "llm/rate_limiter.py",
    "llm/multi_provider_client.py",
    "llm/gemini_client.py",
    "llm/openai_compat_client.py",
    "calibration/__init__.py",
    "calibration/issue_calibration.py",
    "calibration/severity_map.py",
    "calibration/claim_patterns.py",
]

REQUIRED_DATASET = [
    "claims.csv",
    "sample_claims.csv",
    "user_history.csv",
    "evidence_requirements.csv",
]

REQUIRED_ENV_KEYS = [
    ("GEMINI_API_KEY or GEMINI_API_KEYS", ["GEMINI_API_KEY", "GEMINI_API_KEYS"]),
    ("GROQ_API_KEY", ["GROQ_API_KEY"]),
    ("OPENROUTER_API_KEY", ["OPENROUTER_API_KEY"]),
    ("NVIDIA_API_KEY", ["NVIDIA_API_KEY"]),
]

CRITICAL_IMPORTS = [
    "google.genai",
    "openai",
    "groq",
    "pydantic",
]

CHECKMARK = "[OK]"
CROSS = "[FAIL]"


def check_files() -> bool:
    ok = True
    for f in REQUIRED_FILES:
        p = CODE_DIR / f
        if p.exists():
            print(f"  {CHECKMARK} {f}")
        else:
            print(f"  {CROSS} {f}  (MISSING)")
            ok = False
    for f in REQUIRED_DATASET:
        p = DATASET / f
        if p.exists():
            print(f"  {CHECKMARK} dataset/{f}")
        else:
            print(f"  {CROSS} dataset/{f}  (MISSING)")
            ok = False
    images_sample = DATASET / "images" / "sample"
    images_test = DATASET / "images" / "test"
    if images_sample.is_dir():
        print(f"  {CHECKMARK} dataset/images/sample/")
    else:
        print(f"  {CROSS} dataset/images/sample/  (MISSING)")
        ok = False
    if images_test.is_dir():
        print(f"  {CHECKMARK} dataset/images/test/")
    else:
        print(f"  {CROSS} dataset/images/test/  (MISSING)")
        ok = False
    return ok


def check_env() -> bool:
    ok = True
    dotenv = CODE_DIR / ".env"
    if dotenv.exists():
        print(f"  {CHECKMARK} .env file exists")
        with open(dotenv) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    else:
        print(f"  {CHECKMARK} No .env file (will use environment variables)")
    for label, keys in REQUIRED_ENV_KEYS:
        found = False
        for key in keys:
            val = os.environ.get(key, "")
            if val:
                masked = val[:8] + "..." if len(val) > 12 else "(too short)"
                print(f"  {CHECKMARK} {label}={masked}")
                found = True
                break
        if not found:
            print(f"  {CROSS} {label} not set")
            ok = False
    return ok


def check_imports() -> bool:
    ok = True
    for mod in CRITICAL_IMPORTS:
        try:
            __import__(mod)
            print(f"  {CHECKMARK} {mod}")
        except ImportError:
            print(f"  {CROSS} {mod}  (not installed)")
            ok = False
    return ok


def check_cache_integrity() -> bool:
    cache_dir = CODE_DIR / ".cache"
    if not cache_dir.is_dir():
        print(f"  {CHECKMARK} No cache directory (fresh start)")
        return True
    ok = True
    corrupted = 0
    valid = 0
    for f in cache_dir.iterdir():
        if f.suffix == ".json":
            try:
                with open(f) as fh:
                    json.load(fh)
                valid += 1
            except (json.JSONDecodeError, OSError):
                corrupted += 1
                f.unlink()
    print(f"  {CHECKMARK} Cache: {valid} valid, {corrupted} corrupted (cleaned)")
    return ok


def main():
    exit_code = 0
    print("=" * 56)
    print("  Multi-Modal Evidence Review — Pre-Flight Check")
    print("=" * 56)

    print("\n[1/4] File structure:")
    if not check_files():
        exit_code = 1

    print("\n[2/4] Environment / API keys:")
    if not check_env():
        exit_code = 1

    print("\n[3/4] Python dependencies:")
    if not check_imports():
        exit_code = 1

    print("\n[4/4] Cache integrity:")
    if not check_cache_integrity():
        exit_code = 1

    print("\n" + "=" * 56)
    if exit_code:
        print(f"  {CROSS} Some checks FAILED. Fix issues above before running.")
    else:
        print(f"  {CHECKMARK} All checks passed. Ready to run.")
    print("=" * 56)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
