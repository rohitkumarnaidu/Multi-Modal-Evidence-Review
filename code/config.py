"""
Configuration & Constants for Multi-Modal Evidence Review Platform.

All allowed values, paths, model config, rate limits, and operational
parameters are centralized here. No magic strings elsewhere.
"""

import os
from pathlib import Path
from typing import Final

# ─── Load .env file ──────────────────────────────────────────────────────────
# Auto-load .env from code/ directory (no extra dependency needed)
_CODE_DIR_EARLY = Path(__file__).resolve().parent
_env_file = _CODE_DIR_EARLY / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip()
                if not os.environ.get(_key):  # Don't override existing env vars
                    os.environ[_key] = _val

# ─── Paths ────────────────────────────────────────────────────────────────────
# Resolve relative to the repo root (one level up from code/)
CODE_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = CODE_DIR.parent
DATASET_DIR: Final = REPO_ROOT / "dataset"
IMAGES_DIR: Final = DATASET_DIR / "images"

CLAIMS_CSV: Final = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV: Final = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV: Final = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV: Final = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV: Final = DATASET_DIR / "output.csv"

CACHE_DIR: Final = CODE_DIR / ".cache"

# ─── API Configuration ───────────────────────────────────────────────────────
# Support multiple keys for free-tier rate limit rotation
_gemini_keys_raw = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_API_KEYS: list[str] = [k.strip() for k in _gemini_keys_raw.split(",") if k.strip()]

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")

GEMINI_MODEL: Final = "gemini-2.5-flash"
GEMINI_TEMPERATURE: Final = 0.0
GEMINI_MAX_OUTPUT_TOKENS: Final = 4096

# Rate limiting
MAX_RPM: Final = 10          # Requests per minute (conservative)
MAX_TPM: Final = 1_000_000   # Tokens per minute
RETRY_MAX_ATTEMPTS: Final = 5
RETRY_BASE_DELAY: Final = 2.0  # seconds
RETRY_MAX_DELAY: Final = 60.0  # seconds

# Batching
BATCH_SIZE: Final = 5         # Process N claims concurrently
BATCH_DELAY: Final = 1.0      # seconds between batches

# ─── Allowed Values (from problem_statement.md) ──────────────────────────────

CLAIM_STATUSES: Final = frozenset({"supported", "contradicted", "not_enough_information"})

ISSUE_TYPES: Final = frozenset({
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
})

CAR_OBJECT_PARTS: Final = frozenset({
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
})

LAPTOP_OBJECT_PARTS: Final = frozenset({
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
})

PACKAGE_OBJECT_PARTS: Final = frozenset({
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
})

OBJECT_PARTS_BY_TYPE: Final = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

RISK_FLAGS: Final = frozenset({
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
})

SEVERITIES: Final = frozenset({"none", "low", "medium", "high", "unknown"})

CLAIM_OBJECTS: Final = frozenset({"car", "laptop", "package"})

# ─── Prompt Injection Patterns ───────────────────────────────────────────────
# Regex patterns to detect prompt injection in conversation text
PROMPT_INJECTION_KEYWORDS: Final = [
    r"approve.*claim",
    r"skip.*review",
    r"ignore.*previous",
    r"ignore.*instruction",
    r"mark.*supported",
    r"mark.*approved",
    r"accept.*immediately",
    r"follow.*note",
    r"follow.*instruction",
    r"approve.*immediately",
]

# Stock image watermark patterns
STOCK_IMAGE_MARKERS: Final = [
    "veeepik", "veepik", "shutterstock", "getty", "istockphoto",
    "alamy", "dreamstime", "adobe stock", "123rf", "depositphotos",
    "stock photo", "watermark",
]

# ─── Severity Mapping Rules ─────────────────────────────────────────────────
# Visual evidence-based severity, NOT user's claimed severity
SEVERITY_RULES: Final = {
    # (issue_type, visual_extent) -> severity
    "none": "none",
    "unknown": "unknown",
    "scratch_minor": "low",
    "scratch_deep": "medium",
    "dent_small": "low",
    "dent_medium": "medium",
    "dent_large": "high",
    "crack_hairline": "low",
    "crack_spreading": "medium",
    "crack_shattered": "high",
    "glass_shatter": "high",
    "broken_part": "medium",
    "missing_part": "medium",
    "torn_packaging": "medium",
    "crushed_packaging_minor": "low",
    "crushed_packaging_major": "medium",
    "water_damage": "medium",
    "stain": "medium",
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: Final = os.environ.get("LOG_LEVEL", "INFO")
METRICS_LOG: Final = CODE_DIR / ".metrics.json"
