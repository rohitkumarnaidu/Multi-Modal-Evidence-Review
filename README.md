# Multi-Modal Evidence Review

A system that verifies visual evidence for damage claims across **cars**, **laptops**, and **packages** using a 2-call design: text-only LLM + per-image VLM.

**Results**: 44/44 claims processed — 0 unknowns, 0 failures.

---

## Architecture (2-Call Design)

Each claim is processed by **two independent LLM calls**, then passed through deterministic engines:

1. **Call 1 (Text LLM)**: Extract `issue_type`, `object_part`, and `claim_summary` from the conversation transcript. No images sent.
2. **Call 2 (VLM per image)**: Each image is analyzed independently for part, damage, quality, and fraud signals.
3. **Post-processing**: Deterministic engines aggregate VLM outputs, calibrate biases, and make the final decision.

### 8 Engines

| # | Engine | File | Purpose |
|---|--------|------|---------|
| 1 | **Claim** | `engines/claim_engine.py` | Extract claim from conversation text (LLM call 1) |
| 2 | **Vision** | `engines/vision_engine.py` | Per-image VLM analysis (LLM call 2 per image) |
| 3 | **Evidence** | `engines/evidence_engine.py` | Check minimum image requirements |
| 4 | **Quality** | `engines/quality_engine.py` | Assess image usability (blurry, cropped, etc.) |
| 5 | **Fraud** | `engines/fraud_engine.py` | Detect manipulation, wrong object, claim mismatch |
| 6 | **Risk** | `engines/risk_engine.py` | Evaluate user history risk flags |
| 7 | **Decision** | `engines/decision_engine.py` | Aggregate all outputs → final verdict |
| 8 | **Explain** | `engines/explain_engine.py` | Polish justification text |

### Calibration Tools

| Tool | File | Purpose |
|------|------|---------|
| Severity Map | `calibration/severity_map.py` | Map (issue_type × extent) → severity level |
| Issue Calibration | `calibration/issue_calibration.py` | Correct VLM systematic biases (e.g., glass_shatter → crack) |
| Claim Patterns | `calibration/claim_patterns.py` | Detect multi-part claims, Hindi/regional language patterns |

---

## Multi-Provider Fallback Chain

The router tries providers in order, falling back on failure:

`NVIDIA` → `OpenRouter` → `Gemini` → `Groq`

- **NVIDIA**: `meta/llama-4-maverick-17b-128e-instruct` (primary)
- **OpenRouter**: `google/gemini-2.5-flash` (fallback 1)
- **Gemini**: `gemini-2.5-flash` (fallback 2)
- **Groq**: `meta-llama/llama-4-scout-17b-16e-instruct` (fallback 3)

Each provider has dedicated API key rotation and per-provider cache isolation.

---

## Ensemble Voting

All 4 providers can be run independently and their outputs combined via majority voting:

- **Voting fields**: `issue_type`, `object_part`, `claim_status`, `severity`, `evidence_standard_met`, `valid_image`
- **Confidence**: computed from agreement rate across providers
- **Risk flags**: auto-added for low-confidence fields
- **Analysis**: pairwise and majority-agreement reports saved to `ensemble_agreement.json`

Usage:
```bash
python run_ensemble.py        # Run all providers + ensemble
python run_ensemble.py --fast # Skip re-running cached providers
```

---

## Confidence Scoring

- **Default**: `1.0` (single-provider mode)
- **Risk reduction**: confidence decreases when risk flags are present:
  - Blurry/cropped/low-light: −10%
  - Wrong angle/damage not visible: −20%
  - Wrong object/part: −30%
  - Manipulation/non-original image: −40%
- **Ensemble mode**: confidence = agreement rate across providers

---

## Rate Limiting & Caching

- **Rate limiter**: 0.5s minimum interval between API calls
- **Retry**: Exponential backoff, max 10 attempts, max 60s delay
- **Cache**: File-based JSON cache (SHA-256 key), survives restarts
- **Per-provider isolation**: Separate cache dir per provider (`.cache_nvidia`, `.cache_gemini`, etc.)
- **Concurrency**: Optional `ThreadPoolExecutor` with `--parallel --workers N`

---

## Setup

```bash
pip install pydantic
```

Create `.env` in `code/`:
```
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
NVIDIA_API_KEY=your_key
```

---

## Run Instructions

```bash
# Full test run
python main.py

# Sample claims only
python main.py --mode sample

# Force a specific provider
python main.py --provider gemini
python main.py --provider nvidia

# Parallel processing
python main.py --parallel --workers 4

# Retry only failed claims
python main.py --retry-failed

# Ensemble (all 4 providers + majority vote)
python run_ensemble.py
```

### Evaluation

```bash
python evaluation/main.py
```

Generates `evaluation/evaluation_report.md` with:
- Per-field accuracy scores
- Confusion matrices
- Operational analysis (calls, tokens, cost, runtime)
- Strategy comparison (2-call vs mega-prompt)

---

## Output Schema

| Column | Description |
|--------|-------------|
| `evidence_standard_met` | Sufficient image evidence available |
| `evidence_standard_met_reason` | Short reason for evidence decision |
| `risk_flags` | Semicolon-separated flags or `none` |
| `issue_type` | Identified damage type |
| `object_part` | Relevant object part |
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | Concise explanation grounded in the image evidence |
| `supporting_image_ids` | Image IDs supporting the decision, or `none` |
| `valid_image` | Whether image set is usable for automated review |
| `severity` | `none` / `low` / `medium` / `high` / `unknown` |
