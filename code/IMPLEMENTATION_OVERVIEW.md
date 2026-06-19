# Implementation Overview — 100% Honest, Nothing Hidden

## TL;DR

**18 source files**, **~2,000 lines of code**, **2 LLM calls per claim** (1 text + N vision), **8 deterministic engines**, **multi-key rotation** for free-tier rate limits, processes 44 test claims → `output.csv`. Everything downstream of the Gemini calls is pure rule-based Python — no hidden LLM calls.

---

## Architecture — What Actually Happens Per Claim

```
┌─────────────────────────────────────────────────────────────────────┐
│  main.py (Orchestrator) — sequential loop over 44 claims           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. data_loader.load_claims()        → reads claims.csv             │
│  2. data_loader.load_user_history()  → reads user_history.csv       │
│  3. data_loader.load_evidence_requirements()                        │
│                                                                     │
│  FOR EACH CLAIM:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ ★ API CALL 1: claim_engine.extract_claim_with_llm()           │ │
│  │   → Sends conversation TEXT ONLY to Gemini Flash              │ │
│  │   → Gets back: claimed_part, claimed_issue, injection flag    │ │
│  │   → Also runs regex pre-scan for prompt injection patterns    │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │ ★ API CALL 2..N: vision_engine.analyze_single_image()         │ │
│  │   → Sends EACH image INDIVIDUALLY to Gemini Flash             │ │
│  │   → Gets back: visible_part, visible_issue, quality flags,    │ │
│  │     watermark, text_instruction, vehicle_color, confidence    │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │ DETERMINISTIC (no API calls):                                 │ │
│  │   E3: evidence_engine  → is evidence sufficient?              │ │
│  │   E4: quality_engine   → valid_image flag                     │ │
│  │   E5: fraud_engine     → 8 fraud signal checks                │ │
│  │   E6: risk_engine      → user history flag propagation        │ │
│  │   E7: decision_engine  → final claim_status + all fields      │ │
│  │   E8: explain_engine   → consistency polish                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  write_output_csv() → dataset/output.csv                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Key Rotation System

Since we use Gemini free tier (20 RPD per key), the system supports **multiple API keys** with automatic rotation:

```
┌─────────────────────────────────────────────────────────────────────┐
│  GEMINI_API_KEYS = key1, key2, key3  (comma-separated in .env)     │
│                                                                     │
│  Request → key1 → 429 RESOURCE_EXHAUSTED?                           │
│             ↓ YES                                                   │
│  Rotate  → key2 → 429 RESOURCE_EXHAUSTED?                           │
│             ↓ YES                                                   │
│  Rotate  → key3 → 429 RESOURCE_EXHAUSTED?                           │
│             ↓ YES                                                   │
│  All keys exhausted → graceful fallback to "unknown" for that call  │
└─────────────────────────────────────────────────────────────────────┘
```

This is implemented in `gemini_client.py` — on any 429/quota error, the client immediately swaps to the next key without sleeping. With **N keys**, you get **N × 20 = N×20 RPD**, enough for the full pipeline.

Additionally, all successful API responses are **cached to disk** (`.cache/` directory, SHA-256 keyed JSON files). Re-runs skip already-processed claims entirely, meaning you never waste API calls on claims you've already analyzed.

---

## Supported API Providers

The `.env` file supports keys for multiple providers (future extensibility):

| Provider | Env Variable | Current Usage |
|----------|-------------|---------------|
| **Gemini** | `GEMINI_API_KEYS` | ✅ Active — primary VLM (gemini-2.5-flash) |
| **Groq** | `GROQ_API_KEY` | 🔮 Ready — config loaded, client not yet wired |
| **OpenRouter** | `OPENROUTER_API_KEY` | 🔮 Ready — config loaded, client not yet wired |
| **NVIDIA** | `NVIDIA_API_KEY` | 🔮 Ready — config loaded, client not yet wired |

---

## File-by-File Breakdown

### Infrastructure (4 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `config.py` | 158 | All constants, paths, enums, .env loader, multi-key parsing | ✅ Solid. Zero-dependency .env parsing. All enum values from problem_statement.md. Supports `GEMINI_API_KEYS` (comma-separated) with fallback to `GEMINI_API_KEY`. |
| `models.py` | 266 | Pydantic models for every pipeline stage | ✅ Good validation. `to_csv_row()` normalizes invalid values. One concern: `validate_object_part` mutates `self.object_part` on a Pydantic model — works because model_config isn't frozen, but not idiomatic. |
| `data_loader.py` | 162 | CSV I/O, image base64 encoding | ✅ Handles BOM (`utf-8-sig`), Windows path separators. Writes with `QUOTE_ALL` for safety. |
| `.env` | 11 | API keys (Gemini multi-key, Groq, OpenRouter, NVIDIA) | ✅ Auto-loaded by config.py. `.gitignore`'d — never committed. |

### LLM Layer (4 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `gemini_client.py` | 261 | google.genai SDK wrapper, **multi-key rotation**, retries, caching, token tracking | ✅ Detects 429/quota errors → rotates to next key instantly. `Part.from_bytes` for images. `response_mime_type="application/json"` forces structured output. **Weakness**: cost estimate uses hardcoded pricing — may drift. |
| `prompts.py` | 166 | All 3 prompt templates | ⚠️ The core of accuracy. Prompts explicitly list allowed enum values, instruct to IGNORE text in images, and demand JSON-only output. **Risk**: prompt engineering is inherently fragile — VLM may still hallucinate on edge cases. |
| `cache.py` | 74 | SHA-256 keyed file-based JSON cache | ✅ Survives restarts. Avoids re-running API calls on same inputs. 63 cached responses from our first run. |
| `__init__.py` | 1 | Package marker | ✅ |

### 8 Engines (8 files)

| Engine | File | Lines | What It Does | Honest Assessment |
|--------|------|-------|-------------|-------------------|
| **E1** Claim | `claim_engine.py` | 166 | Extract claim from conversation via LLM + regex pre-scan | ✅ Fuzzy matching for ~40 aliases (bumper_front→front_bumper, display→screen, etc.). **Weakness**: fuzzy matching is hand-coded — may miss unusual aliases the LLM returns. |
| **E2** Vision | `vision_engine.py` | 170 | Per-image VLM analysis | ✅ Independent analysis per image — the key design win. Vehicle color extraction for identity matching. Supporting image selection with 4-tier priority. **Weakness**: relies on VLM accuracy for part identification. |
| **E3** Evidence | `evidence_engine.py` | 235 | Deterministic evidence sufficiency check | ⚠️ Maps issue types → requirement families via hardcoded dicts. This is fragile if new issue types are added. Vehicle identity check is basic (color string comparison). |
| **E4** Quality | `quality_engine.py` | 80 | Aggregate image quality → valid_image | ✅ Simple and correct: all watermarked = invalid, mixed blur = valid with flag. |
| **E5** Fraud | `fraud_engine.py` | 246 | 8 fraud signal detectors | ⚠️ The most complex deterministic engine. Checks: wrong_object, wrong_object_part, claim_mismatch, prompt_injection (text), text_instruction (image), non_original_image, vehicle_identity, damage_not_visible. **Weakness**: severity exaggeration check only catches simple cases. Vehicle color comparison is naive string matching — "dark blue" ≠ "blue". |
| **E6** Risk | `risk_engine.py` | 82 | User history propagation | ✅ Always propagates history flags. Auto-computes additional risk from rejection_ratio≥0.4 and claim_frequency. Simplest engine, matches sample labels well. |
| **E7** Decision | `decision_engine.py` | 478 | Aggregates all signals → final output | ⚠️ The most critical engine. 8-rule decision tree. object_part = VISIBLE (not claimed). Edge cases from sample labels handled (case_006, case_019, case_002). **Weakness**: hand-crafted decision tree — unexpected VLM values may produce wrong status. |
| **E8** Explain | `explain_engine.py` | 80 | Consistency checks + polish | ✅ Catches issue_type=none+status=supported inconsistency (flips to contradicted). Truncates long justifications. |

### Pipeline & Evaluation (3 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `main.py` | 219 | Full orchestrator with CLI | ✅ Error recovery per claim (returns safe fallback on exception). Rate limiting (0.3s between claims). Logs everything to run.log. |
| `evaluation/metrics.py` | 176 | Exact match, F1, confusion matrix, Jaccard | ✅ Standard metrics. Risk flags use micro/macro F1 since they're multi-label. |
| `evaluation/main.py` | 122 | Eval pipeline + operational report | ✅ Runs pipeline on sample_claims.csv, compares to ground truth, generates markdown report. |

### Utility & Config Files

| File | Purpose | Notes |
|------|---------|-------|
| `validate.py` | 14-test dry-run validation | All pass. Tests fuzzy matching, injection detection, user risk, quality, fraud, decision logic, CSV format. |
| `check_labels.py` | Prints sample ground truth labels | Dev utility. |
| `check_output.py` | Prints output.csv statistics | Dev utility. |
| `requirements.txt` | `google-genai>=2.9.0`, `pydantic>=2.0.0` | Only 2 deps. |
| `.gitignore` | Ignores .env, cache, logs, outputs, __pycache__ | Standard — API keys are never committed. |

---

## What's ACTUALLY Good (Not Hype)

### 1. Two-Call Design
This is the single best architectural decision. Each image is analyzed independently, which means:
- Blurry img_1 doesn't pollute clear img_2's analysis
- Vehicle identity can be cross-checked (color comparison across images)
- Text instructions in one image (case_020 sticky note) don't influence other image analysis
- Supporting image selection can pick the best one

### 2. Multi-Key Rotation
Automatic rotation across multiple free-tier Gemini API keys. When key1 hits 20 RPD, instantly switches to key2 without any delay. With 3 keys = 60 RPD, with 4 keys = 80 RPD, etc. Combined with disk caching, re-runs cost zero API calls for already-processed claims.

### 3. Enum Enforcement
Every output field is validated against allowed values. Invalid LLM outputs get normalized to "unknown". This prevents CSV schema violations.

### 4. User History Always Propagates
Even on `supported` claims, user risk flags appear in output. This matches sample labels exactly (case_017: supported but with `user_history_risk;manual_review_required`).

### 5. File-Based Cache
API responses cached to `.cache/` directory. Re-runs skip API calls for already-processed claims. Saves money during development. 63 cached responses from first run.

### 6. Deterministic Post-Processing
Engines 3–8 are pure Python logic. No randomness, no LLM calls. Same inputs → same outputs every time. This makes debugging easy.

---

## What's Honestly WEAK or RISKY

### 1. VLM Accuracy is the Bottleneck
> **Reality**: The entire system's accuracy depends on Gemini Flash correctly identifying: visible object type, visible part, visible damage type, damage severity, watermarks, text instructions, and vehicle color. If Gemini gets any of these wrong, the deterministic engines faithfully propagate the wrong signal.

**Specific risks:**
- "front_bumper" vs "hood" — VLM may confuse similar car parts
- Severity ("low" vs "medium") — subjective, VLM may disagree with ground truth
- Subtle watermarks (e.g., "Veeepik" semi-transparent) — VLM may miss them
- Vehicle color under different lighting — "dark blue" vs "black"

### 2. Prompt Engineering is Fragile
The prompts tell the VLM to return specific JSON with specific enum values. If Gemini's behavior changes across versions, or if it returns unexpected formats, the pipeline could break. The `response_mime_type="application/json"` helps but isn't bulletproof.

### 3. No Parallel Processing
Claims are processed sequentially. 44 claims × ~3 API calls each × ~2s per call = ~4-5 minutes. For a hackathon this is fine; for production it's slow.

### 4. Vehicle Identity Matching is Naive
Currently just compares vehicle color strings across images. Real vehicle identity matching would need:
- Make/model detection
- License plate comparison
- Damage location consistency

### 5. Evidence Engine Hardcoded Mappings
The requirement-to-issue family mappings are hardcoded dicts. If new evidence requirements are added, the code must be manually updated.

### 6. No Unit Tests
The `validate.py` script does integration-level dry-run testing, but there are no proper unit tests with mocked LLM responses. In a production system, you'd want tests that verify each engine independently with known inputs/outputs.

### 7. Free-Tier Rate Limits
With Gemini free tier, each key allows only 20 requests per day (RPD). The pipeline needs ~126 API calls for 44 claims. With 3 keys that's 60 RPD — not enough for a single-shot run. **Solution**: multi-key rotation + disk caching means you can spread across multiple runs or add more keys.

---

## What I Did NOT Implement (Claimed vs Reality)

| Claimed | Reality |
|---------|---------|
| "10 engines" in plan | **8 engines actually built**. The "Evaluation Framework" (Engine 9) and "Operational Cost Analysis" (Engine 10) are the eval/ directory and token tracker — they exist but aren't separate "engines" per se. |
| Vehicle identity VLM cross-check | **NOT a separate VLM call**. Vehicle identity is checked deterministically by comparing vehicle_color strings from per-image VLM responses. The `VEHICLE_IDENTITY_PROMPT` template exists in prompts.py but is **not called** in the pipeline — it was planned as a 3rd call for suspicious cases but not wired in. |
| Parallel/batch processing | **Sequential only**. The BATCH_SIZE=5 config exists but is unused. |
| Multi-part claim handling | **Partially implemented**. The LLM extracts `is_multi_part` and `secondary_parts`, but the decision engine only uses the primary claimed part. Secondary parts are extracted but not independently evaluated. |
| Multilingual support | **Delegated to Gemini**. The prompt says "handle multilingual" but there's no explicit translation step. Gemini Flash handles Hindi/Spanish reasonably well, but it's not tested. |
| Groq/OpenRouter/NVIDIA integration | **Keys loaded but clients not wired**. Config reads all 4 provider keys from .env, but only Gemini client is implemented. The other providers are ready for future extension. |

---

## Data Flow Trace (Concrete Example)

**Case 001** (user_001, car, rear_bumper dent):

```
1. data_loader reads: user_id="user_001", image_paths="images/test/case_001/img_1.jpg",
   user_claim="[long conversation about rear bumper dent]", claim_object="car"

2. E1 (claim_engine):
   → Regex pre-scan: no injection patterns found
   → LLM Call 1: sends conversation text → gets {claimed_part: "rear_bumper", claimed_issue: "dent"}
   → Fuzzy match: "rear_bumper" ∈ CAR_OBJECT_PARTS ✓

3. E2 (vision_engine):
   → Loads img_1.jpg as base64 (52KB → ~70K base64 chars)
   → VLM Call 2: sends image + prompt → gets {visible_part: "rear_bumper",
     visible_issue: "dent", severity: "medium", is_blurry: false, is_usable: true}

4. E3 (evidence_engine):
   → claimed part "rear_bumper" visible in img_1 ✓
   → evidence_standard_met = true

5. E4 (quality_engine):
   → img_1: not blurry, not watermarked, usable → valid_image = true

6. E5 (fraud_engine):
   → Object matches ✓, part matches ✓, issue matches ✓
   → No watermarks, no text instructions → risk_flags = []

7. E6 (risk_engine):
   → user_001: past_claim_count=2, rejected=0, flags="none" → no risk flags

8. E7 (decision_engine):
   → visible_part == claimed_part ✓, visible_issue == claimed_issue ✓
   → Decision: "supported"
   → severity = "medium" (from VLM)
   → supporting_image_ids = "img_1"

9. E8 (explain_engine):
   → No inconsistencies found → pass through

OUTPUT: status=supported, issue=dent, part=rear_bumper, severity=medium,
        evidence=true, valid=true, support=img_1, risk=none
```

---

## Actual Run Statistics

From the completed pipeline run:

| Metric | Value |
|--------|-------|
| Total claims processed | 44 |
| Successful API calls | 41 |
| Cached (skipped) calls | 22 |
| Failed API calls | 4 |
| Total input tokens | 30,960 |
| Total output tokens | 6,966 |
| Estimated cost | $0.0088 |
| Pipeline runtime | 682.1 seconds (~11.4 min) |
| Cache files on disk | 63 |
| Output status: supported | 6 |
| Output status: contradicted | 12 |
| Output status: not_enough_information | 26 |

> **Note**: 20 claims have `unknown/unknown` results due to all 3 free-tier keys exhausting their 20 RPD quota during the run. Adding more keys or re-running (cached claims are free) will complete these.

---

## API Call Count

Per claim:
- **1 text-only LLM call** (claim extraction)
- **N VLM calls** (1 per image, typically 1-2 images per claim)

For 44 test claims with ~85 total images:
- **44 text calls + ~85 vision calls = ~129 total API calls**
- With 3 free-tier keys (60 RPD total): requires ~2-3 runs with caching
- With 7+ keys (140+ RPD): single-shot complete run

---

## Summary: What You're Submitting

A system that:
1. **Reads** claims.csv (44 claims, ~85 images)
2. **Calls Gemini 2.5 Flash** ~129 times (text + vision) with **automatic key rotation**
3. **Cross-references** visual evidence against claimed damage using 6 deterministic engines
4. **Outputs** a 14-column output.csv with validated enum values
5. **Handles edge cases**: prompt injection, stock watermarks, wrong objects, vehicle identity, blurry images, multi-image selection
6. **Caches** all API responses to disk — re-runs are free
7. **Costs** ~$0.01 per run (free tier)

What it does NOT do:
- No fine-tuned model
- No parallel processing
- No vehicle make/model detection
- No actual structural similarity matching
- No confidence calibration against ground truth
- Multi-part claims extracted but not independently evaluated
- Groq/OpenRouter/NVIDIA keys loaded but not yet wired as alternative providers
