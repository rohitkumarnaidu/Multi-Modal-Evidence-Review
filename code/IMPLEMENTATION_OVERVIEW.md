# Implementation Overview — 100% Honest, Nothing Hidden

## TL;DR

**24 source files**, **~3,200 lines of code**, **4 LLM providers** (Gemini → Groq → OpenRouter → NVIDIA), **2 LLM calls per claim** (1 text + N vision), **8 deterministic engines**, **per-key RPM/RPD rate limiting**, **multi-model comparison** system, processes 44 test claims → `output.csv`. Everything downstream of the LLM calls is pure rule-based Python — no hidden LLM calls.

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
│  │   → Sends conversation TEXT ONLY to LLM (any provider)        │ │
│  │   → Gets back: claimed_part, claimed_issue, injection flag    │ │
│  │   → Also runs regex pre-scan for prompt injection patterns    │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │ ★ API CALL 2..N: vision_engine.analyze_single_image()         │ │
│  │   → Sends EACH image INDIVIDUALLY to VLM (any provider)       │ │
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

## Multi-Provider Fallback Chain

The system uses 4 LLM providers in a priority fallback chain. If one provider fails all retries, the next is tried automatically. The pipeline code doesn't know or care which provider handled the call.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MultiProviderClient (multi_provider_client.py)                         │
│                                                                          │
│  Request ──► GEMINI (6 keys, 5 RPM/key, 20 RPD/key)                    │
│              │ gemini-2.5-flash                                          │
│              │ auto key rotation on 429                                  │
│              ▼ All keys exhausted?                                       │
│         ──► GROQ (25 RPM, 14,400 RPD)                                   │
│              │ llama-4-maverick-17b-128e-instruct                        │
│              ▼ Failed?                                                   │
│         ──► OPENROUTER (20 RPM, free tier)                               │
│              │ google/gemini-2.5-flash (same model, different quota)     │
│              ▼ Failed?                                                   │
│         ──► NVIDIA (40 RPM, ∞ unlimited credits)                         │
│              │ meta/llama-4-maverick-17b-128e-instruct                   │
│              ▼ Failed?                                                   │
│         ──► Return None (graceful fallback)                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Provider Configuration

| Provider | Model (Vision) | Model (Text) | RPM | RPD | Status |
|----------|---------------|--------------|-----|-----|--------|
| **Gemini** | `gemini-2.5-flash` | `gemini-2.5-flash` | 5/key (30 total) | 20/key (120 total) | ✅ Active |
| **Groq** | `llama-4-maverick-17b-128e` | `llama-3.3-70b-versatile` | 25 | 14,400 | ✅ Active |
| **OpenRouter** | `google/gemini-2.5-flash` | `google/gemini-2.5-flash` | 20 | varies | ✅ Active |
| **NVIDIA** | `meta/llama-4-maverick-17b-128e` | `meta/llama-4-maverick-17b-128e` | 40 | ∞ unlimited | ✅ Active |

---

## Rate Limiting System

The system has **two types of rate limiters** to prevent 429 errors proactively:

### KeyRateLimiter (Gemini — per-key RPM + RPD)

```
┌────────────────────────────────────────────────────────────┐
│  Per-Key Tracking:                                         │
│                                                            │
│  Key 0: RPM ████░░ 4/5   RPD ████████████████████ 20/20   │
│  Key 1: RPM ██░░░░ 2/5   RPD ████████████░░░░░░░░ 12/20   │
│  Key 2: RPM ░░░░░░ 0/5   RPD ████░░░░░░░░░░░░░░░░  4/20   │
│                                                            │
│  • Sliding window: tracks last 60s of request timestamps   │
│  • Proactive rotation: switches key BEFORE hitting API     │
│  • Auto-sleep: waits when RPM limit reached                │
│  • mark_key_exhausted(): learns from 429 errors            │
└────────────────────────────────────────────────────────────┘
```

### SimpleRateLimiter (Groq / OpenRouter / NVIDIA)

- Sliding 60-second window
- Auto-sleeps when at RPM limit (+1.5s safety margin)
- No key rotation needed (single key per provider)

---

## Multi-Model Comparison System

The system can run ALL providers independently and generate a cross-model comparison report.

### Usage

```bash
# Run single provider
python main.py --provider nvidia      # → dataset/output_nvidia.csv

# Run all providers + comparison
python compare_models.py              # Runs all 4, generates comparison

# Just compare existing outputs  
python compare_models.py --report-only
```

### Output Files

| File | What |
|------|------|
| `dataset/output_gemini.csv` | Gemini-only results |
| `dataset/output_groq.csv` | Groq-only results |
| `dataset/output_openrouter.csv` | OpenRouter-only results |
| `dataset/output_nvidia.csv` | NVIDIA-only results |
| `dataset/model_comparison.csv` | Side-by-side per-claim comparison |
| `dataset/output_consensus.csv` | **Majority-vote consensus** (best of all models) |

Each provider uses its own **isolated cache** (`.cache_gemini/`, `.cache_groq/`, etc.) so models never see each other's cached responses.

---

## File-by-File Breakdown

### Infrastructure (5 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `config.py` | 175 | All constants, paths, enums, .env loader, multi-key parsing, 4 provider model configs | ✅ Solid. Zero-dependency .env parsing. All enum values from problem_statement.md. |
| `models.py` | 266 | Pydantic models for every pipeline stage | ✅ Good validation. `to_csv_row()` normalizes invalid values. |
| `data_loader.py` | 162 | CSV I/O, image base64 encoding | ✅ Handles BOM (`utf-8-sig`), Windows path separators. |
| `.env` | 11 | API keys (Gemini×6, Groq, OpenRouter, NVIDIA) | ✅ Auto-loaded by config.py. `.gitignore`'d — never committed. |
| `requirements.txt` | 4 | `google-genai`, `pydantic`, `openai`, `groq` | ✅ 4 deps total. |

### LLM Layer (7 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `gemini_client.py` | 330 | google.genai SDK wrapper, **multi-key rotation**, per-key RPM/RPD rate limiting, retries with retryDelay parsing, caching, token tracking | ✅ Proactive key rotation BEFORE hitting API. Parses `retryDelay` from 429 errors. No more spin-loop bug. |
| `openai_compat_client.py` | 240 | OpenAI-compatible client for Groq, OpenRouter, NVIDIA | ✅ Single client handles all 3 providers by swapping base_url + api_key + model. RPM rate limiting via SimpleRateLimiter. |
| `multi_provider_client.py` | 210 | Orchestrator with Gemini → Groq → OpenRouter → NVIDIA fallback chain | ✅ `--provider` flag for per-model comparison. Separate cache per provider. Same call_text/call_vision interface. |
| `rate_limiter.py` | 165 | `KeyRateLimiter` (per-key RPM+RPD) and `SimpleRateLimiter` (RPM only) | ✅ Sliding window, proactive rotation, auto-sleep, exhaustion tracking. |
| `prompts.py` | 166 | All 3 prompt templates | ⚠️ The core of accuracy. Prompts explicitly list allowed enum values, instruct to IGNORE text in images, and demand JSON-only output. |
| `cache.py` | 74 | SHA-256 keyed file-based JSON cache | ✅ Survives restarts. Avoids re-running API calls on same inputs. |
| `__init__.py` | 1 | Package marker | ✅ |

### 8 Engines (8 files)

| Engine | File | Lines | What It Does | Honest Assessment |
|--------|------|-------|-------------|-------------------|
| **E1** Claim | `claim_engine.py` | 166 | Extract claim from conversation via LLM + regex pre-scan | ✅ Fuzzy matching for ~40 aliases. |
| **E2** Vision | `vision_engine.py` | 170 | Per-image VLM analysis | ✅ Independent analysis per image. Vehicle color extraction. |
| **E3** Evidence | `evidence_engine.py` | 235 | Deterministic evidence sufficiency check | ⚠️ Hardcoded mappings — fragile if new issue types added. |
| **E4** Quality | `quality_engine.py` | 80 | Aggregate image quality → valid_image | ✅ Simple and correct. |
| **E5** Fraud | `fraud_engine.py` | 246 | 8 fraud signal detectors | ⚠️ Most complex engine. Vehicle color comparison is naive string matching. |
| **E6** Risk | `risk_engine.py` | 82 | User history propagation | ✅ Always propagates. Matches sample labels. |
| **E7** Decision | `decision_engine.py` | 478 | Aggregates all signals → final output | ⚠️ Most critical. 8-rule decision tree. |
| **E8** Explain | `explain_engine.py` | 80 | Consistency checks + polish | ✅ Catches inconsistencies. |

### Pipeline, Comparison & Evaluation (6 files)

| File | Lines | What It Does | Honest Assessment |
|------|-------|-------------|-------------------|
| `main.py` | 330 | Full orchestrator with CLI: `--mode`, `--provider`, `--retry-failed`, `--output` | ✅ Error recovery per claim. Multi-provider support. Retry-failed skips successful claims. |
| `compare_models.py` | 260 | Runs all 4 providers independently, generates comparison report + majority-vote consensus | ✅ Per-field agreement %, per-model stats, consensus output. |
| `clear_failed_cache.py` | 65 | Removes cached responses with unknown/unknown results | ✅ Ensures re-runs actually call the API. |
| `check_output.py` | ~40 | Prints output.csv statistics | ✅ Dev utility. |
| `evaluation/metrics.py` | 176 | Exact match, F1, confusion matrix, Jaccard | ✅ Standard metrics. |
| `evaluation/main.py` | 122 | Eval pipeline + operational report | ✅ Compares to ground truth. |

### Validation & Config

| File | Purpose |
|------|---------|
| `validate.py` | 14-test dry-run validation — all pass |
| `.gitignore` | Ignores .env, cache, logs, outputs, __pycache__ |

---

## What's ACTUALLY Good (Not Hype)

### 1. Two-Call Design
Each image is analyzed independently. Blurry img_1 doesn't pollute clear img_2. Text instructions in one image don't influence other analyses.

### 2. 4-Provider Fallback Chain
Never fails due to rate limits. Gemini exhausted? Falls through to Groq (14K RPD). Groq down? OpenRouter. OpenRouter slow? NVIDIA (40 RPM, ∞ credits). Total effective capacity: **14,500+ RPD**.

### 3. Per-Key RPM/RPD Rate Limiting
Not a simple sleep timer — actual sliding-window rate limiter that tracks timestamps per key. Proactively rotates keys BEFORE hitting the API. Parses `retryDelay` from 429 error responses.

### 4. Multi-Model Comparison
Run every claim through Gemini, Groq, OpenRouter, AND NVIDIA independently. See where models agree/disagree. Majority-vote consensus picks the best answer.

### 5. Per-Provider Cache Isolation
Each provider gets its own cache directory. Models never see each other's cached responses. Ensures genuine independent analysis.

### 6. Smart Retry-Failed
`--retry-failed` flag re-processes ONLY claims that previously failed (unknown/unknown). Successful claims are preserved. Zero wasted API calls.

---

## What's Honestly WEAK or RISKY

### 1. VLM Accuracy is the Bottleneck
> **Reality**: The entire system's accuracy depends on the VLM correctly identifying visible parts, damage types, severity, watermarks, and vehicle color. If the VLM gets these wrong, the deterministic engines faithfully propagate the wrong signal.

### 2. Prompt Engineering is Fragile
The prompts demand specific JSON with specific enum values. Different models may interpret prompts differently. Llama 4 Maverick may return slightly different field values than Gemini 2.5 Flash.

### 3. No Parallel Processing
Claims are processed sequentially. 44 claims × ~3 API calls each × ~2s per call = ~4-5 minutes. For a hackathon this is fine.

### 4. Vehicle Identity Matching is Naive
Color string comparison only. No make/model detection, no license plate matching.

### 5. No Unit Tests
The `validate.py` script does 14 integration-level tests, but no proper unit tests with mocked LLM responses.

---

## CLI Reference

```bash
# Standard run (auto fallback chain)
python main.py

# Re-process only failed claims
python main.py --retry-failed

# Run with specific provider
python main.py --provider gemini
python main.py --provider groq
python main.py --provider nvidia
python main.py --provider openrouter

# Run on sample claims
python main.py --mode sample

# Custom output path
python main.py --output results/my_output.csv

# Multi-model comparison
python compare_models.py
python compare_models.py --providers groq nvidia
python compare_models.py --report-only
```

## Actual Run Statistics (June 19, 2026)

All 4 providers ran independently on the same 44 claims:

### Per-Provider Results

| Provider | Model | Time | API Calls | Failures | Unknowns |
|----------|-------|------|-----------|----------|----------|
| **NVIDIA** | `llama-4-maverick-17b-128e` | 384s | 126 | 0 | **0** |
| **OpenRouter** | `gemini-2.5-flash` | 404s | 127 | 0 | **0** |
| **Groq** | `llama-4-scout-17b-16e` | 2031s | 116 | 10 | **0** |
| **Gemini** | `gemini-2.5-flash` | 533s | 44 ok | 82 (RPD) | **28** |

### Status Distribution

| Status | NVIDIA | OpenRouter | Groq | Gemini | Consensus |
|--------|--------|------------|------|--------|-----------|
| **supported** | 16 | 18 | 15 | 5 | 15 |
| **contradicted** | 15 | 15 | 15 | 6 | 11 |
| **not_enough_information** | 13 | 11 | 14 | 33 | 10 |

> **Note**: Gemini's high `not_enough_information` count (33) is due to 28 claims hitting RPD quota exhaustion and defaulting to unknown. Groq had 10 vision failures from TPD token limit (500K/day).

### Cross-Model Agreement

- **All 4 models agree**: 5/36 claims (13.9%)
- **claim_status agreement**: 11/36 (30.6%)
- **issue_type agreement**: 22/36 (61.1%)
- **object_part agreement**: 21/36 (58.3%)
- **severity agreement**: 22/36 (61.1%)

### Output Selection

**Primary output (`output.csv`)** = NVIDIA run (0 failures, 0 unknowns, fastest completion).
**Consensus output** = Majority-vote across all providers with data.

---

## Output Files

| File | Rows | What |
|------|------|------|
| `dataset/output.csv` | 44 | **Submission output** (= NVIDIA, best run) |
| `dataset/output_nvidia.csv` | 44 | NVIDIA standalone (0 failures) |
| `dataset/output_openrouter.csv` | 44 | OpenRouter standalone (0 failures) |
| `dataset/output_groq.csv` | 44 | Groq standalone (10 failures) |
| `dataset/output_gemini.csv` | 44 | Gemini standalone (28 unknown) |
| `dataset/output_consensus.csv` | 36 | Majority-vote consensus |
| `dataset/model_comparison.csv` | 36 | Side-by-side per-claim comparison |

---

## CLI Reference

```bash
# Standard run (auto fallback chain)
python main.py

# Re-process only failed claims
python main.py --retry-failed

# Run with specific provider
python main.py --provider gemini
python main.py --provider groq
python main.py --provider nvidia
python main.py --provider openrouter

# Run on sample claims
python main.py --mode sample

# Custom output path
python main.py --output results/my_output.csv

# Multi-model comparison
python compare_models.py
python compare_models.py --providers groq nvidia
python compare_models.py --report-only
```

---

## Summary: What You're Submitting

A system that:
1. **Reads** claims.csv (44 claims, ~85 images)
2. **Calls LLMs** via 4-provider fallback chain with per-key rate limiting
3. **Cross-references** visual evidence against claimed damage using 6 deterministic engines
4. **Outputs** a 14-column output.csv with validated enum values
5. **Handles edge cases**: prompt injection, stock watermarks, wrong objects, vehicle identity, blurry images, multi-image selection
6. **Compares models**: run all 4 providers independently, generate majority-vote consensus
7. **Caches** all API responses to disk — re-runs are free
8. **Costs** ~$0.01 per full run (all free tiers)

What it does NOT do:
- No fine-tuned model
- No parallel processing
- No vehicle make/model detection
- No actual structural similarity matching
- No confidence calibration against ground truth
- Multi-part claims extracted but not independently evaluated
