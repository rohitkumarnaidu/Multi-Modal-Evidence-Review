"""
Evaluation Main — Run system on sample_claims.csv and compare against ground truth.

Usage:
    python evaluation/main.py              # full eval (uses cached output if exists)
    python evaluation/main.py --fresh      # re-run pipeline from scratch
    python evaluation/main.py --no-run     # only compare existing output
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, SAMPLE_CLAIMS_CSV
from data_loader import load_sample_claims
from evaluation.metrics import compute_all_metrics, format_report, generate_html_report

logger = logging.getLogger(__name__)


def run_evaluation(fresh: bool = False, no_run: bool = False):
    """Run evaluation on sample_claims.csv.
    
    Args:
        fresh: If True, re-run the pipeline even if cached output exists
        no_run: If True, skip pipeline execution entirely (use existing output)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Evaluation Pipeline")
    logger.info("=" * 60)

    sample_output = DATASET_DIR / "sample_output.csv"
    eval_dir = Path(__file__).resolve().parent

    # Step 1: Get predictions
    if no_run:
        if not sample_output.exists():
            logger.error(f"No existing output found at {sample_output}")
            return None
        logger.info(f"Using existing output: {sample_output}")
        metrics = {}
    elif fresh:
        from main import run_pipeline
        predictions_list, metrics = run_pipeline(
            claims_csv=SAMPLE_CLAIMS_CSV,
            output_csv=sample_output,
            mode="sample",
        )
    elif not sample_output.exists():
        sample_candidates = [
            DATASET_DIR / "sample_output_nvidia_v6.csv",
            DATASET_DIR / "sample_output_nvidia_v5.csv",
            DATASET_DIR / "sample_output_nvidia.csv",
        ]
        sample_source = None
        for cand in sample_candidates:
            if cand.exists():
                sample_source = cand
                logger.info(f"Found existing sample output: {cand}")
                break
        if sample_source:
            shutil.copy2(sample_source, sample_output)
            logger.info(f"Copied {sample_source} → {sample_output}")
            metrics = {}
        else:
            from main import run_pipeline
            predictions_list, metrics = run_pipeline(
                claims_csv=SAMPLE_CLAIMS_CSV,
                output_csv=sample_output,
                mode="sample",
            )
    else:
        logger.info(f"Using cached output: {sample_output}")
        metrics = {
            "mode": "sample",
            "total_claims": 0,
            "elapsed_seconds": 0,
        }

    # Step 2: Load ground truth
    ground_truth = load_sample_claims(SAMPLE_CLAIMS_CSV)
    logger.info(f"Loaded {len(ground_truth)} ground truth rows")

    # Step 3: Load predictions
    predictions = []
    with open(sample_output, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({k.strip(): v.strip() for k, v in row.items()})

    # Step 4: Compute metrics
    eval_metrics = compute_all_metrics(predictions, ground_truth)
    from output_validation import validate_output_rows
    consistency_errors = validate_output_rows(predictions, expected_rows=len(ground_truth))
    eval_metrics["consistency_violations"] = consistency_errors

    # Step 5: Load history and save new snapshot
    history_path = eval_dir / "metrics_history.json"
    history = _load_history(history_path)
    eval_metrics["_timestamp"] = datetime.now().isoformat()
    _save_history(history_path, eval_metrics, history)

    # Step 6: Generate reports
    report_md = format_report(eval_metrics)
    report_md += "\n\n## Consistency Violations\n"
    if consistency_errors:
        report_md += "\n".join(f"- {error}" for error in consistency_errors)
    else:
        report_md += "No output schema or evidence-consistency violations found."
    report_md += "\n\n" + _build_operational_analysis(metrics)

    report_html = generate_html_report(eval_metrics, history)

    with open(eval_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(eval_dir / "evaluation_report.html", "w", encoding="utf-8") as f:
        f.write(report_html)
    logger.info(f"Reports saved to {eval_dir / 'evaluation_report.md'} and .html")

    with open(eval_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, default=str)

    _print_summary(eval_metrics)
    return eval_metrics


def _load_history(path: Path) -> list[dict]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_history(path: Path, current: dict, history: list[dict]):
    snapshot = {}
    for k, v in current.items():
        if isinstance(v, dict) and "accuracy" in v:
            snapshot[k] = {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
    snapshot["_timestamp"] = current.get("_timestamp", datetime.now().isoformat())
    history.append(snapshot)
    if len(history) > 20:
        history = history[-20:]
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _build_operational_analysis(pipeline_metrics: dict) -> str:
    llm = pipeline_metrics.get("llm_stats", {})
    cache = pipeline_metrics.get("cache_stats", {})
    provider = pipeline_metrics.get("provider_stats", {})

    lines = [
        "# Operational Analysis\n",
        "## Strategy Comparison\n",
        "### Strategy A: Two-Call Design (Primary — Used for output.csv)",
        "- **Call 1**: Text-only LLM for claim extraction from conversation",
        "- **Call 2**: VLM per-image for independent vision analysis",
        "- **Post-processing**: Deterministic engines for evidence, quality, fraud, risk, decision",
        "- **Pros**: Independent image verification, controllable, debuggable, no cross-contamination",
        "- **Cons**: More API calls than single mega-prompt\n",
        "### Strategy B: Single Mega-Prompt (Considered, Not Used)",
        "- All images + conversation + history in one VLM call",
        "- **Rejected because**: Cannot independently compare vehicle identities,",
        "  cannot pick best supporting image, susceptible to text injection in images\n",
        "## Model Calls\n",
        f"- Total API calls: {llm.get('total_calls', 'N/A')}",
        f"- Cached calls: {llm.get('cached_calls', 'N/A')}",
        f"- Failed calls: {llm.get('failed_calls', 'N/A')}\n",
        f"- Provider routing: {provider.get('provider_usage', {})}",
        f"- Selective second opinions: {provider.get('second_opinion_successes', 0)}/{provider.get('second_opinion_requests', 0)} successful\n",
        "## Token Usage\n",
        f"- Total input tokens: {_fmt_num(llm.get('total_input_tokens', 'N/A'))}",
        f"- Total output tokens: {_fmt_num(llm.get('total_output_tokens', 'N/A'))}",
        f"- Image tokens (estimated): ~258 tokens per image\n",
        "## Cost Estimate\n",
        f"- Estimated cost: ${llm.get('estimated_cost_usd', 'N/A')}",
        "- Pricing assumptions: Gemini 2.5 Flash ($0.15/1M input, $0.60/1M output)",
        "- Image tokens included in input token count\n",
        "## Runtime\n",
        f"- Total elapsed: {pipeline_metrics.get('elapsed_seconds', 'N/A')}s",
        f"- Avg per claim: {pipeline_metrics.get('avg_seconds_per_claim', 'N/A')}s\n",
        "## TPM/RPM Strategy\n",
        "- Rate limiting: 0.5s minimum interval between API calls",
        "- Retry: Exponential backoff, max 5 attempts, max 60s delay",
        "- Caching: File-based JSON cache (SHA-256 key), survives restarts",
        "- Batching: Sequential processing with rate-limit pauses",
        f"- Cache hit rate: {cache.get('hit_rate', 0):.1%}\n",
        "## Images Processed\n",
        f"- Total images: estimated ~115 (sample + test)",
        "- Image sizes: 6KB - 355KB (mostly JPEG)",
    ]
    return "\n".join(lines)


def _fmt_num(val):
    if isinstance(val, (int, float)):
        return f"{val:,}"
    return str(val)


def _print_summary(metrics: dict):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            print(f"  {m['field']:30s}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")
    if "risk_flags_f1" in metrics:
        print(f"  {'risk_flags_f1':30s}: {metrics['risk_flags_f1']['micro_f1']:.4f}")
    if "supporting_images_jaccard" in metrics:
        print(f"  {'supporting_images_jaccard':30s}: {metrics['supporting_images_jaccard']['avg_jaccard']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--fresh", action="store_true", help="Re-run pipeline from scratch")
    parser.add_argument("--no-run", action="store_true", help="Skip pipeline execution")
    args = parser.parse_args()
    run_evaluation(fresh=args.fresh, no_run=args.no_run)
