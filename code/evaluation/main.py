"""
Evaluation Main — Run system on sample_claims.csv and compare against ground truth.

Usage:
    python evaluation/main.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add code/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, SAMPLE_CLAIMS_CSV
from data_loader import load_sample_claims, write_output_csv
from evaluation.metrics import compute_all_metrics, format_report

logger = logging.getLogger(__name__)


def run_evaluation():
    """Run evaluation on sample_claims.csv.
    
    Steps:
      1. Run pipeline on sample_claims.csv (inputs only)
      2. Compare predictions against ground truth labels
      3. Generate evaluation report
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Evaluation Pipeline")
    logger.info("=" * 60)

    # Step 1: Run pipeline on sample claims
    from main import run_pipeline

    sample_output = DATASET_DIR / "sample_output.csv"
    predictions_list, metrics = run_pipeline(
        claims_csv=SAMPLE_CLAIMS_CSV,
        output_csv=sample_output,
        mode="sample",
    )

    # Step 2: Load ground truth
    ground_truth = load_sample_claims(SAMPLE_CLAIMS_CSV)
    logger.info(f"Loaded {len(ground_truth)} ground truth rows")

    # Step 3: Load predictions from output
    import csv
    predictions = []
    with open(sample_output, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({k.strip(): v.strip() for k, v in row.items()})

    # Step 4: Compute metrics
    eval_metrics = compute_all_metrics(predictions, ground_truth)

    # Step 5: Generate report
    report = format_report(eval_metrics)

    # Add operational analysis
    report += "\n\n" + _build_operational_analysis(metrics)

    # Step 6: Save report
    eval_dir = Path(__file__).resolve().parent
    report_path = eval_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Evaluation report saved to {report_path}")

    # Also save raw metrics
    metrics_path = eval_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, default=str)
    logger.info(f"Raw metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key in sorted(eval_metrics.keys()):
        if key.endswith("_accuracy"):
            m = eval_metrics[key]
            print(f"  {m['field']:30s}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")
    if "risk_flags_f1" in eval_metrics:
        print(f"  {'risk_flags_f1':30s}: {eval_metrics['risk_flags_f1']['micro_f1']:.4f}")
    if "supporting_images_jaccard" in eval_metrics:
        print(f"  {'supporting_images_jaccard':30s}: {eval_metrics['supporting_images_jaccard']['avg_jaccard']:.4f}")
    print("=" * 60)

    return eval_metrics


def _build_operational_analysis(pipeline_metrics: dict) -> str:
    """Build operational analysis section for the report."""
    llm = pipeline_metrics.get("llm_stats", {})
    cache = pipeline_metrics.get("cache_stats", {})

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
        "## Token Usage\n",
        f"- Total input tokens: {llm.get('total_input_tokens', 'N/A'):,}",
        f"- Total output tokens: {llm.get('total_output_tokens', 'N/A'):,}",
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


if __name__ == "__main__":
    run_evaluation()
