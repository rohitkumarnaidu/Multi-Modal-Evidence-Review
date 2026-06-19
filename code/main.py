"""
Multi-Modal Evidence Review — Main Orchestrator.

Entry point for processing all claims in claims.csv → output.csv.

Pipeline per claim:
  1. Load claim data, user history, evidence requirements
  2. Engine 1: Claim extraction (text-only LLM call)
  3. Engine 2: Per-image vision analysis (VLM call per image)
  4. Engine 3: Evidence sufficiency check (deterministic)
  5. Engine 4: Image quality assessment (deterministic)
  6. Engine 5: Fraud detection (deterministic cross-reference)
  7. Engine 6: User risk propagation (deterministic)
  8. Engine 7: Decision aggregation
  9. Engine 8: Explainability polish
  10. Write output.csv

Two-call design:
  - Call 1: LLM text-only for claim extraction
  - Call 2: VLM per-image for vision analysis (N calls for N images)
  - All other engines are deterministic
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add code/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLAIMS_CSV,
    CODE_DIR,
    DATASET_DIR,
    INTER_CLAIM_DELAY,
    METRICS_LOG,
    OUTPUT_CSV,
    SAMPLE_CLAIMS_CSV,
)
from data_loader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
    write_output_csv,
)
from engines.claim_engine import extract_claim_with_llm
from engines.decision_engine import make_decision
from engines.evidence_engine import check_evidence_sufficiency
from engines.explain_engine import polish_output
from engines.fraud_engine import detect_fraud
from engines.quality_engine import assess_image_quality
from engines.risk_engine import get_risk_summary, get_user_risk_flags
from engines.vision_engine import analyze_all_images
from llm.multi_provider_client import MultiProviderClient
from models import ClaimOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CODE_DIR / "run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def process_single_claim(
    claim,
    user_history_map: dict,
    evidence_requirements: list,
    llm_client: MultiProviderClient,
    claim_index: int,
    total_claims: int,
) -> ClaimOutput:
    """Process a single claim through the full 8-engine pipeline."""
    logger.info(
        f"Processing claim {claim_index + 1}/{total_claims}: "
        f"user={claim.user_id}, object={claim.claim_object}, "
        f"images={len(claim.image_path_list)}"
    )

    try:
        # ── Engine 1: Claim Extraction (LLM Call 1) ──────────────────────
        extraction = extract_claim_with_llm(claim, llm_client)
        logger.info(f"  → Extracted: part={extraction.claimed_object_part}, issue={extraction.claimed_issue_type}")

        # ── Engine 2: Per-Image Vision Analysis (VLM Calls) ──────────────
        image_analyses = analyze_all_images(claim, llm_client)
        logger.info(f"  → Analyzed {len(image_analyses)} images")

        # Mark which images show the claimed part and damage
        for a in image_analyses:
            a.shows_claimed_part = (
                a.visible_object_part == extraction.claimed_object_part
            )
            a.shows_claimed_damage = (
                a.visible_issue_type == extraction.claimed_issue_type
                and a.visible_issue_type not in ("none", "unknown")
            )

        # ── Engine 3: Evidence Sufficiency (Deterministic) ───────────────
        evidence = check_evidence_sufficiency(
            claim, extraction, image_analyses, evidence_requirements
        )
        logger.info(f"  → Evidence met: {evidence.evidence_standard_met}")

        # ── Engine 4: Image Quality (Deterministic) ──────────────────────
        quality = assess_image_quality(image_analyses)
        logger.info(f"  → Valid image: {quality['valid_image']}")

        # ── Engine 5: Fraud Detection (Deterministic) ────────────────────
        fraud = detect_fraud(claim, extraction, image_analyses)
        logger.info(f"  → Fraud flags: {fraud.risk_flags}")

        # ── Engine 6: User Risk (Deterministic) ─────────────────────────
        user_history = user_history_map.get(claim.user_id)
        user_risk_flags = get_user_risk_flags(user_history)
        user_risk_summary = get_risk_summary(user_history)
        logger.info(f"  → User risk flags: {user_risk_flags}")

        # ── Engine 7: Decision (Deterministic Aggregation) ───────────────
        output = make_decision(
            claim=claim,
            extraction=extraction,
            image_analyses=image_analyses,
            evidence=evidence,
            fraud=fraud,
            quality=quality,
            user_risk_flags=user_risk_flags,
            user_risk_summary=user_risk_summary,
        )

        # ── Engine 8: Explainability Polish ──────────────────────────────
        output = polish_output(output)

        logger.info(
            f"  ✓ Result: status={output.claim_status}, "
            f"part={output.object_part}, issue={output.issue_type}, "
            f"severity={output.severity}"
        )
        return output

    except Exception as e:
        logger.error(f"  ✗ Error processing claim {claim.user_id}: {e}", exc_info=True)
        # Return safe fallback
        return ClaimOutput(
            user_id=claim.user_id,
            image_paths=claim.image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met="false",
            evidence_standard_met_reason=f"Processing error: {str(e)[:100]}",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="The claim could not be processed automatically.",
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
        )


def run_pipeline(
    claims_csv: Path = CLAIMS_CSV,
    output_csv: Path = OUTPUT_CSV,
    mode: str = "test",
    retry_failed: bool = False,
    provider: str | None = None,
):
    """Run the full pipeline on all claims.
    
    Args:
        claims_csv: Path to input claims CSV
        output_csv: Path for output CSV
        mode: "test" for claims.csv, "sample" for sample_claims.csv
        retry_failed: If True, only re-process claims that previously failed
        provider: If set, force a specific provider (gemini/groq/openrouter/nvidia)
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Multi-Modal Evidence Review Pipeline")
    logger.info(f"Mode: {mode}")
    logger.info(f"Provider: {provider or 'auto (fallback chain)'}")
    logger.info(f"Input: {claims_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    claims = load_claims(claims_csv)
    user_history = load_user_history()
    evidence_reqs = load_evidence_requirements()

    logger.info(f"Loaded {len(claims)} claims, {len(user_history)} user histories, {len(evidence_reqs)} requirements")

    # ── Initialize LLM client (multi-provider fallback or single) ────────
    llm_client = MultiProviderClient(only_provider=provider)

    # When running per-provider comparison, use a separate cache directory
    # so each model sees fresh images (no cross-model cache hits)
    if provider:
        from llm.cache import ResponseCache
        provider_cache_dir = CODE_DIR / f".cache_{provider}"
        llm_client.cache = ResponseCache(cache_dir=provider_cache_dir)
        # Also set on child providers
        for _, client in llm_client.providers:
            client.cache = llm_client.cache

    # ── Load existing results for retry-failed mode ──────────────────────
    existing_results: dict[str, ClaimOutput] = {}
    if retry_failed and output_csv.exists():
        import csv
        with open(output_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id", "")
                # Mark as "needs retry" if issue_type and object_part are both unknown
                is_failed = (
                    row.get("issue_type") == "unknown"
                    and row.get("object_part") == "unknown"
                )
                if uid and not is_failed:
                    # Keep successful results — rebuild ClaimOutput
                    existing_results[uid] = ClaimOutput(
                        user_id=uid,
                        image_paths=row.get("image_paths", ""),
                        user_claim=row.get("user_claim", ""),
                        claim_object=row.get("claim_object", ""),
                        evidence_standard_met=row.get("evidence_standard_met", "false"),
                        evidence_standard_met_reason=row.get("evidence_standard_met_reason", ""),
                        risk_flags=row.get("risk_flags", "none"),
                        issue_type=row.get("issue_type", "unknown"),
                        object_part=row.get("object_part", "unknown"),
                        claim_status=row.get("claim_status", "not_enough_information"),
                        claim_status_justification=row.get("claim_status_justification", ""),
                        supporting_image_ids=row.get("supporting_image_ids", "none"),
                        valid_image=row.get("valid_image", "false"),
                        severity=row.get("severity", "unknown"),
                    )
        if existing_results:
            logger.info(
                f"Retry-failed mode: keeping {len(existing_results)} successful results, "
                f"re-processing {len(claims) - len(existing_results)} failed claims"
            )

    # ── Process claims ───────────────────────────────────────────────────
    outputs: list[ClaimOutput] = []
    total = len(claims)

    for i, claim in enumerate(claims):
        # Skip already-successful claims in retry-failed mode
        if retry_failed and claim.user_id in existing_results:
            outputs.append(existing_results[claim.user_id])
            logger.info(
                f"Skipping claim {i + 1}/{total}: {claim.user_id} (already successful)"
            )
            continue

        output = process_single_claim(
            claim=claim,
            user_history_map=user_history,
            evidence_requirements=evidence_reqs,
            llm_client=llm_client,
            claim_index=i,
            total_claims=total,
        )
        outputs.append(output)

        # Brief pause between claims to respect rate limits
        if i < total - 1:
            time.sleep(INTER_CLAIM_DELAY)

    # ── Write output ─────────────────────────────────────────────────────
    rows = [o.to_csv_row() for o in outputs]
    write_output_csv(rows, output_csv)

    # ── Log metrics ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    metrics = {
        "mode": mode,
        "provider": provider or "auto",
        "total_claims": total,
        "elapsed_seconds": round(elapsed, 2),
        "avg_seconds_per_claim": round(elapsed / max(1, total), 2),
        "llm_stats": llm_client.tracker.stats,
        "cache_stats": llm_client.cache.stats,
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Multi-Provider Stats: {json.dumps(llm_client.stats, indent=2)}")
    logger.info("=" * 60)

    # Save metrics
    metrics_path = METRICS_LOG if not provider else (CODE_DIR / f".metrics_{provider}.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save metrics: {e}")

    return outputs, metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review")
    parser.add_argument(
        "--mode",
        choices=["test", "sample"],
        default="test",
        help="Run on test claims (default) or sample claims",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to dataset/output.csv)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only re-process claims that previously failed (unknown/unknown)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "openrouter", "nvidia"],
        default=None,
        help="Force a specific LLM provider (for model comparison)",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        claims_csv = SAMPLE_CLAIMS_CSV
        output_csv = Path(args.output) if args.output else (DATASET_DIR / "sample_output.csv")
    else:
        claims_csv = CLAIMS_CSV
        if args.output:
            output_csv = Path(args.output)
        elif args.provider:
            output_csv = DATASET_DIR / f"output_{args.provider}.csv"
        else:
            output_csv = OUTPUT_CSV

    run_pipeline(
        claims_csv=claims_csv,
        output_csv=output_csv,
        mode=args.mode,
        retry_failed=args.retry_failed,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()

