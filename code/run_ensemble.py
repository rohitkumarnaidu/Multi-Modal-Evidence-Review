"""
Ensemble Runner — Run all providers and ensemble their outputs.

Usage:
    python run_ensemble.py
    python run_ensemble.py --fast   (skip re-running cached providers)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CODE_DIR, DATASET_DIR
from engines.ensemble_engine import analyze_ensemble_agreement, run_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-provider ensemble")
    parser.add_argument("--fast", action="store_true", help="Skip re-running providers, use existing outputs")
    args = parser.parse_args()

    # Provider output paths
    output_files = {}
    provider_configs = [
        ("nvidia", "output_nvidia.csv"),
        ("openrouter", "output_openrouter.csv"),
        ("gemini", "output_gemini.csv"),
        ("groq", "output_groq.csv"),
    ]

    if not args.fast:
        # Run pipeline for each available provider
        from main import run_pipeline
        for name, out_name in provider_configs:
            out_path = DATASET_DIR / out_name
            logger.info(f"Running pipeline with provider: {name}")
            try:
                run_pipeline(
                    output_csv=out_path,
                    mode="test",
                    provider=name,
                )
                if out_path.exists():
                    output_files[name] = out_path
            except Exception as e:
                logger.warning(f"Provider {name} failed: {e}")
    else:
        for name, out_name in provider_configs:
            out_path = DATASET_DIR / out_name
            if out_path.exists():
                output_files[name] = out_path
                logger.info(f"Found existing output for {name}: {out_path}")

    if not output_files:
        logger.error("No provider outputs available. Run without --fast first.")
        return

    logger.info(f"Ensembling {len(output_files)} providers: {list(output_files.keys())}")

    # Run ensemble
    ensemble_path = DATASET_DIR / "output_ensemble.csv"
    ensemble_rows = run_ensemble(
        output_files=output_files,
        ensemble_output=ensemble_path,
    )

    # Analyze agreement
    provider_rows = {}
    for p, path in output_files.items():
        import csv
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            provider_rows[p] = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

    agreement = analyze_ensemble_agreement(provider_rows)

    # Save agreement analysis
    agreement_path = CODE_DIR / "ensemble_agreement.json"
    with open(agreement_path, "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)
    logger.info(f"Agreement analysis saved to {agreement_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    print(f"Providers: {list(output_files.keys())}")
    print(f"Total claims: {len(ensemble_rows)}")
    low_conf = [r for r in ensemble_rows if r.get('_confidence_avg', 1.0) < 0.75]
    print(f"Low confidence: {len(low_conf)}/{len(ensemble_rows)}")
    print(f"\nMajority Agreement (field-level):")
    for p, fields in agreement.get("majority_agreement", {}).items():
        avg = sum(fields.values()) / max(1, len(fields))
        print(f"  {p}: avg={avg:.1%}")
    print(f"\nEnsemble output: {ensemble_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
