"""
Multi-Model Comparison Runner.

Runs the pipeline with each configured provider separately, then generates
a comparison report showing where models agree/disagree.

Usage:
    python compare_models.py                    # Run all providers
    python compare_models.py --providers gemini groq nvidia  # Run specific ones
    python compare_models.py --report-only      # Just generate report from existing outputs

Output:
    dataset/output_gemini.csv
    dataset/output_groq.csv
    dataset/output_openrouter.csv
    dataset/output_nvidia.csv
    dataset/model_comparison.csv    ← side-by-side comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_DIR, CODE_DIR


PROVIDERS = ["gemini", "groq", "openrouter", "nvidia"]
COMPARE_FIELDS = ["claim_status", "issue_type", "object_part", "severity"]


def run_provider(provider: str) -> bool:
    """Run the pipeline with a specific provider. Returns True on success."""
    output_csv = DATASET_DIR / f"output_{provider}.csv"
    print(f"\n{'='*60}")
    print(f"  Running pipeline with: {provider.upper()}")
    print(f"  Output: {output_csv}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "main.py",
        "--provider", provider,
        "--output", str(output_csv),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(CODE_DIR),
            timeout=1800,  # 30 min timeout per provider
        )
        if result.returncode == 0:
            print(f"\n✓ {provider} completed successfully")
            return True
        else:
            print(f"\n✗ {provider} failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ {provider} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"\n✗ {provider} error: {e}")
        return False


def load_output(provider: str) -> dict[str, dict]:
    """Load a provider's output CSV into {user_id: row_dict}."""
    csv_path = DATASET_DIR / f"output_{provider}.csv"
    if not csv_path.exists():
        return {}
    rows = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id", "")
            if uid:
                rows[uid] = dict(row)
    return rows


def generate_comparison_report(providers: list[str]):
    """Generate a side-by-side comparison of model outputs."""
    print(f"\n{'='*60}")
    print(f"  GENERATING COMPARISON REPORT")
    print(f"{'='*60}\n")

    # Load all outputs
    outputs = {}
    for p in providers:
        data = load_output(p)
        if data:
            outputs[p] = data
            print(f"  Loaded {len(data)} rows from {p}")
        else:
            print(f"  ⚠ No output found for {p}")

    if len(outputs) < 2:
        print("\n✗ Need at least 2 provider outputs to compare")
        return

    # Get all user_ids
    all_users = set()
    for data in outputs.values():
        all_users.update(data.keys())
    all_users = sorted(all_users)

    # ── Build comparison CSV ─────────────────────────────────────────────
    comparison_rows = []
    agreement_count = 0
    disagreement_count = 0
    field_agreements = {f: 0 for f in COMPARE_FIELDS}
    field_disagreements = {f: 0 for f in COMPARE_FIELDS}

    for uid in all_users:
        row = {"user_id": uid}

        # Get claim_object from any provider
        for p in outputs:
            if uid in outputs[p]:
                row["claim_object"] = outputs[p][uid].get("claim_object", "")
                break

        all_agree = True
        for field in COMPARE_FIELDS:
            values = {}
            for p in outputs:
                if uid in outputs[p]:
                    values[p] = outputs[p][uid].get(field, "N/A")
                else:
                    values[p] = "MISSING"

            # Add per-provider values
            for p in outputs:
                row[f"{field}_{p}"] = values.get(p, "MISSING")

            # Check agreement
            unique_values = set(v for v in values.values() if v not in ("MISSING", "unknown"))
            if len(unique_values) <= 1:
                row[f"{field}_consensus"] = "✓ AGREE"
                field_agreements[field] += 1
            else:
                row[f"{field}_consensus"] = f"✗ DISAGREE: {unique_values}"
                field_disagreements[field] += 1
                all_agree = False

        if all_agree:
            agreement_count += 1
        else:
            disagreement_count += 1

        comparison_rows.append(row)

    # Write comparison CSV
    comparison_csv = DATASET_DIR / "model_comparison.csv"
    if comparison_rows:
        fieldnames = comparison_rows[0].keys()
        with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

    # ── Print summary ────────────────────────────────────────────────────
    active_providers = list(outputs.keys())
    total = len(all_users)

    print(f"\n{'='*60}")
    print(f"  COMPARISON RESULTS ({len(active_providers)} models)")
    print(f"{'='*60}")
    print(f"\n  Models: {', '.join(active_providers)}")
    print(f"  Claims: {total}")
    print(f"\n  Overall Agreement:")
    print(f"    ✓ All models agree: {agreement_count}/{total} ({100*agreement_count/max(1,total):.1f}%)")
    print(f"    ✗ Disagreement:     {disagreement_count}/{total} ({100*disagreement_count/max(1,total):.1f}%)")

    print(f"\n  Per-Field Agreement:")
    for field in COMPARE_FIELDS:
        agree = field_agreements[field]
        total_f = agree + field_disagreements[field]
        pct = 100 * agree / max(1, total_f)
        print(f"    {field:30s} {agree}/{total_f} agree ({pct:.1f}%)")

    # ── Per-model stats ──────────────────────────────────────────────────
    print(f"\n  Per-Model Status Distribution:")
    for p in active_providers:
        statuses = Counter()
        unknowns = 0
        for uid, row in outputs[p].items():
            status = row.get("claim_status", "?")
            statuses[status] += 1
            if row.get("issue_type") == "unknown" and row.get("object_part") == "unknown":
                unknowns += 1
        print(f"\n    {p.upper()}:")
        for status, count in sorted(statuses.items()):
            print(f"      {status}: {count}")
        if unknowns:
            print(f"      ⚠ API failures (unknown/unknown): {unknowns}")

    # ── Majority vote result ─────────────────────────────────────────────
    print(f"\n  Generating majority-vote consensus output...")
    consensus_rows = []
    for uid in all_users:
        # Take majority vote for claim_status
        votes = Counter()
        best_row = None
        for p in active_providers:
            if uid in outputs[p]:
                status = outputs[p][uid].get("claim_status", "")
                if outputs[p][uid].get("issue_type") != "unknown":
                    votes[status] += 1
                    if best_row is None:
                        best_row = outputs[p][uid]

        if best_row:
            if votes:
                majority_status = votes.most_common(1)[0][0]
                # Find the provider that voted for majority and use their full row
                for p in active_providers:
                    if uid in outputs[p] and outputs[p][uid].get("claim_status") == majority_status:
                        best_row = outputs[p][uid]
                        break
            consensus_rows.append(best_row)
        else:
            # All providers failed — use first available
            for p in active_providers:
                if uid in outputs[p]:
                    consensus_rows.append(outputs[p][uid])
                    break

    # Write consensus output
    consensus_csv = DATASET_DIR / "output_consensus.csv"
    if consensus_rows:
        fieldnames = consensus_rows[0].keys()
        with open(consensus_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(consensus_rows)
        print(f"  → Written to {consensus_csv}")

    print(f"\n  Comparison CSV: {comparison_csv}")
    print(f"  Consensus CSV:  {consensus_csv}")
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline with multiple models and compare")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=None,
        help="Which providers to run (default: all configured)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip running, just generate comparison from existing outputs",
    )
    args = parser.parse_args()

    # Determine which providers to use
    selected = args.providers or PROVIDERS

    if not args.report_only:
        # Run each provider
        results = {}
        for provider in selected:
            success = run_provider(provider)
            results[provider] = success

        print(f"\n{'='*60}")
        print(f"  RUN SUMMARY")
        print(f"{'='*60}")
        for p, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {p}")

    # Generate comparison report from all available outputs
    generate_comparison_report(selected)


if __name__ == "__main__":
    main()
