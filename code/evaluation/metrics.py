"""
Evaluation Metrics for Multi-Modal Evidence Review.

Per-field accuracy metrics comparing predictions against ground truth.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def exact_match_accuracy(
    predictions: list[str],
    ground_truth: list[str],
    field_name: str = "",
) -> dict:
    """Compute exact match accuracy for a single field."""
    assert len(predictions) == len(ground_truth), (
        f"Length mismatch: {len(predictions)} vs {len(ground_truth)}"
    )
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip().lower() == g.strip().lower())
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    # Error analysis
    errors = []
    for i, (p, g) in enumerate(zip(predictions, ground_truth)):
        if p.strip().lower() != g.strip().lower():
            errors.append({
                "index": i,
                "predicted": p.strip(),
                "expected": g.strip(),
            })

    return {
        "field": field_name,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "errors": errors[:10],  # Top 10 errors
    }


def risk_flags_f1(
    pred_flags: list[str],
    true_flags: list[str],
) -> dict:
    """Compute F1 score for multi-label risk flags.
    
    Each entry is a semicolon-separated string of flags.
    """
    total_precision_num = 0
    total_precision_den = 0
    total_recall_num = 0
    total_recall_den = 0

    per_row = []
    for p, t in zip(pred_flags, true_flags):
        pred_set = _parse_flags(p)
        true_set = _parse_flags(t)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_row.append({
            "predicted": sorted(pred_set),
            "expected": sorted(true_set),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

        total_precision_num += tp
        total_precision_den += tp + fp
        total_recall_num += tp
        total_recall_den += tp + fn

    micro_precision = total_precision_num / max(1, total_precision_den)
    micro_recall = total_recall_num / max(1, total_recall_den)
    micro_f1 = (
        2 * micro_precision * micro_recall / max(0.001, micro_precision + micro_recall)
    )
    macro_f1 = sum(r["f1"] for r in per_row) / max(1, len(per_row))

    return {
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_row": [r for r in per_row if r["f1"] < 1.0][:10],  # Show errors
    }


def confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    field_name: str = "",
) -> dict:
    """Build confusion matrix for categorical fields."""
    all_labels = sorted(set(
        [p.strip().lower() for p in predictions]
        + [g.strip().lower() for g in ground_truth]
    ))

    matrix = {label: Counter() for label in all_labels}
    for p, g in zip(predictions, ground_truth):
        matrix[g.strip().lower()][p.strip().lower()] += 1

    return {
        "field": field_name,
        "labels": all_labels,
        "matrix": {k: dict(v) for k, v in matrix.items()},
    }


def supporting_images_jaccard(
    pred_ids: list[str],
    true_ids: list[str],
) -> dict:
    """Compute Jaccard similarity for supporting_image_ids."""
    scores = []
    for p, t in zip(pred_ids, true_ids):
        pred_set = _parse_flags(p)
        true_set = _parse_flags(t)

        if not pred_set and not true_set:
            scores.append(1.0)
        elif not pred_set or not true_set:
            scores.append(0.0)
        else:
            intersection = len(pred_set & true_set)
            union = len(pred_set | true_set)
            scores.append(intersection / union if union > 0 else 0.0)

    avg = sum(scores) / max(1, len(scores))
    return {
        "avg_jaccard": round(avg, 4),
        "per_row_scores": [round(s, 4) for s in scores],
    }


def compute_all_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Compute all metrics across all fields.
    
    predictions and ground_truth are lists of dicts with column names as keys.
    """
    results = {}

    # Key categorical fields
    for field in [
        "claim_status", "issue_type", "object_part",
        "evidence_standard_met", "valid_image", "severity",
    ]:
        pred_vals = [str(p.get(field, "")).strip().lower() for p in predictions]
        true_vals = [str(g.get(field, "")).strip().lower() for g in ground_truth]
        results[f"{field}_accuracy"] = exact_match_accuracy(
            pred_vals, true_vals, field
        )
        results[f"{field}_confusion"] = confusion_matrix(
            pred_vals, true_vals, field
        )

    # Risk flags (multi-label)
    pred_flags = [str(p.get("risk_flags", "none")) for p in predictions]
    true_flags = [str(g.get("risk_flags", "none")) for g in ground_truth]
    results["risk_flags_f1"] = risk_flags_f1(pred_flags, true_flags)

    # Supporting image IDs (Jaccard)
    pred_ids = [str(p.get("supporting_image_ids", "none")) for p in predictions]
    true_ids = [str(g.get("supporting_image_ids", "none")) for g in ground_truth]
    results["supporting_images_jaccard"] = supporting_images_jaccard(pred_ids, true_ids)

    return results


def _parse_flags(flags_str: str) -> set[str]:
    """Parse semicolon-separated flags into a set."""
    flags_str = flags_str.strip().lower()
    if flags_str in ("none", ""):
        return set()
    return {f.strip() for f in flags_str.split(";") if f.strip() and f.strip() != "none"}


def format_report(metrics: dict) -> str:
    """Format metrics into a readable markdown report."""
    lines = ["# Evaluation Results\n"]

    # Summary table
    lines.append("## Accuracy Summary\n")
    lines.append("| Field | Accuracy | Correct / Total |")
    lines.append("|-------|----------|-----------------|")
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            lines.append(
                f"| {m['field']} | {m['accuracy']:.1%} | {m['correct']}/{m['total']} |"
            )

    # Risk flags F1
    if "risk_flags_f1" in metrics:
        rf = metrics["risk_flags_f1"]
        lines.append(f"\n## Risk Flags F1")
        lines.append(f"- Micro F1: {rf['micro_f1']:.4f}")
        lines.append(f"- Macro F1: {rf['macro_f1']:.4f}")

    # Supporting images Jaccard
    if "supporting_images_jaccard" in metrics:
        sj = metrics["supporting_images_jaccard"]
        lines.append(f"\n## Supporting Images Jaccard: {sj['avg_jaccard']:.4f}")

    # Error details
    lines.append("\n## Error Details\n")
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            if m["errors"]:
                lines.append(f"\n### {m['field']} Errors")
                for err in m["errors"]:
                    lines.append(
                        f"- Row {err['index']}: predicted=`{err['predicted']}`, "
                        f"expected=`{err['expected']}`"
                    )

    return "\n".join(lines)
