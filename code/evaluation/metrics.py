"""
Evaluation Metrics for Multi-Modal Evidence Review.

Per-field accuracy metrics with partial credit for ordinal and
hierarchical fields.
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
    assert len(predictions) == len(ground_truth), (
        f"Length mismatch: {len(predictions)} vs {len(ground_truth)}"
    )
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip().lower() == g.strip().lower())
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

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
        "errors": errors[:10],
    }


def partial_credit_severity(
    predictions: list[str],
    ground_truth: list[str],
) -> dict:
    """Compute accuracy with partial credit for ordinal severity.

    Score matrix:
      identical → 1.0
      adjacent (low↔medium, medium↔high) → 0.5
      two apart (low↔high) → 0.25
      extreme (none↔high, none↔low → 0.0)
      unknown mismatch → 0.0
    """
    severity_distance = {
        ("none", "none"): 1.0, ("none", "low"): 0.5, ("none", "medium"): 0.25, ("none", "high"): 0.0, ("none", "unknown"): 0.0,
        ("low", "none"): 0.5, ("low", "low"): 1.0, ("low", "medium"): 0.5, ("low", "high"): 0.25, ("low", "unknown"): 0.0,
        ("medium", "none"): 0.25, ("medium", "low"): 0.5, ("medium", "medium"): 1.0, ("medium", "high"): 0.5, ("medium", "unknown"): 0.0,
        ("high", "none"): 0.0, ("high", "low"): 0.25, ("high", "medium"): 0.5, ("high", "high"): 1.0, ("high", "unknown"): 0.0,
        ("unknown", "none"): 0.0, ("unknown", "low"): 0.0, ("unknown", "medium"): 0.0, ("unknown", "high"): 0.0, ("unknown", "unknown"): 1.0,
    }

    scores = []
    per_row = []
    for i, (p, g) in enumerate(zip(predictions, ground_truth)):
        p = p.strip().lower()
        g = g.strip().lower()
        score = severity_distance.get((p, g), 0.0)
        scores.append(score)
        per_row.append({
            "index": i,
            "predicted": p,
            "expected": g,
            "score": score,
        })

    avg = sum(scores) / max(1, len(scores))
    return {
        "partial_credit_accuracy": round(avg, 4),
        "exact_match_accuracy": round(sum(1 for s in scores if s == 1.0) / max(1, len(scores)), 4),
        "average_score": round(avg, 4),
        "per_row": [r for r in per_row if r["score"] < 1.0][:10],
    }


def partial_credit_object_part(
    predictions: list[str],
    ground_truth: list[str],
) -> dict:
    """Compute accuracy with partial credit for object_part.

    Same object type, same category → 0.5
    Same part → 1.0
    Different → 0.0
    """
    car_exterior = {"front_bumper", "rear_bumper", "door", "hood", "windshield", "fender", "body"}
    car_lighting = {"headlight", "taillight", "side_mirror"}
    car_quarter = {"quarter_panel"}
    car_parts = car_exterior | car_lighting | car_quarter

    laptop_display = {"screen", "lid"}
    laptop_input = {"keyboard", "trackpad"}
    laptop_structure = {"hinge", "corner", "port", "base", "body"}
    laptop_parts = laptop_display | laptop_input | laptop_structure

    package_exterior = {"box", "package_corner", "package_side"}
    package_closure = {"seal", "label"}
    package_inner = {"contents", "item"}
    package_parts = package_exterior | package_closure | package_inner

    part_categories = {
        "car": [car_exterior, car_lighting, car_quarter],
        "laptop": [laptop_display, laptop_input, laptop_structure],
        "package": [package_exterior, package_closure, package_inner],
    }

    def same_category(a: str, b: str) -> bool:
        for obj, categories in part_categories.items():
            for cat in categories:
                if a in cat and b in cat:
                    return True
        return False

    scores = []
    per_row = []
    for i, (p, g) in enumerate(zip(predictions, ground_truth)):
        p = p.strip().lower()
        g = g.strip().lower()

        if p == g:
            score = 1.0
        elif p == "unknown" or g == "unknown":
            score = 0.0
        elif same_category(p, g):
            score = 0.5
        else:
            score = 0.0

        scores.append(score)
        per_row.append({
            "index": i,
            "predicted": p,
            "expected": g,
            "score": score,
        })

    avg = sum(scores) / max(1, len(scores))
    return {
        "partial_credit_accuracy": round(avg, 4),
        "exact_match_accuracy": round(sum(1 for s in scores if s == 1.0) / max(1, len(scores)), 4),
        "per_row": [r for r in per_row if r["score"] < 1.0][:10],
    }


def risk_flags_f1(
    pred_flags: list[str],
    true_flags: list[str],
) -> dict:
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
        "per_row": [r for r in per_row if r["f1"] < 1.0][:10],
    }


def confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    field_name: str = "",
) -> dict:
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
    results = {}

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

    # Partial credit for severity (ordinal)
    pred_sev = [str(p.get("severity", "")).strip().lower() for p in predictions]
    true_sev = [str(g.get("severity", "")).strip().lower() for g in ground_truth]
    results["severity_partial_credit"] = partial_credit_severity(pred_sev, true_sev)

    # Partial credit for object_part (hierarchical)
    pred_part = [str(p.get("object_part", "")).strip().lower() for p in predictions]
    true_part = [str(g.get("object_part", "")).strip().lower() for g in ground_truth]
    results["object_part_partial_credit"] = partial_credit_object_part(pred_part, true_part)

    # Risk flags
    pred_flags = [str(p.get("risk_flags", "none")) for p in predictions]
    true_flags = [str(g.get("risk_flags", "none")) for g in ground_truth]
    results["risk_flags_f1"] = risk_flags_f1(pred_flags, true_flags)

    # Supporting image IDs
    pred_ids = [str(p.get("supporting_image_ids", "none")) for p in predictions]
    true_ids = [str(g.get("supporting_image_ids", "none")) for g in ground_truth]
    results["supporting_images_jaccard"] = supporting_images_jaccard(pred_ids, true_ids)

    return results


def _parse_flags(flags_str: str) -> set[str]:
    flags_str = flags_str.strip().lower()
    if flags_str in ("none", ""):
        return set()
    return {f.strip() for f in flags_str.split(";") if f.strip() and f.strip() != "none"}


def format_report(metrics: dict) -> str:
    lines = ["# Evaluation Results\n"]

    lines.append("## Accuracy Summary\n")
    lines.append("| Field | Accuracy | Correct / Total |")
    lines.append("|-------|----------|-----------------|")
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            lines.append(
                f"| {m['field']} | {m['accuracy']:.1%} | {m['correct']}/{m['total']} |"
            )

    # Partial credit
    if "severity_partial_credit" in metrics:
        sc = metrics["severity_partial_credit"]
        lines.append(f"\n## Severity Partial Credit: {sc['partial_credit_accuracy']:.4f} "
                      f"(exact: {sc['exact_match_accuracy']:.4f})")

    if "object_part_partial_credit" in metrics:
        oc = metrics["object_part_partial_credit"]
        lines.append(f"\n## Object Part Partial Credit: {oc['partial_credit_accuracy']:.4f} "
                      f"(exact: {oc['exact_match_accuracy']:.4f})")

    if "risk_flags_f1" in metrics:
        rf = metrics["risk_flags_f1"]
        lines.append(f"\n## Risk Flags F1")
        lines.append(f"- Micro F1: {rf['micro_f1']:.4f}")
        lines.append(f"- Macro F1: {rf['macro_f1']:.4f}")

    if "supporting_images_jaccard" in metrics:
        sj = metrics["supporting_images_jaccard"]
        lines.append(f"\n## Supporting Images Jaccard: {sj['avg_jaccard']:.4f}")

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
