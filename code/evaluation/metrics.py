"""
Evaluation Metrics for Multi-Modal Evidence Review.

Per-field accuracy metrics with partial credit for ordinal and
hierarchical fields. Generates HTML reports with confusion matrices.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
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

    pred_sev = [str(p.get("severity", "")).strip().lower() for p in predictions]
    true_sev = [str(g.get("severity", "")).strip().lower() for g in ground_truth]
    results["severity_partial_credit"] = partial_credit_severity(pred_sev, true_sev)

    pred_part = [str(p.get("object_part", "")).strip().lower() for p in predictions]
    true_part = [str(g.get("object_part", "")).strip().lower() for g in ground_truth]
    results["object_part_partial_credit"] = partial_credit_object_part(pred_part, true_part)

    pred_flags = [str(p.get("risk_flags", "none")) for p in predictions]
    true_flags = [str(g.get("risk_flags", "none")) for g in ground_truth]
    results["risk_flags_f1"] = risk_flags_f1(pred_flags, true_flags)

    pred_ids = [str(p.get("supporting_image_ids", "none")) for p in predictions]
    true_ids = [str(g.get("supporting_image_ids", "none")) for g in ground_truth]
    results["supporting_images_jaccard"] = supporting_images_jaccard(pred_ids, true_ids)

    return results


def _parse_flags(flags_str: str) -> set[str]:
    flags_str = flags_str.strip().lower()
    if flags_str in ("none", ""):
        return set()
    return {f.strip() for f in flags_str.split(";") if f.strip() and f.strip() != "none"}


def generate_html_report(metrics: dict, history: list[dict] | None = None) -> str:
    """Generate a standalone HTML report with styled tables and confusion matrices."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    acc_rows = ""
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            pct = m['accuracy'] * 100
            bar_color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 50 else "#ef4444"
            acc_rows += f"""
            <tr>
                <td>{m['field']}</td>
                <td>{m['correct']}/{m['total']}</td>
                <td><div class="bar" style="width:{pct}%;background:{bar_color}"></div></td>
                <td style="font-weight:bold;color:{bar_color}">{m['accuracy']:.1%}</td>
            </tr>"""

    def matrix_to_html(cm: dict) -> str:
        labels = cm['labels']
        matrix = cm['matrix']
        rows_html = "<tr><th>True \\ Pred</th>"
        for label in labels:
            rows_html += f"<th>{label}</th>"
        rows_html += "</tr>"
        for true_label in labels:
            row = matrix.get(true_label, {})
            rows_html += f"<tr><td><strong>{true_label}</strong></td>"
            for pred_label in labels:
                count = row.get(pred_label, 0)
                cls = "cell-high" if count > 2 else "cell-mid" if count > 0 else ""
                rows_html += f'<td class="{cls}">{count}</td>'
            rows_html += "</tr>"
        return rows_html

    cm_sections = ""
    for key in sorted(metrics.keys()):
        if key.endswith("_confusion"):
            cm = metrics[key]
            cm_sections += f"""
            <div class="section">
                <h3>{cm['field']} Confusion Matrix</h3>
                <table class="matrix">{matrix_to_html(cm)}</table>
            </div>"""

    error_rows = ""
    for key in sorted(metrics.keys()):
        if key.endswith("_accuracy"):
            m = metrics[key]
            for err in m.get("errors", []):
                error_rows += f"""
                <tr>
                    <td>{m['field']}</td>
                    <td>{err['index']}</td>
                    <td style="color:#ef4444">{err['predicted']}</td>
                    <td style="color:#22c55e">{err['expected']}</td>
                </tr>"""

    partial_html = ""
    if "severity_partial_credit" in metrics:
        sc = metrics["severity_partial_credit"]
        partial_html += f"""
        <tr>
            <td>Severity (partial credit)</td>
            <td colspan="2">{sc['partial_credit_accuracy']:.4f}</td>
            <td>(exact: {sc['exact_match_accuracy']:.4f})</td>
        </tr>"""
    if "object_part_partial_credit" in metrics:
        oc = metrics["object_part_partial_credit"]
        partial_html += f"""
        <tr>
            <td>Object Part (partial credit)</td>
            <td colspan="2">{oc['partial_credit_accuracy']:.4f}</td>
            <td>(exact: {oc['exact_match_accuracy']:.4f})</td>
        </tr>"""

    rf = metrics.get("risk_flags_f1", {})
    sj = metrics.get("supporting_images_jaccard", {})

    history_html = ""
    if history and len(history) > 1:
        history_html = '<div class="section"><h2>Trend (Last 5 Runs)</h2><table><tr><th>Run</th>'
        fields = ["claim_status", "issue_type", "object_part", "severity"]
        for f in fields:
            history_html += f"<th>{f}</th>"
        history_html += "</tr>"
        for i, h in enumerate(history[-5:]):
            history_html += f"<tr><td>{i+1}</td>"
            for f in fields:
                acc_key = f"{f}_accuracy"
                if acc_key in h:
                    pct = h[acc_key]['accuracy'] * 100
                    color = "#22c55e" if pct >= 80 else "#eab308"
                    history_html += f'<td style="color:{color}">{h[acc_key]["accuracy"]:.1%}</td>'
                else:
                    history_html += "<td>N/A</td>"
            history_html += "</tr>"
        history_html += "</table></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — Multi-Modal Evidence Review</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #f8fafc; }}
h2 {{ font-size: 1.3rem; margin: 1.5rem 0 1rem; color: #94a3b8; border-bottom: 1px solid #1e293b; padding-bottom: 0.3rem; }}
h3 {{ font-size: 1rem; margin: 1rem 0 0.5rem; color: #cbd5e1; }}
.timestamp {{ color: #64748b; margin-bottom: 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #1e293b; font-size: 0.9rem; }}
th {{ background: #1e293b; color: #94a3b8; font-weight: 600; position: sticky; top: 0; }}
tr:hover {{ background: #1e293b; }}
.bar-container {{ width: 100%; background: #1e293b; border-radius: 4px; height: 20px; }}
.bar {{ height: 20px; border-radius: 4px; min-width: 4px; }}
.section {{ margin: 1rem 0; background: #1a2332; padding: 1rem; border-radius: 8px; }}
.matrix {{ font-size: 0.8rem; }}
.matrix th, .matrix td {{ text-align: center; min-width: 60px; }}
.cell-high {{ background: #166534 !important; color: #bbf7d0; }}
.cell-mid {{ background: #713f12 !important; color: #fde68a; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
.card {{ background: #1a2332; padding: 1rem; border-radius: 8px; text-align: center; }}
.card-value {{ font-size: 2rem; font-weight: bold; }}
.card-label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }}
footer {{ margin-top: 2rem; text-align: center; color: #475569; font-size: 0.8rem; }}
a {{ color: #60a5fa; }}
</style>
</head>
<body>
<div class="container">
<h1>Evaluation Report</h1>
<p class="timestamp">Generated: {now}</p>

<div class="metrics-grid">
    <div class="card">
        <div class="card-value" style="color:#22c55e">{sum(m['correct'] for k, m in metrics.items() if k.endswith('_accuracy'))}/{sum(m['total'] for k, m in metrics.items() if k.endswith('_accuracy'))}</div>
        <div class="card-label">Total Correct</div>
    </div>
    <div class="card">
        <div class="card-value" style="color:#60a5fa">{rf.get('micro_f1', 0):.3f}</div>
        <div class="card-label">Risk Flags F1</div>
    </div>
    <div class="card">
        <div class="card-value" style="color:#a78bfa">{sj.get('avg_jaccard', 0):.3f}</div>
        <div class="card-label">Supporting Images Jaccard</div>
    </div>
</div>

<div class="section">
<h2>Accuracy Summary</h2>
<table>
<tr><th>Field</th><th>Correct/Total</th><th>Accuracy</th><th>%</th></tr>
{acc_rows}
</table>
</div>

{history_html}

<div class="section">
<h2>Partial Credit</h2>
<table>
<tr><th>Field</th><th colspan="3">Score</th></tr>
{partial_html}
</table>
</div>

{cm_sections}

<div class="section">
<h2>Error Details</h2>
<table>
<tr><th>Field</th><th>Row</th><th>Predicted</th><th>Expected</th></tr>
{error_rows if error_rows else '<tr><td colspan="4" style="text-align:center;color:#64748b">No errors</td></tr>'}
</table>
</div>

<footer>Multi-Modal Evidence Review — HackerRank Orchestrate June 2026</footer>
</div>
</body>
</html>"""


def format_report(metrics: dict) -> str:
    """Legacy Markdown report format."""
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
