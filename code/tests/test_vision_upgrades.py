from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.vision_engine import _merge_vision_opinions, _needs_second_opinion


def test_second_opinion_only_for_ambiguous_visual_evidence():
    assert _needs_second_opinion({
        "damage_evidence_level": "clear",
        "visible_issue_type": "dent",
        "visible_object_type": "car",
        "confidence": 0.91,
    }) is False
    assert _needs_second_opinion({
        "damage_evidence_level": "partial",
        "visible_issue_type": "dent",
        "visible_object_type": "car",
        "confidence": 0.91,
    }) is True


def test_disagreeing_second_opinion_stays_conservative():
    primary = {
        "visible_object_type": "car",
        "visible_object_part": "door",
        "visible_issue_type": "dent",
        "damage_evidence_level": "clear",
        "confidence": 0.90,
    }
    secondary = {
        "visible_object_type": "car",
        "visible_object_part": "hood",
        "visible_issue_type": "scratch",
        "damage_evidence_level": "clear",
        "confidence": 0.70,
    }
    merged = _merge_vision_opinions(primary, secondary)
    assert merged["visible_object_part"] == "door"
    assert merged["damage_evidence_level"] == "partial"
    assert merged["confidence"] == 0.70


def test_agreeing_second_opinion_increases_confidence():
    result = {
        "visible_object_type": "laptop",
        "visible_object_part": "screen",
        "visible_issue_type": "crack",
        "confidence": 0.70,
    }
    merged = _merge_vision_opinions(result, {**result, "confidence": 0.80})
    assert merged["confidence"] > result["confidence"]
