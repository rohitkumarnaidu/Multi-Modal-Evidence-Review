from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ClaimExtraction, ClaimInput, ClaimOutput, EvidenceRequirement,
    EvidenceSufficiency, FraudSignals, ImageAnalysis, UserHistory,
)
from config import CAR_OBJECT_PARTS, LAPTOP_OBJECT_PARTS, PACKAGE_OBJECT_PARTS


class MockLLMClient:
    def __init__(self):
        self.text_responses: dict[str, dict] = {}
        self.vision_responses: dict[str, dict] = {}
        self.text_calls = 0
        self.vision_calls = 0

    def call_text(self, prompt: str, **kwargs) -> dict | None:
        self.text_calls += 1
        for key, response in self.text_responses.items():
            if key in prompt:
                return response
        return {
            "claimed_issue_type": "dent",
            "claimed_object_part": "door",
            "claimed_severity_hint": "medium",
            "claim_summary": "User claims door dent",
            "has_prompt_injection": False,
            "prompt_injection_detail": "",
            "is_multi_part": False,
            "secondary_parts": [],
        }

    def call_vision(self, prompt: str, image_data: list, **kwargs) -> dict | None:
        self.vision_calls += 1
        for key, response in self.vision_responses.items():
            if key in prompt:
                return response
        return {
            "visible_object_type": "car",
            "visible_object_part": "door",
            "visible_issue_type": "dent",
            "visible_severity": "medium",
            "vehicle_color": "blue",
            "is_blurry": False,
            "is_low_light": False,
            "is_cropped": False,
            "has_wrong_angle": False,
            "has_watermark": False,
            "watermark_text": "",
            "has_text_instruction": False,
            "text_instruction_content": "",
            "is_usable": True,
            "damage_description": "Visible dent on door panel",
            "confidence": 0.92,
        }


def make_claim(
    user_id: str = "test_user",
    image_paths: str = "images/test/case_001/img_1.jpg",
    user_claim: str = "My car door is dented",
    claim_object: str = "car",
) -> ClaimInput:
    return ClaimInput(
        user_id=user_id,
        image_paths=image_paths,
        user_claim=user_claim,
        claim_object=claim_object,
    )


def make_analysis(
    image_id: str = "img_1",
    image_path: str = "images/test/case_001/img_1.jpg",
    visible_object_type: str = "car",
    visible_object_part: str = "door",
    visible_issue_type: str = "dent",
    visible_severity: str = "medium",
    vehicle_color: str = "blue",
    is_blurry: bool = False,
    is_low_light: bool = False,
    is_cropped: bool = False,
    has_wrong_angle: bool = False,
    has_watermark: bool = False,
    watermark_text: str = "",
    has_text_instruction: bool = False,
    text_instruction_content: str = "",
    is_usable: bool = True,
    confidence: float = 0.9,
    damage_description: str = "",
) -> ImageAnalysis:
    return ImageAnalysis(
        image_id=image_id,
        image_path=image_path,
        visible_object_type=visible_object_type,
        visible_object_part=visible_object_part,
        visible_issue_type=visible_issue_type,
        visible_severity=visible_severity,
        vehicle_color=vehicle_color,
        is_blurry=is_blurry,
        is_low_light=is_low_light,
        is_cropped=is_cropped,
        has_wrong_angle=has_wrong_angle,
        has_watermark=has_watermark,
        watermark_text=watermark_text,
        has_text_instruction=has_text_instruction,
        text_instruction_content=text_instruction_content,
        is_usable=is_usable,
        confidence=confidence,
        damage_description=damage_description,
    )
