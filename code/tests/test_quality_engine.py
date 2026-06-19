from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.quality_engine import assess_image_quality
from models import ImageAnalysis


class TestImageQuality:
    def test_valid_usable_image(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True),
        ]
        result = assess_image_quality(analyses)
        assert result["valid_image"] is True
        assert result["quality_flags"] == []

    def test_empty_analyses(self):
        result = assess_image_quality([])
        assert result["valid_image"] is False

    def test_all_watermarked(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          has_watermark=True),
            ImageAnalysis(image_id="img_2", image_path="b.jpg", is_usable=True,
                          has_watermark=True),
        ]
        result = assess_image_quality(analyses)
        assert result["valid_image"] is False
        assert "non_original_image" in result["quality_flags"]

    def test_one_watermark_one_clean(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          has_watermark=True),
            ImageAnalysis(image_id="img_2", image_path="b.jpg", is_usable=True,
                          has_watermark=False),
        ]
        result = assess_image_quality(analyses)
        assert result["valid_image"] is True
        assert "non_original_image" not in result["quality_flags"]

    def test_all_watermark_all_unusable(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=False,
                          has_watermark=True),
        ]
        result = assess_image_quality(analyses)
        assert result["valid_image"] is False

    def test_blurry_flag(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          is_blurry=True),
        ]
        result = assess_image_quality(analyses)
        assert "blurry_image" in result["quality_flags"]
        assert result["valid_image"] is True

    def test_cropped_flag(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          is_cropped=True),
        ]
        result = assess_image_quality(analyses)
        assert "cropped_or_obstructed" in result["quality_flags"]

    def test_low_light_flag(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          is_low_light=True),
        ]
        result = assess_image_quality(analyses)
        assert "low_light_or_glare" in result["quality_flags"]

    def test_wrong_angle_flag(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          has_wrong_angle=True),
        ]
        result = assess_image_quality(analyses)
        assert "wrong_angle" in result["quality_flags"]

    def test_mixed_quality_flags(self):
        analyses = [
            ImageAnalysis(image_id="img_1", image_path="a.jpg", is_usable=True,
                          is_blurry=True, is_cropped=True),
        ]
        result = assess_image_quality(analyses)
        assert "blurry_image" in result["quality_flags"]
        assert "cropped_or_obstructed" in result["quality_flags"]
        assert result["valid_image"] is True
