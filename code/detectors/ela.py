"""
Error Level Analysis (ELA) for JPEG tampering detection.

Detects localized compression differences between the original and
re-saved image. Inconsistent compression quality across regions
suggests digital splicing or local editing.

Pure PIL implementation — no new dependencies.
"""

from __future__ import annotations

import io
import logging
from PIL import Image, ImageChops, ImageStat

logger = logging.getLogger(__name__)


def analyze_ela(image_path: str, quality: int = 75, threshold: float = 15.0) -> dict:
    """Run ELA on a JPEG image.

    Args:
        image_path: Path to image file.
        quality: Re-save JPEG quality (default 75).
        threshold: Mean pixel difference threshold for flagging.

    Returns:
        Dict with:
            - has_anomaly: bool — True if ELA mean diff exceeds threshold
            - ela_mean_diff: float — mean absolute pixel difference
            - ela_max_diff: float — max absolute pixel difference
            - ela_description: str — human-readable summary
    """
    try:
        img = Image.open(image_path)

        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        diff = ImageChops.difference(img, recompressed)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / len(stat.mean) if stat.mean else 0.0
        extrema = diff.getextrema()
        max_diff = max((e[1] for e in extrema if e is not None), default=0.0)

        has_anomaly = mean_diff > threshold

        if has_anomaly:
            desc = (
                f"ELA anomaly detected: mean_diff={mean_diff:.1f}, "
                f"max_diff={max_diff:.1f} — possible image splicing or editing"
            )
        else:
            desc = f"ELA clean: mean_diff={mean_diff:.1f}, max_diff={max_diff:.1f}"

        return {
            "has_anomaly": has_anomaly,
            "ela_mean_diff": round(mean_diff, 2),
            "ela_max_diff": round(max_diff, 2),
            "ela_description": desc,
        }
    except Exception as e:
        logger.warning(f"ELA analysis failed for {image_path}: {e}")
        return {
            "has_anomaly": False,
            "ela_mean_diff": 0.0,
            "ela_max_diff": 0.0,
            "ela_description": f"ELA could not be computed: {e}",
        }
