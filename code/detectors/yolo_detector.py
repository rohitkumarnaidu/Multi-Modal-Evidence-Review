"""
YOLOv8n Object Detection for deterministic object type prior.

Provides a lightweight, deterministic object detection layer that runs
before the VLM call. YOLO results serve as priors fed into the VLM prompt,
reducing hallucination and improving object type accuracy.

Design:
  - YOLOv8n (nano) — 6.2MB, ~3ms on CPU, no GPU needed
  - Detected COCO classes mapped to our 3 object types + "vehicle" / "other"
  - Highest-confidence detection wins (if multiple candidates)
  - Lazy model loading: model is loaded on first call, then cached
  - If model fails to load, gracefully returns None (VLM falls back)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = None

# COCO class IDs and their mapping to our claim object types
# (YOLOv8n is trained on COCO 80-class)
_COCO_TO_OBJECT: dict[int, str] = {
    2: "car",       # car
    3: "car",       # motorcycle → vehicle
    5: "car",       # bus → vehicle
    7: "car",       # truck → vehicle
    63: "laptop",   # laptop
    39: "package",  # bottle → container/package
    41: "package",  # cup → container/package
    67: "package",  # cell phone → similar to laptop/package category
}

_VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
_CONTAINER_IDS = {24, 25, 26, 27, 28, 29, 30, 31, 39, 40, 41, 44, 46}
_LAPTOP_LIKE_IDS = {63, 64, 67, 73, 74, 75, 76, 77, 78, 79}  # electronics


def _load_model():
    """Lazy-load YOLOv8n model (cached after first call)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        _MODEL = YOLO(model_path, verbose=False)
        logger.info("YOLOv8n model loaded")
    except Exception as e:
        logger.warning(f"Failed to load YOLOv8n: {e}")
        _MODEL = None
    return _MODEL


def detect_objects(image_path: str) -> Optional[dict]:
    """Detect objects in image using YOLOv8n.

    Args:
        image_path: Path to image file.

    Returns:
        Dict with:
            - object_type: str — detected type (car/laptop/package/other/unknown)
            - confidence: float — confidence of the best detection
            - all_objects: list[dict] — all detected objects
        or None if inference fails.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        results = model(image_path, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, "unknown")

                obj_type = _COCO_TO_OBJECT.get(cls_id, None)
                if obj_type is None:
                    if cls_id in _VEHICLE_IDS:
                        obj_type = "car"
                    elif cls_id in _LAPTOP_LIKE_IDS:
                        obj_type = "laptop"
                    elif cls_id in _CONTAINER_IDS:
                        obj_type = "package"
                    else:
                        obj_type = "other"

                detections.append({
                    "label": label,
                    "class_id": cls_id,
                    "object_type": obj_type,
                    "confidence": conf,
                })

        if not detections:
            return {
                "object_type": "unknown",
                "confidence": 0.0,
                "all_objects": [],
            }

        # Pick highest-confidence detection as primary
        best = max(detections, key=lambda d: d["confidence"])
        return {
            "object_type": best["object_type"],
            "confidence": best["confidence"],
            "all_objects": sorted(
                detections, key=lambda d: d["confidence"], reverse=True
            ),
        }
    except Exception as e:
        logger.warning(f"YOLO inference failed for {image_path}: {e}")
        return None
