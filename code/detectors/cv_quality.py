import cv2
import numpy as np

BLUR_THRESHOLD = 100
GLARE_PIXEL_THRESHOLD = 250
GLARE_RATIO_THRESHOLD = 0.15
DARK_PIXEL_THRESHOLD = 50
DARK_RATIO_THRESHOLD = 0.30
OBSTRUCTION_EDGE_THRESHOLD = 0.01


def _is_blurry(image_path: str, threshold: int = BLUR_THRESHOLD) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold


def _has_glare(image_path: str, pixel_threshold: int = GLARE_PIXEL_THRESHOLD,
               ratio_threshold: float = GLARE_RATIO_THRESHOLD) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_pixels = float((gray > pixel_threshold).sum())
    total_pixels = float(gray.shape[0] * gray.shape[1])
    return (bright_pixels / total_pixels) > ratio_threshold


def _is_dark(image_path: str, pixel_threshold: int = DARK_PIXEL_THRESHOLD,
             ratio_threshold: float = DARK_RATIO_THRESHOLD) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    dark_pixels = float((img < pixel_threshold).sum())
    total_pixels = float(img.shape[0] * img.shape[1])
    return (dark_pixels / total_pixels) > ratio_threshold


def _is_obstructed(image_path: str, edge_threshold: float = OBSTRUCTION_EDGE_THRESHOLD) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(edges.sum()) / float(img.shape[0] * img.shape[1] * 255)
    return edge_density < edge_threshold


def _get_aspect_ratio(image_path: str) -> float | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return max(w, h) / max(min(w, h), 1)


def _has_wrong_angle(image_path: str) -> bool:
    ratio = _get_aspect_ratio(image_path)
    if ratio is None:
        return False
    return ratio > 3.0 or ratio < 0.33


def analyze_image_quality(image_path: str) -> dict:
    result = {
        "is_blurry": _is_blurry(image_path),
        "is_low_light": _is_dark(image_path),
        "has_glare": _has_glare(image_path),
        "is_obstructed": _is_obstructed(image_path),
        "has_wrong_angle": _has_wrong_angle(image_path),
    }
    # Combine low_light and glare into is_low_light
    result["is_low_light"] = result["is_low_light"] or result.pop("has_glare")
    result.pop("has_glare", None)
    # Map is_obstructed → is_cropped
    result["is_cropped"] = result.pop("is_obstructed", False)
    return result
