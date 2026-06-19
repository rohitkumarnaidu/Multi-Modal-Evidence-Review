"""
Semantic Part Matcher — embedding-based part name matching.

Replaces hardcoded alias tables and fuzzy string matching with
sentence transformer embeddings. Handles synonyms ("bonnet"→"hood",
"windscreen"→"windshield", "display"→"screen") and multilingual
part descriptions.

Design:
  - all-MiniLM-L6-v2 model (~80MB, cached after first load)
  - Pre-computed embeddings for all known parts across car/laptop/package
  - Cosine similarity threshold (default 0.6) for fuzzy matching
  - Lazy model loading, cached after first call
"""

from __future__ import annotations

import logging
from typing import Optional

from config import OBJECT_PARTS_BY_TYPE

logger = logging.getLogger(__name__)

_MODEL = None
_PART_EMBEDDINGS: dict[str, list[float]] = {}
_EMBEDDING_DIM = 384


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=None,
        )
        logger.info("SentenceTransformer model loaded (all-MiniLM-L6-v2)")
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer: {e}")
        _MODEL = None
    return _MODEL


def _build_part_embeddings():
    global _PART_EMBEDDINGS
    if _PART_EMBEDDINGS:
        return _PART_EMBEDDINGS

    model = _load_model()
    if model is None:
        return {}

    import numpy as np
    all_parts = set()
    for parts in OBJECT_PARTS_BY_TYPE.values():
        all_parts.update(parts)

    for part in sorted(all_parts):
        label = part.replace("_", " ")
        emb = model.encode(label, normalize_embeddings=True)
        _PART_EMBEDDINGS[part] = emb.tolist()

    logger.info(f"Built embeddings for {len(_PART_EMBEDDINGS)} parts")
    return _PART_EMBEDDINGS


def match_part(
    detected_part: str,
    allowed_parts: set[str],
    threshold: float = 0.6,
) -> tuple[str, float]:
    """Match a detected part name to the closest allowed part.

    Args:
        detected_part: Part name from VLM (may be misspelled, synonym, etc.)
        allowed_parts: Set of allowed part names to match against.
        threshold: Minimum cosine similarity for a match.

    Returns:
        Tuple of (best_matching_part, similarity_score).
        If no match above threshold, returns (detected_part, 0.0).
    """
    if detected_part in allowed_parts:
        return detected_part, 1.0

    model = _load_model()
    if model is None:
        return detected_part, 0.0

    _build_part_embeddings()

    import numpy as np

    label = detected_part.replace("_", " ")
    query_emb = model.encode(label, normalize_embeddings=True)

    best_part = detected_part
    best_score = 0.0

    for part in allowed_parts:
        part_emb = np.array(_PART_EMBEDDINGS.get(part, []), dtype=np.float32)
        if part_emb.size == 0:
            continue
        score = float(np.dot(query_emb, part_emb))
        if score > best_score:
            best_score = score
            best_part = part

    if best_score >= threshold:
        return best_part, round(best_score, 4)
    return detected_part, 0.0
