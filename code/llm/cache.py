"""
File-based JSON cache for API responses.

Cache key = hash of (prompt_text + sorted image_paths).
Stores responses as JSON files in .cache/ directory.
Avoids redundant API calls on re-runs.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """File-based cache for API responses."""

    def __init__(self, cache_dir: Path | None = None):
        from config import CACHE_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, image_paths: list[str] | None = None) -> str:
        """Generate a deterministic cache key."""
        parts = [prompt]
        if image_paths:
            parts.extend(sorted(image_paths))
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def get(self, prompt: str, image_paths: list[str] | None = None) -> Optional[dict]:
        """Retrieve cached response if exists."""
        key = self._make_key(prompt, image_paths)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.hits += 1
                logger.debug(f"Cache HIT: {key}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Cache read error for {key}: {e}")
        self.misses += 1
        logger.debug(f"Cache MISS: {key}")
        return None

    def put(self, prompt: str, response: dict, image_paths: list[str] | None = None) -> None:
        """Store response in cache."""
        key = self._make_key(prompt, image_paths)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache PUT: {key}")
        except OSError as e:
            logger.warning(f"Cache write error for {key}: {e}")

    def clear(self) -> None:
        """Clear all cached responses."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached responses")

    @property
    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / max(1, self.hits + self.misses)}
