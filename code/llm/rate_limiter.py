"""
Rate Limiter — Per-key RPM and RPD tracking.

Tracks request timestamps per API key to enforce:
  - RPM (Requests Per Minute): sliding window, sleeps when limit is near
  - RPD (Requests Per Day): counter per key, signals when key is exhausted

Usage:
    limiter = KeyRateLimiter(rpm_limit=5, rpd_limit=20, num_keys=3)
    
    # Before each request:
    can_proceed = limiter.wait_and_check(key_index=0)
    if not can_proceed:
        # Key exhausted for the day, rotate to next key
        ...
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class KeyRateLimiter:
    """Per-key RPM and RPD rate limiter with sliding window."""

    def __init__(self, rpm_limit: int = 5, rpd_limit: int = 20):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit

        # Per-key tracking
        self._timestamps: dict[int, list[float]] = defaultdict(list)
        self._daily_counts: dict[int, int] = defaultdict(int)
        self._day_start: float = time.time()

    def _reset_day_if_needed(self):
        """Reset daily counters if a new day has started."""
        elapsed = time.time() - self._day_start
        if elapsed >= 86400:  # 24 hours
            logger.info("[RateLimiter] Day reset — clearing daily counters")
            self._daily_counts.clear()
            self._day_start = time.time()

    def wait_and_check(self, key_index: int) -> bool:
        """Wait for RPM limit and check RPD.

        Returns:
            True if this key can be used (RPD not exhausted).
            False if this key has hit its daily limit — caller should rotate.
        """
        self._reset_day_if_needed()

        # ── Check RPD first ──────────────────────────────────────────────
        if self._daily_counts[key_index] >= self.rpd_limit:
            logger.debug(
                f"[RateLimiter] Key {key_index} hit RPD limit "
                f"({self._daily_counts[key_index]}/{self.rpd_limit})"
            )
            return False  # Signal to rotate

        # ── Enforce RPM via sliding window ───────────────────────────────
        now = time.time()
        cutoff = now - 60.0  # 1-minute window

        # Clean old timestamps
        self._timestamps[key_index] = [
            t for t in self._timestamps[key_index] if t > cutoff
        ]

        recent_count = len(self._timestamps[key_index])

        if recent_count >= self.rpm_limit:
            # We're at the RPM limit — calculate how long to wait
            oldest_in_window = self._timestamps[key_index][0]
            wait_time = oldest_in_window - cutoff + 1.0  # +1s safety margin
            wait_time = max(wait_time, 1.0)
            logger.info(
                f"[RateLimiter] Key {key_index} at RPM limit "
                f"({recent_count}/{self.rpm_limit}). "
                f"Waiting {wait_time:.1f}s"
            )
            time.sleep(wait_time)

        # Record this request
        self._timestamps[key_index].append(time.time())
        self._daily_counts[key_index] += 1

        logger.debug(
            f"[RateLimiter] Key {key_index}: "
            f"RPM={len(self._timestamps[key_index])}/{self.rpm_limit}, "
            f"RPD={self._daily_counts[key_index]}/{self.rpd_limit}"
        )
        return True

    def mark_key_exhausted(self, key_index: int):
        """Manually mark a key as exhausted (e.g., after a 429 RPD error)."""
        self._daily_counts[key_index] = self.rpd_limit
        logger.info(
            f"[RateLimiter] Key {key_index} manually marked as exhausted"
        )

    def get_next_available_key(self, current: int, total_keys: int) -> int | None:
        """Find the next key that hasn't hit its RPD limit.

        Returns key index or None if all keys are exhausted.
        """
        self._reset_day_if_needed()
        for offset in range(1, total_keys + 1):
            candidate = (current + offset) % total_keys
            if self._daily_counts[candidate] < self.rpd_limit:
                return candidate
        return None  # All keys exhausted

    def all_keys_exhausted(self, total_keys: int) -> bool:
        """Check if every key has hit its daily limit."""
        self._reset_day_if_needed()
        return all(
            self._daily_counts[i] >= self.rpd_limit
            for i in range(total_keys)
        )

    @property
    def stats(self) -> dict:
        return {
            "rpm_limit": self.rpm_limit,
            "rpd_limit": self.rpd_limit,
            "daily_counts": dict(self._daily_counts),
            "active_windows": {
                k: len(v) for k, v in self._timestamps.items()
            },
        }


class SimpleRateLimiter:
    """Simple RPM limiter for providers like Groq/OpenRouter/NVIDIA.

    No multi-key support — just enforces requests per minute.
    """

    def __init__(self, rpm_limit: int = 25):
        self.rpm_limit = rpm_limit
        self._timestamps: list[float] = []
        self.total_requests = 0

    def wait_if_needed(self):
        """Block until it's safe to make a request."""
        now = time.time()
        cutoff = now - 60.0

        # Clean old timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.rpm_limit:
            oldest = self._timestamps[0]
            wait_time = oldest - cutoff + 1.5  # +1.5s safety margin
            wait_time = max(wait_time, 1.0)
            logger.info(
                f"[RateLimiter] At RPM limit ({len(self._timestamps)}/{self.rpm_limit}). "
                f"Waiting {wait_time:.1f}s"
            )
            time.sleep(wait_time)

        self._timestamps.append(time.time())
        self.total_requests += 1
