"""
Gemini API Client with retries, rate limiting, and token tracking.

Two-call design:
  Call 1: LLM text-only — claim extraction from conversation
  Call 2: VLM per-image — vision analysis with structured output

Uses the modern google.genai SDK.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from config import (
    GEMINI_API_KEYS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    RETRY_BASE_DELAY,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
)
from llm.cache import ResponseCache

logger = logging.getLogger(__name__)


class TokenTracker:
    """Track token usage across all API calls."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0
        self.cached_calls = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def record_failure(self):
        self.failed_calls += 1

    def record_cached(self):
        self.cached_calls += 1

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "failed_calls": self.failed_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        # Gemini 2.5 Flash pricing (approx)
        # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
        # Image tokens: ~258 tokens per image included in input
        input_cost = (self.total_input_tokens / 1_000_000) * 0.15
        output_cost = (self.total_output_tokens / 1_000_000) * 0.60
        return round(input_cost + output_cost, 4)


class GeminiClient:
    """Gemini API client with retries, caching, and structured output."""

    def __init__(self):
        if not GEMINI_API_KEYS:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEYS via environment variables."
            )
        from google import genai
        from llm.rate_limiter import KeyRateLimiter

        self._genai = genai
        self.keys = GEMINI_API_KEYS
        self.current_key_idx = 0
        self._client = genai.Client(api_key=self.keys[self.current_key_idx])
        self.cache = ResponseCache()
        self.tracker = TokenTracker()
        # Per-key rate limiter: Gemini free tier = 5 RPM, 20 RPD
        self.rate_limiter = KeyRateLimiter(rpm_limit=5, rpd_limit=20)
        logger.info(
            f"[Gemini] Initialized with {len(self.keys)} keys, "
            f"effective limits: {5 * len(self.keys)} RPM, {20 * len(self.keys)} RPD"
        )

    def _ensure_key_available(self) -> bool:
        """Check rate limits and rotate key if needed. Returns False if all keys exhausted."""
        # Check if current key can handle a request
        can_proceed = self.rate_limiter.wait_and_check(self.current_key_idx)
        if can_proceed:
            return True

        # Current key RPD exhausted — find next available
        next_key = self.rate_limiter.get_next_available_key(
            self.current_key_idx, len(self.keys)
        )
        if next_key is not None:
            old = self.current_key_idx
            self.current_key_idx = next_key
            self._client = self._genai.Client(api_key=self.keys[self.current_key_idx])
            logger.info(
                f"[Gemini] Proactive key rotation: {old} → {self.current_key_idx} "
                f"(key {old} hit RPD limit)"
            )
            # Now wait for RPM on new key
            self.rate_limiter.wait_and_check(self.current_key_idx)
            return True

        # All keys exhausted
        logger.warning("[Gemini] All keys have hit RPD limit")
        return False

    def call_text(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Call 1: Text-only LLM call for claim extraction.

        Returns parsed JSON dict or None on failure.
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                self.tracker.record_cached()
                return cached

        # Make API call with retries
        result = self._call_with_retry(prompt, images=None)
        if result is not None and use_cache:
            self.cache.put(prompt, result)
        return result

    def call_vision(
        self,
        prompt: str,
        image_data: list[dict],  # [{"mime_type": ..., "data": base64_str}]
        image_paths: list[str] | None = None,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Call 2: VLM call with one or more images.

        image_data: list of {"mime_type": str, "data": base64_str}
        Returns parsed JSON dict or None on failure.
        """
        cache_paths = image_paths or []
        if use_cache:
            cached = self.cache.get(prompt, cache_paths)
            if cached is not None:
                self.tracker.record_cached()
                return cached

        result = self._call_with_retry(prompt, images=image_data)
        if result is not None and use_cache:
            self.cache.put(prompt, result, cache_paths)
        return result

    def _call_with_retry(
        self,
        prompt: str,
        images: list[dict] | None = None,
    ) -> Optional[dict]:
        """Execute API call with exponential backoff retry and smart key rotation."""
        import re as _re

        text = ""
        exhausted_in_cycle: set[int] = set()  # Track which keys hit 429 this cycle
        max_attempts = RETRY_MAX_ATTEMPTS

        for attempt in range(max_attempts):
            try:
                # Proactive rate limit check — rotate BEFORE hitting the API
                if not self._ensure_key_available():
                    logger.error("[Gemini] All keys exhausted (RPD). Giving up.")
                    self.tracker.record_failure()
                    return None

                # Build content parts
                contents = []
                if images:
                    for img in images:
                        part = self._genai.types.Part.from_bytes(
                            data=__import__("base64").b64decode(img["data"]),
                            mime_type=img["mime_type"],
                        )
                        contents.append(part)
                contents.append(prompt)

                # Generate with structured config
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=self._genai.types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        response_mime_type="application/json",
                    ),
                )

                # Track tokens
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self.tracker.record(
                        input_tokens=getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        ),
                        output_tokens=getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        ),
                    )
                else:
                    # Estimate if metadata not available
                    self.tracker.record(
                        input_tokens=len(prompt) // 4
                        + (len(images or []) * 258),
                        output_tokens=len(response.text) // 4
                        if response.text
                        else 100,
                    )

                # Parse JSON response
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                parsed = json.loads(text)
                # Success — clear exhausted tracking
                exhausted_in_cycle.clear()
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse error on attempt {attempt + 1}: {e}"
                )
                if attempt == max_attempts - 1:
                    logger.error(
                        f"All retries failed (JSON). Last text: {text[:200]}"
                    )
                    self.tracker.record_failure()
                    return None
                # Retry without backoff for parse errors
                continue

            except Exception as e:
                error_msg = str(e)
                error_lower = error_msg.lower()

                # Check for rate limit / quota exhaustion (429)
                is_quota_error = (
                    "429" in error_lower
                    or "quota" in error_lower
                    or "exhausted" in error_lower
                    or "rate" in error_lower
                )

                if is_quota_error and len(self.keys) > 1:
                    # Mark current key as exhausted in both trackers
                    exhausted_in_cycle.add(self.current_key_idx)
                    self.rate_limiter.mark_key_exhausted(self.current_key_idx)

                    # Find next available key via rate limiter
                    next_key = self.rate_limiter.get_next_available_key(
                        self.current_key_idx, len(self.keys)
                    )
                    if next_key is not None:
                        old_idx = self.current_key_idx
                        self.current_key_idx = next_key
                        logger.warning(
                            f"Quota hit on key {old_idx}. "
                            f"Rotating to key {self.current_key_idx} "
                            f"({len(exhausted_in_cycle)}/{len(self.keys)} exhausted)"
                        )
                        self._client = self._genai.Client(api_key=self.keys[self.current_key_idx])
                        continue  # Retry immediately with new key
                    else:
                        # ALL keys exhausted — parse retryDelay and wait
                        retry_delay = 60.0  # default
                        delay_match = _re.search(r'retryDelay.*?(\d+)', error_msg)
                        if delay_match:
                            retry_delay = float(delay_match.group(1))
                        logger.warning(
                            f"All {len(self.keys)} keys exhausted. "
                            f"Waiting {retry_delay}s before next cycle..."
                        )
                        time.sleep(retry_delay)
                        exhausted_in_cycle.clear()  # Reset for new cycle
                        self.current_key_idx = 0
                        self._client = self._genai.Client(api_key=self.keys[0])
                        continue

                elif is_quota_error and len(self.keys) == 1:
                    # Single key — parse retryDelay and wait
                    retry_delay = 60.0
                    delay_match = _re.search(r'retryDelay.*?(\d+)', error_msg)
                    if delay_match:
                        retry_delay = float(delay_match.group(1))
                    logger.warning(
                        f"Quota hit (single key). Waiting {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue

                # Non-quota error — standard exponential backoff
                delay = min(
                    RETRY_BASE_DELAY * (2 ** attempt),
                    RETRY_MAX_DELAY,
                )
                logger.warning(
                    f"API call attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} retries exhausted for Gemini")
                    self.tracker.record_failure()
                    return None
                time.sleep(delay)

        self.tracker.record_failure()
        return None

