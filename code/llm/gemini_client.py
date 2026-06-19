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

        self._genai = genai
        self.keys = GEMINI_API_KEYS
        self.current_key_idx = 0
        self._client = genai.Client(api_key=self.keys[self.current_key_idx])
        self.cache = ResponseCache()
        self.tracker = TokenTracker()
        self._last_call_time = 0.0
        self._min_call_interval = 0.5  # seconds between calls

    def _rate_limit_wait(self):
        """Enforce minimum interval between API calls."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_call_interval:
            wait = self._min_call_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait:.2f}s")
            time.sleep(wait)
        self._last_call_time = time.time()

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
        """Execute API call with exponential backoff retry."""
        text = ""
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                self._rate_limit_wait()

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
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse error on attempt {attempt + 1}: {e}"
                )
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    logger.error(
                        f"All retries failed (JSON). Last text: {text[:200]}"
                    )
                    self.tracker.record_failure()
                    return None
                # Retry without backoff for parse errors
                continue

            except Exception as e:
                error_msg = str(e).lower()
                # Check for rate limit or quota exhaustion (429)
                if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    if len(self.keys) > 1:
                        # Rotate key
                        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                        logger.warning(
                            f"Quota limit hit on key index {(self.current_key_idx - 1) % len(self.keys)}. "
                            f"Rotating to key index {self.current_key_idx} ({len(self.keys)} total keys)."
                        )
                        self._client = self._genai.Client(api_key=self.keys[self.current_key_idx])
                        # Reset attempt counter to allow full retries on new key
                        attempt -= 1 
                        continue
                    else:
                        logger.warning(f"Quota limit hit but only 1 key configured. Waiting...")

                delay = min(
                    RETRY_BASE_DELAY * (2**attempt),
                    RETRY_MAX_DELAY,
                )
                logger.warning(
                    f"API call attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    logger.error(f"All {RETRY_MAX_ATTEMPTS} retries exhausted")
                    self.tracker.record_failure()
                    return None
                time.sleep(delay)

        return None
