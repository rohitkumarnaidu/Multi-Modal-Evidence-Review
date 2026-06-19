"""
OpenAI-Compatible API Client for Groq, OpenRouter, and NVIDIA.

All three providers expose OpenAI-compatible chat completion endpoints.
This single client handles all of them by swapping base_url + api_key + model.

Supports:
  - Text-only calls (claim extraction)
  - Vision calls with base64 images (per-image analysis)
  - JSON-mode output
  - Retries with exponential backoff
  - Token tracking
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from config import (
    RETRY_BASE_DELAY,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
)
from llm.cache import ResponseCache

logger = logging.getLogger(__name__)


class OpenAICompatClient:
    """Client for any OpenAI-compatible API (Groq, OpenRouter, NVIDIA)."""

    # Default RPM limits per provider (conservative for free tiers)
    DEFAULT_RPM = {"groq": 6, "openrouter": 20, "nvidia": 40}
    # Providers that support response_format={"type": "json_object"}
    JSON_FORMAT_PROVIDERS = {"groq", "openrouter"}

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        text_model: str,
        vision_model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache: ResponseCache | None = None,
    ):
        if not api_key:
            raise ValueError(f"{provider_name} API key not set.")

        from openai import OpenAI
        from llm.rate_limiter import SimpleRateLimiter

        self.provider_name = provider_name
        # Disable OpenAI SDK auto-retries — we handle retries ourselves
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self.text_model = text_model
        self.vision_model = vision_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache or ResponseCache()

        # RPM limiter
        rpm = self.DEFAULT_RPM.get(provider_name.lower(), 20)
        self.rate_limiter = SimpleRateLimiter(rpm_limit=rpm)

        # Simple token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0
        self.cached_calls = 0

        logger.info(
            f"[{provider_name}] Initialized: text={text_model}, "
            f"vision={vision_model}, rpm_limit={rpm}"
        )

    def call_text(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Text-only LLM call for claim extraction."""
        if use_cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                self.cached_calls += 1
                return cached

        result = self._call_with_retry(prompt, images=None, use_vision=False)
        if result is not None and use_cache:
            self.cache.put(prompt, result)
        return result

    def call_vision(
        self,
        prompt: str,
        image_data: list[dict],
        image_paths: list[str] | None = None,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """VLM call with one or more images."""
        cache_paths = image_paths or []
        if use_cache:
            cached = self.cache.get(prompt, cache_paths)
            if cached is not None:
                self.cached_calls += 1
                return cached

        result = self._call_with_retry(prompt, images=image_data, use_vision=True)
        if result is not None and use_cache:
            self.cache.put(prompt, result, cache_paths)
        return result

    def _call_with_retry(
        self,
        prompt: str,
        images: list[dict] | None = None,
        use_vision: bool = False,
    ) -> Optional[dict]:
        """Execute API call with exponential backoff retry."""
        text = ""
        model = self.vision_model if use_vision else self.text_model

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                self.rate_limiter.wait_if_needed()

                # Build messages
                if images:
                    # Vision: multimodal content array
                    content_parts = []
                    for img in images:
                        img_data = img["data"]
                        img_mime = img["mime_type"]
                        # Groq has a 4MB request limit — compress large images
                        if self.provider_name.lower() == "groq":
                            img_data, img_mime = self._compress_image(
                                img_data, max_size_bytes=3 * 1024 * 1024
                            )
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img_mime};base64,{img_data}"
                            },
                        })
                    content_parts.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content_parts}]
                else:
                    # Text only
                    messages = [{"role": "user", "content": prompt}]

                # Make API call
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }

                # JSON mode — only for providers that support it
                if self.provider_name.lower() in self.JSON_FORMAT_PROVIDERS:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**kwargs)

                # Track tokens
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens or 0
                    self.total_output_tokens += response.usage.completion_tokens or 0
                self.total_calls += 1

                # Parse response
                text = response.choices[0].message.content or ""
                text = text.strip()
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
                    f"[{self.provider_name}] JSON parse error attempt "
                    f"{attempt + 1}: {e}"
                )
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    logger.error(
                        f"[{self.provider_name}] All retries failed (JSON). "
                        f"Last text: {text[:200]}"
                    )
                    self.failed_calls += 1
                    return None
                continue

            except Exception as e:
                error_msg = str(e)
                error_lower = error_msg.lower()

                # 400 client errors (invalid image, bad request) — don't retry
                is_client_error = "400" in error_lower and (
                    "invalid" in error_lower
                    or "bad request" in error_lower
                )
                if is_client_error:
                    logger.error(
                        f"[{self.provider_name}] Client error (no retry): {error_msg[:200]}"
                    )
                    self.failed_calls += 1
                    return None

                is_quota = (
                    "429" in error_lower
                    or "quota" in error_lower
                    or "rate" in error_lower
                    or "limit" in error_lower
                )

                delay = min(
                    RETRY_BASE_DELAY * (2 ** attempt),
                    RETRY_MAX_DELAY,
                )

                if is_quota:
                    # For quota errors, wait longer
                    delay = max(delay, 30.0)

                logger.warning(
                    f"[{self.provider_name}] Attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS} "
                    f"failed: {e}. Retrying in {delay:.1f}s"
                )

                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    logger.error(
                        f"[{self.provider_name}] All {RETRY_MAX_ATTEMPTS} retries exhausted"
                    )
                    self.failed_calls += 1
                    return None
                time.sleep(delay)

        self.failed_calls += 1
        return None

    @staticmethod
    def _compress_image(
        b64_data: str, max_size_bytes: int = 3 * 1024 * 1024
    ) -> tuple[str, str]:
        """Compress base64 image to fit within size limit.

        Returns (compressed_b64, mime_type).
        """
        import base64
        import io
        from PIL import Image

        raw = base64.b64decode(b64_data)

        # If already small enough, return as-is
        if len(raw) <= max_size_bytes:
            return b64_data, "image/jpeg"

        img = Image.open(io.BytesIO(raw))
        # Resize if very large
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        # Convert to RGB (strip alpha) and compress as JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        quality = 85
        img.save(buf, format="JPEG", quality=quality)

        # If still too large, reduce quality
        while buf.tell() > max_size_bytes and quality > 30:
            buf = io.BytesIO()
            quality -= 15
            img.save(buf, format="JPEG", quality=quality)

        compressed = base64.b64encode(buf.getvalue()).decode()
        logger.debug(
            f"Image compressed: {len(raw)}B → {buf.tell()}B (q={quality})"
        )
        return compressed, "image/jpeg"

    @property
    def stats(self) -> dict:
        return {
            "provider": self.provider_name,
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "failed_calls": self.failed_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
