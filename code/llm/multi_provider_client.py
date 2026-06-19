"""
Multi-Provider LLM Client — Enterprise Fallback Chain.

Priority: Gemini (multi-key) → Groq → OpenRouter → NVIDIA

Each provider is tried in order. If one fails all retries,
the next provider is automatically attempted. The pipeline
code doesn't need to know which provider succeeded — same
call_text() / call_vision() interface throughout.

Also tracks which provider handled each call for comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import (
    GEMINI_API_KEYS,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_TEXT_MODEL,
    GROQ_VISION_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_TEXT_MODEL,
    NVIDIA_VISION_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_API_KEYS,
    OPENROUTER_BASE_URL,
    OPENROUTER_TEXT_MODEL,
    OPENROUTER_VISION_MODEL,
)
from llm.cache import ResponseCache

logger = logging.getLogger(__name__)


class MultiProviderClient:
    """Orchestrates multiple LLM providers with automatic fallback.

    Same interface as GeminiClient: call_text() and call_vision().
    Internally tries providers in priority order until one succeeds.

    Args:
        only_provider: If set, only initialize this provider (for model comparison).
                      Options: "gemini", "groq", "openrouter", "nvidia"
    """

    def __init__(self, only_provider: str | None = None):
        self.cache = ResponseCache()
        self.providers = []
        self.provider_usage: dict[str, int] = {}  # provider → success count
        self.only_provider = only_provider

        # ── Provider 1: NVIDIA (unlimited credits, 40 RPM — best first) ───
        if NVIDIA_API_KEY and (not only_provider or only_provider == "nvidia"):
            from llm.openai_compat_client import OpenAICompatClient
            try:
                nvidia = OpenAICompatClient(
                    provider_name="nvidia",
                    api_key=NVIDIA_API_KEY,
                    base_url=NVIDIA_BASE_URL,
                    text_model=NVIDIA_TEXT_MODEL,
                    vision_model=NVIDIA_VISION_MODEL,
                    cache=self.cache,
                )
                self.providers.append(("nvidia", nvidia))
                logger.info("[MultiProvider] NVIDIA enabled")
            except Exception as e:
                logger.warning(f"[MultiProvider] NVIDIA init failed: {e}")

        # ── Provider 2: OpenRouter (20 RPM, 200+ RPD free, good accuracy) ─
        if OPENROUTER_API_KEY and (not only_provider or only_provider == "openrouter"):
            from llm.openai_compat_client import OpenAICompatClient
            try:
                openrouter = OpenAICompatClient(
                    provider_name="openrouter",
                    api_key=OPENROUTER_API_KEYS,  # pass all keys for rotation
                    base_url=OPENROUTER_BASE_URL,
                    text_model=OPENROUTER_TEXT_MODEL,
                    vision_model=OPENROUTER_VISION_MODEL,
                    cache=self.cache,
                )
                self.providers.append(("openrouter", openrouter))
                logger.info(
                    f"[MultiProvider] OpenRouter enabled ({len(OPENROUTER_API_KEYS)} keys)"
                )
            except Exception as e:
                logger.warning(f"[MultiProvider] OpenRouter init failed: {e}")

        # ── Provider 3: Gemini (6 keys × 5 RPM / 20 RPD — fallback) ──────
        if GEMINI_API_KEYS and (not only_provider or only_provider == "gemini"):
            from llm.gemini_client import GeminiClient
            try:
                gemini = GeminiClient()
                gemini.cache = self.cache
                self.providers.append(("gemini", gemini))
                logger.info(
                    f"[MultiProvider] Gemini enabled ({len(GEMINI_API_KEYS)} keys)"
                )
            except Exception as e:
                logger.warning(f"[MultiProvider] Gemini init failed: {e}")

        # ── Provider 4: Groq (25 RPM, TPD limited — last resort) ─────────
        if GROQ_API_KEY and (not only_provider or only_provider == "groq"):
            from llm.openai_compat_client import OpenAICompatClient
            try:
                groq = OpenAICompatClient(
                    provider_name="groq",
                    api_key=GROQ_API_KEY,
                    base_url=GROQ_BASE_URL,
                    text_model=GROQ_TEXT_MODEL,
                    vision_model=GROQ_VISION_MODEL,
                    cache=self.cache,
                )
                self.providers.append(("groq", groq))
                logger.info("[MultiProvider] Groq enabled")
            except Exception as e:
                logger.warning(f"[MultiProvider] Groq init failed: {e}")

        if not self.providers:
            raise ValueError(
                "No LLM providers available. Set at least one API key in .env"
            )

        logger.info(
            f"[MultiProvider] {len(self.providers)} providers ready: "
            f"{[name for name, _ in self.providers]}"
        )

    def call_text(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Text-only call with fallback across providers."""
        for name, client in self.providers:
            try:
                result = client.call_text(prompt, use_cache=use_cache)
                if result is not None:
                    self.provider_usage[name] = self.provider_usage.get(name, 0) + 1
                    logger.debug(f"[MultiProvider] text call succeeded via {name}")
                    return result
                else:
                    logger.warning(
                        f"[MultiProvider] {name} returned None for text call. "
                        f"Trying next provider..."
                    )
            except Exception as e:
                logger.warning(
                    f"[MultiProvider] {name} text call error: {e}. "
                    f"Trying next provider..."
                )

        logger.error("[MultiProvider] All providers failed for text call")
        return None

    def call_vision(
        self,
        prompt: str,
        image_data: list[dict],
        image_paths: list[str] | None = None,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Vision call with fallback across providers."""
        for name, client in self.providers:
            try:
                result = client.call_vision(
                    prompt, image_data, image_paths, use_cache=use_cache
                )
                if result is not None:
                    self.provider_usage[name] = self.provider_usage.get(name, 0) + 1
                    logger.debug(f"[MultiProvider] vision call succeeded via {name}")
                    return result
                else:
                    logger.warning(
                        f"[MultiProvider] {name} returned None for vision call. "
                        f"Trying next provider..."
                    )
            except Exception as e:
                logger.warning(
                    f"[MultiProvider] {name} vision call error: {e}. "
                    f"Trying next provider..."
                )

        logger.error("[MultiProvider] All providers failed for vision call")
        return None

    @property
    def tracker(self):
        """Return tracker from primary provider (Gemini) for backward compat."""
        for name, client in self.providers:
            if hasattr(client, "tracker"):
                return client.tracker
        # Fallback: create a minimal tracker
        from llm.gemini_client import TokenTracker
        return TokenTracker()

    @property
    def stats(self) -> dict:
        """Aggregate stats from all providers."""
        all_stats = {}
        for name, client in self.providers:
            if hasattr(client, "tracker"):
                all_stats[name] = client.tracker.stats
            elif hasattr(client, "stats"):
                all_stats[name] = client.stats
        all_stats["provider_usage"] = self.provider_usage
        all_stats["cache_stats"] = self.cache.stats
        return all_stats
