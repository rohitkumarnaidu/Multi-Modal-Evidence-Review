"""
Multi-Provider LLM Client — Enterprise Fallback Chain.

Priority: NVIDIA (40 RPM, free unlimited) → OpenRouter (20 RPM)
       → Gemini (6 keys, 5 RPM/key) → Groq (25 RPM, TPD limited)

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
    ENSEMBLE_ENABLED,
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
    PROVIDER_CONFIDENCE_WEIGHTS,
    SELF_CONSISTENCY_AGREEMENT_THRESHOLD,
    SELF_CONSISTENCY_TEMPERATURES,
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

    def call_text_ensemble(
        self,
        prompt: str,
    ) -> Optional[dict]:
        """Text call with self-consistency check.
        
        Runs the prompt twice at different temperatures, compares results,
        and returns the higher-confidence output. Falls back to single call
        if self-consistency is disabled or fails.
        """
        if not ENSEMBLE_ENABLED:
            return self.call_text(prompt, use_cache=True)

        t0, t1 = SELF_CONSISTENCY_TEMPERATURES

        result_a = self._call_text_with_temp(prompt, t0)
        if result_a is None:
            return None

        result_b = self._call_text_with_temp(prompt, t1)
        if result_b is None:
            return result_a

        agreement = self._compute_agreement(result_a, result_b)
        logger.debug(f"[Ensemble] text self-consistency agreement={agreement:.2f}")

        if agreement >= SELF_CONSISTENCY_AGREEMENT_THRESHOLD:
            result_a["confidence"] = min(1.0, result_a.get("confidence", 0.5) * (1.0 + agreement * 0.3))
            return result_a

        logger.info(f"[Ensemble] low self-consistency ({agreement:.2f}), trying secondary provider")
        secondary = self._call_secondary_text(prompt)
        if secondary:
            return self._merge_text_results(result_a, secondary)
        return result_a

    def call_vision_ensemble(
        self,
        prompt: str,
        image_data: list[dict],
        image_paths: list[str] | None = None,
    ) -> Optional[dict]:
        """Vision call with optional multi-provider voting.
        
        For VLM calls, self-consistency is expensive. Instead, we boost
        confidence from the primary result and only fall back to secondary
        if the primary fails.
        
        Returns the primary result with boosted confidence based on
        the provider's trust weight.
        """
        if not ENSEMBLE_ENABLED:
            return self.call_vision(prompt, image_data, image_paths, use_cache=True)

        result = self.call_vision(prompt, image_data, image_paths, use_cache=True)
        if result is None:
            return None

        last_provider = list(self.provider_usage.keys())[-1] if self.provider_usage else "unknown"
        weight = PROVIDER_CONFIDENCE_WEIGHTS.get(last_provider, 0.25)
        boost = 1.0 + (weight - 0.2)
        result["confidence"] = min(1.0, result.get("confidence", 0.5) * boost)
        return result

    def _call_text_with_temp(self, prompt: str, temperature: float) -> Optional[dict]:
        """Call text with modified temperature."""
        for name, client in self.providers:
            try:
                old_temp = None
                if hasattr(client, 'text_kwargs') and 'temperature' in client.text_kwargs:
                    old_temp = client.text_kwargs['temperature']
                    client.text_kwargs['temperature'] = temperature
                result = client.call_text(prompt, use_cache=False)
                if old_temp is not None:
                    client.text_kwargs['temperature'] = old_temp
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"[Ensemble] {name} text at T={temperature} failed: {e}")
        return None

    def _call_secondary_text(self, prompt: str) -> Optional[dict]:
        """Try a secondary provider for a second opinion."""
        if len(self.providers) < 2:
            return None
        primary_name = self.providers[0][0]
        for name, client in self.providers:
            if name == primary_name:
                continue
            try:
                result = client.call_text(prompt, use_cache=False)
                if result is not None:
                    self.provider_usage[name] = self.provider_usage.get(name, 0) + 1
                    logger.info(f"[Ensemble] secondary text opinion from {name}")
                    return result
            except Exception:
                continue
        return None

    def _compute_agreement(self, a: dict, b: dict) -> float:
        """Compute agreement ratio between two result dicts.
        
        Compares semantic fields (issue_type, object_type, part, etc.)
        Returns 0.0 to 1.0 where 1.0 = complete agreement.
        """
        compare_fields = [
            "visible_object_type", "visible_object_part", "visible_issue_type",
            "visible_severity", "claimed_issue_type", "claimed_object_part",
            "has_prompt_injection", "is_blurry", "is_low_light",
            "has_watermark", "has_text_instruction",
        ]
        matches = 0
        total = 0
        for field in compare_fields:
            va = a.get(field)
            vb = b.get(field)
            if va is not None and vb is not None:
                total += 1
                if str(va).strip().lower() == str(vb).strip().lower():
                    matches += 1
        return matches / max(total, 1)

    def _merge_text_results(self, primary: dict, secondary: dict) -> dict:
        """Merge two text results, preferring the more confident one per field."""
        merged = dict(primary)
        p_conf = primary.get("confidence", 0.5)
        s_conf = secondary.get("confidence", 0.5)
        fields_to_check = [
            "claimed_issue_type", "claimed_object_part", "claimed_severity_hint",
            "has_prompt_injection", "is_multi_part",
        ]
        for field in fields_to_check:
            pv = primary.get(field)
            sv = secondary.get(field)
            if pv != sv and sv is not None and pv is not None:
                if s_conf > p_conf:
                    merged[field] = sv
        merged["confidence"] = (p_conf + s_conf) / 2.0
        return merged

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
