"""Google Gemini backend -- REST `generateContent` API.

Requires GEMINI_API_KEY in the environment (see .env.example). Get a free
key at https://aistudio.google.com/apikey -- Gemini Flash models have a
usable free tier, which is why this is one of the framework's two
"proprietary API" backends.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from .base import ModelClient, ModelResponse, RateLimitError

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiClient(ModelClient):
    # Free-tier usage -> $0. If you exceed the free quota and move to
    # pay-as-you-go, update these from https://ai.google.dev/pricing
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> ModelResponse:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                # Gemini 3.x defaults to an internal "thinking" pass that
                # draws from the SAME maxOutputTokens budget as the visible
                # answer. Verified empirically: on the short fairness-probe
                # prompts this left enough headroom to still answer, but on
                # the longer summarization prompts it consumed the entire
                # 800-token budget on hidden reasoning, returning empty text
                # on 100% of items (finishReason=MAX_TOKENS, 0 visible
                # tokens) -- the same silent-failure class as the Qwen and
                # judge-budget findings, just on a third backend. Disabled
                # outright (thinkingBudget=0) rather than raised, since this
                # framework's standardization principle (Section III) calls
                # for one shared, fixed budget across every model; a model
                # that needs a variable, much larger budget to "think" isn't
                # comparable to the others under that budget in the first
                # place.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = GEMINI_ENDPOINT_TEMPLATE.format(model=self.model_id)
        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimitError(
                f"Gemini rate limit: {resp.text[:300]}",
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates or "content" not in candidates[0]:
            # Usually means the safety filter blocked the response.
            reason = candidates[0].get("finishReason", "UNKNOWN") if candidates else "NO_CANDIDATES"
            raise RuntimeError(f"Gemini returned no usable content (finishReason={reason})")

        parts = candidates[0]["content"].get("parts", [{}])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return ModelResponse(
            text=text,
            prompt_tokens=usage.get("promptTokenCount", max(1, len(prompt) // 4)),
            completion_tokens=usage.get("candidatesTokenCount", max(1, len(text) // 4)),
            latency_seconds=0.0,
            estimated_cost_usd=0.0,
            raw=data,
        )
