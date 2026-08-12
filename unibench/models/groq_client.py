"""Groq backend -- OpenAI-compatible chat completions API.

Used as the framework's "open-weight model" source (e.g. Llama-3.3-70B,
Qwen-2.5-72B) via Groq's free-tier, high-throughput inference -- the same
approach used by the Agentic-AI-HEMS paper in the literature review.

Requires GROQ_API_KEY in the environment (see .env.example). Get a free key
at https://console.groq.com/keys
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests

from .base import ModelClient, ModelResponse, RateLimitError

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Groq's 429 body embeds a hint like "Please try again in 75ms." or
# "...in 1.234s." -- parsed as a floor for the retry backoff in base.py.
_RETRY_HINT_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)


def _parse_retry_after(resp: requests.Response) -> Optional[float]:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_HINT_RE.search(resp.text)
    if match:
        value, unit = match.groups()
        return float(value) / 1000.0 if unit.lower() == "ms" else float(value)
    return None


class GroqClient(ModelClient):
    # Groq's free tier is $0 for the models this project targets; set to 0.0
    # so the "efficiency" comparison reflects real cost. If you move to a
    # paid Groq tier, update these from https://groq.com/pricing/
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> ModelResponse:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        if resp.status_code == 429:
            raise RateLimitError(f"Groq rate limit: {resp.text[:300]}",
                                  retry_after_seconds=_parse_retry_after(resp))
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ModelResponse(
            text=choice,
            prompt_tokens=usage.get("prompt_tokens", max(1, len(prompt) // 4)),
            completion_tokens=usage.get("completion_tokens", max(1, len(choice) // 4)),
            latency_seconds=0.0,
            estimated_cost_usd=0.0,
            raw=data,
        )
