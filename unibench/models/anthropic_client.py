"""Anthropic (Claude) backend.

Added as a genuine second provider and as a judge from outside the Groq
model pool -- the accompanying paper's panel is three Groq-served models,
two of them the same family at two sizes, which cannot rule out a shared
blind spot.

Unlike the Groq and Gemini backends, this one is billed, so it is the only
client that reports a real (not zeroed) cost. `estimated_cost_usd` is
computed from the per-token list prices below and the token counts the API
itself returns, so the efficiency comparison for this provider rests on
measured usage rather than an assumed rate. Prices are USD per 1K tokens,
from Anthropic's published rates; update them if the rates change.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import ModelClient, ModelResponse, RateLimitError

# USD per 1K tokens (list prices / 1000).
PRICES = {
    "claude-opus-5":    (0.005,  0.025),
    "claude-sonnet-5":  (0.002,  0.010),
    "claude-haiku-4-5": (0.001,  0.005),
}
DEFAULT_PRICE = PRICES["claude-sonnet-5"]


class AnthropicClient(ModelClient):
    """Claude via the official `anthropic` SDK.

    Thinking is left off. A judge verdict here is one sentence of JSON, and
    the paper's Findings 2, 3 and 5 are all the same failure: a reasoning
    pass consuming a shared output budget until no visible answer is left.
    Keeping this backend non-thinking means its budget means the same thing
    as every other backend's, so the shared-protocol comparison holds.
    """

    def __init__(self, name: str, model_id: str):
        super().__init__(name=name, model_id=model_id)
        prompt_price, completion_price = PRICES.get(model_id, DEFAULT_PRICE)
        # Instance-level, not class-level: two AnthropicClients in one pool
        # may be different models at different prices.
        self.cost_per_1k_prompt = prompt_price
        self.cost_per_1k_completion = completion_price

    def _takes_sampling_params(self) -> bool:
        """Whether this model still accepts `temperature`. The Haiku 4.5 and
        earlier generations do; Opus 5, Sonnet 5, Fable and the 4.6-4.8 family
        removed sampling and return 400 if it is sent."""
        return self.model_id.startswith(("claude-haiku", "claude-3"))

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> ModelResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example)."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The 'anthropic' package is required: pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        # Current Claude models removed the sampling parameters: sending
        # `temperature` at all returns 400 "`temperature` is deprecated for
        # this model". They also run adaptive thinking by default, which this
        # backend turns off so its output budget means the same thing as every
        # other backend's (see the class docstring). Older models keep
        # temperature and have no thinking unless asked.
        if self._takes_sampling_params():
            if temperature is not None:
                kwargs["temperature"] = temperature
        else:
            kwargs["thinking"] = {"type": "disabled"}

        try:
            msg = client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after", "") or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
            raise RateLimitError(f"Anthropic rate limit: {exc}", retry_after_seconds=retry_after)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise RateLimitError(f"Anthropic server error {exc.status_code}: {exc}")
            raise RuntimeError(f"Anthropic API error {exc.status_code}: {str(exc)[:300]}")
        except anthropic.APIConnectionError as exc:
            raise RateLimitError(f"Anthropic connection error: {exc}")

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        prompt_tokens = msg.usage.input_tokens
        completion_tokens = msg.usage.output_tokens
        cost = (prompt_tokens / 1000.0) * self.cost_per_1k_prompt + \
               (completion_tokens / 1000.0) * self.cost_per_1k_completion
        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=0.0,
            estimated_cost_usd=cost,
            raw={"stop_reason": msg.stop_reason, "model": msg.model},
        )
