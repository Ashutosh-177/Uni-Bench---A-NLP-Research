"""Abstract model-client interface.

Every backend (Groq, Gemini, Mock, ...) implements `ModelClient.generate()`
and returns a `ModelResponse`. This is the seam that lets the rest of the
framework (tasks, judges, aggregator) stay completely provider-agnostic --
adding a new backend later (OpenAI, Anthropic, local Ollama, ...) only means
adding one new file in `unibench/models/`.
"""

from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class RateLimitError(Exception):
    """Raised by a backend's `_call()` specifically for HTTP 429 / quota
    responses -- distinct from other failures so `generate()` knows this one
    is worth automatically retrying rather than immediately giving up.
    `retry_after_seconds`, if the provider told us how long to wait, is
    honored as a floor (real waits add backoff + jitter on top of it)."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning_trace(text: str) -> str:
    """Some models (e.g. Qwen's thinking mode) expose their chain-of-thought
    inline as `<think>...</think>` before the actual answer. Left in, this
    would unfairly tank that model's automatic/judge scores against models
    that only emit a final answer -- so it's stripped here, uniformly, for
    EVERY backend, keeping the comparison apples-to-apples regardless of
    which models happen to think out loud.

    If `<think>` is never closed (max_tokens cut the response off mid-
    reasoning, before any final answer was produced), there is no usable
    answer left after stripping -- return empty rather than a dangling
    reasoning fragment, so it's scored as a genuine failure, not a lucky
    partial credit."""
    if "<think>" not in text.lower():
        return text
    if "</think>" in text.lower():
        return _THINK_BLOCK_RE.sub("", text).strip()
    return _UNCLOSED_THINK_RE.sub("", text).strip()


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelClient(ABC):
    """Base class every model backend must implement.

    `name` is the short identifier used everywhere in results/reports
    (e.g. "llama-3.3-70b"); it need not match the provider's raw model id.
    `cost_per_1k_prompt` / `cost_per_1k_completion` are USD estimates used
    ONLY for the efficiency comparison in the report -- they are
    approximate list prices, not billing-accurate, and free-tier calls
    (e.g. Groq) should set both to 0.0.
    """

    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0
    max_rate_limit_retries: int = 5

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @abstractmethod
    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> ModelResponse:
        """Provider-specific implementation. Must NOT set latency_seconds --
        the public `generate()` wrapper measures wall-clock time uniformly
        so every backend is timed the same way. Raise `RateLimitError` (not
        a generic exception) for HTTP 429 / quota responses so `generate()`
        knows to retry automatically instead of recording a permanent
        failure -- free-tier rate limits (e.g. Groq's per-minute token cap)
        are transient and self-heal within seconds."""
        raise NotImplementedError

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 400) -> ModelResponse:
        start = time.perf_counter()
        response: Optional[ModelResponse] = None

        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                response = self._call(prompt, system, temperature, max_tokens)
                break
            except RateLimitError as exc:
                if attempt >= self.max_rate_limit_retries:
                    response = ModelResponse(
                        text="", prompt_tokens=0, completion_tokens=0,
                        latency_seconds=0.0, estimated_cost_usd=0.0,
                        error=f"rate-limited after {attempt + 1} attempts: {exc}",
                    )
                    break
                # Honor the provider's suggested wait as a floor, then add
                # exponential backoff + jitter on top -- the provider's own
                # estimate (e.g. Groq's "try again in 75ms") is frequently
                # too optimistic once multiple requests are already queued.
                floor = exc.retry_after_seconds or 1.0
                backoff = floor + (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
                time.sleep(backoff)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
                # OTHER backend failure (bad model id, auth, network, ...)
                # becomes a recorded, scoreable error rather than crashing
                # the whole benchmark run. Not retried -- these don't self-heal.
                response = ModelResponse(
                    text="", prompt_tokens=0, completion_tokens=0,
                    latency_seconds=0.0, estimated_cost_usd=0.0, error=str(exc),
                )
                break

        response.latency_seconds = time.perf_counter() - start
        if response.error is None:
            response.text = _strip_reasoning_trace(response.text)
            response.estimated_cost_usd = (
                response.prompt_tokens / 1000 * self.cost_per_1k_prompt
                + response.completion_tokens / 1000 * self.cost_per_1k_completion
            )
        return response
