"""A fake, zero-cost, zero-key model backend.

Purpose: let you run the ENTIRE pipeline end-to-end -- tasks, judging,
telemetry, calibration, aggregation, reporting -- before spending a single
real API call or needing any key. Set a model's `provider: mock` in
config.yaml to use it. It is deliberately not very "smart"; it should never
top a real leaderboard, it exists purely for plumbing tests.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional

from .base import ModelClient, ModelResponse


class MockClient(ModelClient):
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> ModelResponse:
        # Deterministic-ish "response" so repeated runs are reproducible:
        # seed off a hash of the prompt so the same prompt always gets the
        # same mock output within a run.
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        words = prompt.split()
        summary_len = min(len(words), rng.randint(8, 20))
        fake_text = " ".join(words[:summary_len]) + " [mock-response]"
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(fake_text) // 4)
        return ModelResponse(
            text=fake_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=0.0,
            estimated_cost_usd=0.0,
        )
