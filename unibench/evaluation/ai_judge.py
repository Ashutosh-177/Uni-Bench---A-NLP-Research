"""Multi-AI-judge scoring.

Each configured "judge" model reads the task rubric plus the item's
judge-context (built by `Task.build_judge_context`) and returns a 0-10
score as JSON. Judges are just ordinary `ModelClient`s -- in this project's
config, the same Groq and Gemini models used as subjects also serve as
judges, cross-judging each other's outputs (and their own).

Parsing is defensive: models frequently wrap JSON in markdown code fences
or add a stray sentence before/after it, so we try a few fallbacks before
giving up and recording a null score (which the aggregator will exclude,
not silently treat as zero).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..models.base import ModelClient, ModelResponse
from ..tasks.base import Task, TaskItem

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    judge_name: str
    score: Optional[float]   # 0-10, or None if parsing failed
    reason: str
    response: ModelResponse  # for telemetry (tokens/latency/cost)


def _parse_score(text: str) -> tuple[Optional[float], str]:
    text = text.strip()
    # Strip common markdown code-fence wrapping.
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    candidates = [text]
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            score = float(data.get("score"))
            reason = str(data.get("reason", ""))
            if 0 <= score <= 10:
                return score, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Last-resort fallback: grab the first standalone number 0-10 in the text.
    number_match = re.search(r"\b(10|[0-9](?:\.\d+)?)\b", text)
    if number_match:
        return float(number_match.group(1)), "(unparsed reason -- fallback numeric extraction)"

    return None, "(failed to parse judge response)"


def judge_item(judge: ModelClient, task: Task, item: TaskItem,
                outputs: dict[str, str]) -> JudgeResult:
    context = task.build_judge_context(item, outputs)
    prompt = f"{task.rubric}\n\n---\n{context}\n---\n\nRespond with ONLY the JSON object."
    response = judge.generate(prompt, temperature=0.0, max_tokens=150)

    if not response.ok:
        return JudgeResult(judge_name=judge.name, score=None,
                            reason=f"(judge call failed: {response.error})",
                            response=response)

    score, reason = _parse_score(response.text)
    return JudgeResult(judge_name=judge.name, score=score, reason=reason, response=response)
