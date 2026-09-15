"""Shared Task interface.

A `TaskItem` holds one or more named "prompt variants". Most tasks
(summarization, NER, QA) have exactly one variant, `{"default": prompt}`.
The fairness task has two, `{"variant_a": ..., "variant_b": ...}` -- the
CFE-style contrastive pair that differs only in the sensitive attribute.
Modeling it this way lets `runner.py` treat every task identically: fetch
each variant's prompt, call the model on each, hand all the outputs back
to the task's own `score_automatic()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Shown in place of an empty response so neither judges nor human raters read
# the next block of text (e.g. the reference answer) as the model's output.
EMPTY_RESPONSE_MARKER = "(EMPTY RESPONSE: the model returned no text)"


def response_or_marker(text: Optional[str]) -> str:
    return (text or "").strip() or EMPTY_RESPONSE_MARKER


@dataclass
class TaskItem:
    id: str
    prompts: Dict[str, str]          # variant name -> prompt text
    reference: Optional[str] = None  # gold answer, if any (e.g. summary)
    metadata: dict = field(default_factory=dict)


class Task(ABC):
    name: str
    category: str
    rubric: str  # instructions handed to the AI-judge panel for this task

    @abstractmethod
    def get_items(self) -> List[TaskItem]:
        """Return the (small, curated) evaluation set for this task."""
        raise NotImplementedError

    @abstractmethod
    def score_automatic(self, item: TaskItem, outputs: Dict[str, str]) -> Dict[str, float]:
        """outputs maps variant name -> model response text.
        Returns a dict of metric_name -> float (higher = better, by
        convention, except metrics explicitly named '..._disparity' or
        '..._error', which are lower-is-better and handled specially by
        the aggregator)."""
        raise NotImplementedError

    def build_judge_context(self, item: TaskItem, outputs: Dict[str, str]) -> str:
        """Text block shown to the AI judge (and to human raters) alongside
        `self.rubric`. The reference answer, if any, comes before the model
        response and is labelled as such: placed after an empty response, it
        was graded as if it were the response. Override for tasks (e.g.
        fairness) that need clearer structure."""
        lines = [f"[{variant}] PROMPT: {prompt}" for variant, prompt in item.prompts.items()]
        if item.reference:
            lines.append(f"REFERENCE ANSWER (for comparison only; this is NOT the response being graded): "
                         f"{item.reference}")
        for variant in item.prompts:
            lines.append(f"[{variant}] MODEL RESPONSE TO GRADE: {response_or_marker(outputs.get(variant))}")
        return "\n".join(lines)
