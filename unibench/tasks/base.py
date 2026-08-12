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
        """Text block shown to the AI judge alongside `self.rubric`. Default
        implementation dumps every prompt variant next to its model output
        and the reference answer, if any; override for tasks (e.g. fairness)
        that need clearer structure."""
        lines = []
        for variant, prompt in item.prompts.items():
            lines.append(f"[{variant}] PROMPT: {prompt}")
            lines.append(f"[{variant}] MODEL RESPONSE: {outputs.get(variant, '')}")
        if item.reference:
            lines.append(f"REFERENCE ANSWER: {item.reference}")
        return "\n".join(lines)
