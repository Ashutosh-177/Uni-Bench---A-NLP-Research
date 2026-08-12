"""Summarization task -- a small, hand-authored sample set (MVP scope).

These 6 short articles are self-authored specifically for this project (no
licensing concerns) so the pipeline can be tested cheaply and quickly. To
scale up to the full paper design, swap `get_items()` to load a subset of
CNN/DailyMail or XSum via the HuggingFace `datasets` library -- the rest of
the framework (scoring, judging, calibration, aggregation) needs no changes
since it only depends on the `Task`/`TaskItem` interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rouge_score import rouge_scorer

from .base import Task, TaskItem

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "summarization_samples.json"

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


class SummarizationTask(Task):
    name = "summarization"
    category = "summarization"
    rubric = (
        "You are grading a one-to-two-sentence summary of a short news "
        "article. Score 0-10 on how well the summary captures the key facts "
        "of the article WITHOUT adding invented details or missing the main "
        "point. Respond with ONLY a JSON object: {\"score\": <0-10 integer>, "
        "\"reason\": \"<one short sentence>\"}"
    )

    def get_items(self) -> List[TaskItem]:
        raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        items = []
        for row in raw:
            prompt = (
                "Summarize the following news article in 1-2 sentences. "
                "Respond with ONLY the summary, no preamble.\n\n"
                f"Article:\n{row['article']}"
            )
            items.append(
                TaskItem(
                    id=row["id"],
                    prompts={"default": prompt},
                    reference=row["reference_summary"],
                    metadata={"article": row["article"]},
                )
            )
        return items

    def score_automatic(self, item: TaskItem, outputs: Dict[str, str]) -> Dict[str, float]:
        summary = (outputs.get("default", "") or "").strip()
        if not summary or not item.reference:
            return {"rouge1_f": float("nan"), "rouge2_f": float("nan"), "rougeL_f": float("nan")}
        scores = _scorer.score(item.reference, summary)
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }
