"""Summarization over Wikipedia articles, with references we did not write.

The six articles in `summarization_task.py` were written for this project to
avoid licensing restrictions, which bought a clean licence at the cost of two
problems a reviewer will reach for: six items is too few to separate models,
and the same authors wrote the articles, the reference summaries, the rubric,
and then rated the outputs.

This task replaces both. Items come from the English Wikipedia (CC-BY-SA-4.0,
attributed per item in the cached data file): the article body is the input
and the lead section is the reference, because a lead is written to summarize
the article. The lead is removed from the body before the model sees it --
left in, the task is copying, not summarizing.

Scoring is ROUGE against that reference, exactly as in the hand-written task,
so the two are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rouge_score import rouge_scorer

from .base import Task, TaskItem

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "wikipedia_summaries.json"

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


class WikipediaSummarizationTask(Task):
    name = "wikipedia_summarization"
    category = "summarization"
    rubric = (
        "You are grading a short summary of an encyclopedia article. Score "
        "0-10 on how well the summary captures the main subject and key facts "
        "of the article WITHOUT adding details the article does not contain "
        "and without missing the main point. A summary that is accurate but "
        "omits the central subject should score low. Respond with ONLY a JSON "
        "object: {\"score\": <0-10 integer>, \"reason\": \"<one short sentence>\"}"
    )

    def get_items(self) -> List[TaskItem]:
        raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        items: List[TaskItem] = []
        for row in raw:
            prompt = (
                "Summarize the following encyclopedia article in 2-4 sentences. "
                "Respond with ONLY the summary, no preamble.\n\n"
                f"Article:\n{row['article']}"
            )
            items.append(TaskItem(
                id=row["id"],
                prompts={"default": prompt},
                reference=row["reference"],
                metadata={
                    "title": row["title"],
                    "article": row["article"],
                    "source": row.get("source", ""),
                    "license": row.get("license", "CC-BY-SA-4.0"),
                },
            ))
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
