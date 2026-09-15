"""Human-calibration step, shared by the CLI and the web dashboard.

The CLI (`run_benchmark.py calibrate`) is blind: a person rates each
(model, task, item) output once, seeing the same rubric and context the AI
judges saw but not which model wrote it or what any judge scored it, so the
rating cannot be anchored on an AI verdict. Each rating goes to
results/human_ratings.csv and is paired with every judge's parsed score for
that output in results/human_calibration.csv, which
`evaluation.bias_calibration` reads to correct each judge.

`sample_calibration_candidates()` and `append_calibration_row()` are also
used by the dashboard's calibration endpoints (webapp/backend/main.py).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List, Optional

from .tasks import build_task

CALIBRATION_FIELDNAMES = ["model", "task", "item_id", "judge_name",
                           "ai_score", "ai_reason", "human_score"]
HUMAN_RATING_FIELDNAMES = ["model", "task", "item_id", "human_score"]


def _load_raw_results(results_dir: Path) -> List[dict]:
    path = results_dir / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run_benchmark.py run` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _append_csv(path: Path, fieldnames: List[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in fieldnames})


def sample_calibration_candidates(results_dir: Path, n: int = 10,
                                   seed: Optional[int] = None) -> List[dict]:
    """Returns up to `n` randomly sampled (model, task, item, judge) records
    that have a parsed judge score, with the context the judge saw."""
    results_dir = Path(results_dir)
    raw_results = _load_raw_results(results_dir)

    candidates = [(record, jr) for record in raw_results if not record.get("error")
                  for jr in record.get("judge_records", []) if jr["score"] is not None]
    if not candidates:
        return []

    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(n, len(candidates)))

    task_cache = {}
    resolved = []
    for record, jr in sample:
        task = task_cache.setdefault(record["task"], build_task(record["task"]))
        item = next((it for it in task.get_items() if it.id == record["item_id"]), None)
        context = task.build_judge_context(item, record["outputs"]) if item else "(context unavailable)"
        resolved.append({
            "model": record["model"], "task": record["task"], "item_id": record["item_id"],
            "judge_name": jr["judge_name"], "ai_score": jr["score"], "ai_reason": jr["reason"],
            "context": context,
        })
    return resolved


def append_calibration_row(results_dir: Path, row: dict) -> None:
    """Appends one (model, task, item, judge, AI score, human score) row to
    results/human_calibration.csv. Extra keys in `row` are ignored."""
    _append_csv(Path(results_dir) / "human_calibration.csv", CALIBRATION_FIELDNAMES, row)


def ratings_path(results_dir: Path, rater: Optional[str] = None) -> Path:
    """The primary rater writes human_ratings.csv; any additional rater gets
    their own file, used only to measure agreement between raters."""
    if not rater:
        return Path(results_dir) / "human_ratings.csv"
    safe = "".join(ch for ch in rater.lower() if ch.isalnum() or ch == "_")
    return Path(results_dir) / f"human_ratings_{safe}.csv"


def _rated_outputs(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["model"], r["task"], r["item_id"]) for r in csv.DictReader(f)}


def _rubric_for_humans(rubric: str) -> str:
    return rubric.split("Respond with ONLY")[0].strip()


def run_calibration(results_dir: Path, limit: Optional[int] = None, seed: int = 0,
                    rater: Optional[str] = None) -> None:
    """CLI entrypoint: blind rating of every output not rated yet (or at most
    `limit` of them this session). Safe to stop with 'q' and resume later.
    With `rater`, ratings go to a separate file and are not used for judge
    calibration (they measure agreement with the primary rater)."""
    results_dir = Path(results_dir)
    records = [r for r in _load_raw_results(results_dir) if not r.get("error")]
    random.Random(seed).shuffle(records)
    out_path = ratings_path(results_dir, rater)
    rated = _rated_outputs(out_path)
    pending = [r for r in records if (r["model"], r["task"], r["item_id"]) not in rated]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"All {len(records)} outputs have already been rated.")
        return

    print(f"\n=== Blind human rating{f' (rater: {rater})' if rater else ''}: {len(pending)} outputs to rate "
          f"({len(rated)} of {len(records)} already rated) -> {out_path.name} ===")
    print("You will not see which model wrote each output or what the judges scored it.")
    print("Score each one 0-10 using the rubric shown. 's' skips, 'q' saves and stops.\n")

    tasks = {}
    written = 0
    for i, record in enumerate(pending, 1):
        task = tasks.setdefault(record["task"], build_task(record["task"]))
        item = next((it for it in task.get_items() if it.id == record["item_id"]), None)
        if item is None:
            continue
        print(f"--- [{i}/{len(pending)}] task={task.name}  item={record['item_id']} ---")
        print(f"RUBRIC: {_rubric_for_humans(task.rubric)}\n")
        print(task.build_judge_context(item, record["outputs"]))
        if any(not (t or "").strip() for t in record["outputs"].values()):
            print("(Note: at least one response above is empty.)")
        raw = input("Your score (0-10, 's' skip, 'q' quit): ").strip().lower()
        print()
        if raw == "q":
            break
        if raw in ("s", ""):
            continue
        try:
            human_score = float(raw)
            assert 0 <= human_score <= 10
        except (ValueError, AssertionError):
            print("  (invalid input, skipping this one)\n")
            continue

        key = {"model": record["model"], "task": record["task"], "item_id": record["item_id"]}
        _append_csv(out_path, HUMAN_RATING_FIELDNAMES, {**key, "human_score": human_score})
        for jr in ([] if rater else record.get("judge_records", [])):
            if jr["score"] is not None:
                append_calibration_row(results_dir, {**key, "judge_name": jr["judge_name"],
                                                     "ai_score": jr["score"], "ai_reason": jr["reason"],
                                                     "human_score": human_score})
        written += 1

    print(f"Saved {written} ratings this session "
          f"({len(rated) + written} of {len(records)} outputs rated in total).")
