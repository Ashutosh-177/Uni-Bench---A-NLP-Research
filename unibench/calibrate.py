"""Human-calibration step, shared by the CLI and the web dashboard.

Samples a handful of (model, task, item, judge) records from
results/raw_results.json that already have a parsed AI-judge score, and
resolves the same context the AI judge saw. A human -- YOU, via the CLI's
interactive loop or the web dashboard's calibration page -- rates each one
0-10. Saved to results/human_calibration.csv, where
`evaluation.bias_calibration` picks it up to correct each judge's scores
before the final report.

`sample_calibration_candidates()` and `append_calibration_row()` are the
reusable core (no I/O prompting, fully JSON-serializable) -- the CLI's
`run_calibration()` below is just an interactive loop built on top of them,
and `webapp/backend/main.py`'s calibration endpoints use the exact same two
functions, so the two surfaces can't drift into different sampling logic.
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


def _load_raw_results(results_dir: Path) -> List[dict]:
    path = results_dir / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run_benchmark.py run` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def sample_calibration_candidates(results_dir: Path, n: int = 10,
                                   seed: Optional[int] = None) -> List[dict]:
    """Returns up to `n` randomly sampled, fully-resolved candidates ready to
    show a human rater: {model, task, item_id, judge_name, ai_score,
    ai_reason, context}. `seed=None` (the default) gives a fresh random
    sample each call, which is what the web UI wants; the CLI passes a fixed
    seed=0 for reproducible runs."""
    results_dir = Path(results_dir)
    raw_results = _load_raw_results(results_dir)

    candidates = []
    for record in raw_results:
        if record.get("error"):
            continue
        for jr in record.get("judge_records", []):
            if jr["score"] is not None:
                candidates.append((record, jr))

    if not candidates:
        return []

    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(n, len(candidates)))

    task_cache = {}
    resolved = []
    for record, jr in sample:
        task_name = record["task"]
        if task_name not in task_cache:
            task_cache[task_name] = build_task(task_name)
        task = task_cache[task_name]
        item = next((it for it in task.get_items() if it.id == record["item_id"]), None)
        context = task.build_judge_context(item, record["outputs"]) if item else "(context unavailable)"

        resolved.append({
            "model": record["model"], "task": task_name, "item_id": record["item_id"],
            "judge_name": jr["judge_name"], "ai_score": jr["score"], "ai_reason": jr["reason"],
            "context": context,
        })
    return resolved


def append_calibration_row(results_dir: Path, row: dict) -> None:
    """Appends one human-rated row to results/human_calibration.csv. `row`
    must have all of CALIBRATION_FIELDNAMES (extra keys, e.g. `context`,
    are silently dropped)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "human_calibration.csv"
    is_new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATION_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in CALIBRATION_FIELDNAMES})


def run_calibration(results_dir: Path, sample_size: int = 10, seed: int = 0) -> None:
    """CLI entrypoint: interactive terminal loop over
    `sample_calibration_candidates()`, writing each confirmed rating via
    `append_calibration_row()`."""
    results_dir = Path(results_dir)
    sample = sample_calibration_candidates(results_dir, n=sample_size, seed=seed)

    if not sample:
        print("No scored (model, task, item, judge) records found -- run `run` first.")
        return

    print(f"\n=== Human calibration: {len(sample)} items to rate ===")
    print("For each one, you'll see the same context the AI judge saw. Enter your own")
    print("0-10 score (or 's' to skip this one, 'q' to stop early and save what you have).\n")

    rows_written = 0
    for i, cand in enumerate(sample, 1):
        print(f"--- [{i}/{len(sample)}] model={cand['model']}  task={cand['task']}  item={cand['item_id']} ---")
        print(cand["context"])
        print(f"\nAI judge ({cand['judge_name']}) gave: {cand['ai_score']}/10 -- \"{cand['ai_reason']}\"")
        raw = input("Your score (0-10, 's' skip, 'q' quit): ").strip().lower()

        if raw == "q":
            break
        if raw == "s" or raw == "":
            continue
        try:
            human_score = float(raw)
            assert 0 <= human_score <= 10
        except (ValueError, AssertionError):
            print("  (invalid input, skipping this one)")
            continue

        append_calibration_row(results_dir, {**cand, "human_score": human_score})
        rows_written += 1

    print(f"\nSaved {rows_written} human-calibration rows to {results_dir / 'human_calibration.csv'}")
