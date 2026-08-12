"""Orchestration: for every task, for every item, for every model -- call
the model on each prompt variant, score automatically, and have every
configured judge score it too. Judge scores are stored RAW here (bias
correction happens later, in report.py, once human calibration data
exists) so `run` never needs to know whether calibration has happened yet.

Saves results/raw_results.json after EVERY item (not just at the end) so a
crash, rate limit, or Ctrl-C partway through a run doesn't lose the API
calls already paid for -- rerunning `run` simply overwrites with a fuller
file next time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from .evaluation.ai_judge import judge_item
from .models.base import ModelClient
from .tasks.base import Task
from .telemetry import TelemetryLogger


def _save(raw_results: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")


def run_benchmark(models: List[ModelClient], tasks: List[Task], judges: List[ModelClient],
                   results_dir: Path, temperature: float = 0.0, max_tokens: int = 400,
                   verbose: bool = True,
                   on_progress: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """`on_progress`, if given, is called once per (task, item, model) record
    right after it's scored, with a small JSON-serializable event dict --
    this is the hook the web UI's live progress view polls through
    (webapp/backend/main.py), added without changing any existing caller
    (the CLI just doesn't pass it, `verbose` printing still works standalone)."""
    results_dir = Path(results_dir)
    raw_results_path = results_dir / "raw_results.json"
    raw_results: List[dict] = []

    with TelemetryLogger(results_dir / "run_log.csv") as telemetry:
        for task in tasks:
            items = task.get_items()
            for item in items:
                for model in models:
                    outputs = {}
                    prompt_tokens = completion_tokens = 0
                    latency = cost = 0.0
                    had_error = False

                    for variant_name, prompt in item.prompts.items():
                        response = model.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                        telemetry.log(
                            model=model.name, task=task.name, item_id=item.id, role="subject",
                            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                            latency_seconds=response.latency_seconds,
                            estimated_cost_usd=response.estimated_cost_usd, error=response.error,
                        )
                        outputs[variant_name] = response.text
                        prompt_tokens += response.prompt_tokens
                        completion_tokens += response.completion_tokens
                        latency += response.latency_seconds
                        cost += response.estimated_cost_usd
                        had_error = had_error or not response.ok

                    if verbose:
                        status = "ERROR" if had_error else "ok"
                        print(f"[run] {task.name:14s} {item.id:20s} {model.name:20s} -> {status}")

                    automatic = {} if had_error else task.score_automatic(item, outputs)

                    judge_records = []
                    if not had_error:
                        for judge in judges:
                            result = judge_item(judge, task, item, outputs)
                            telemetry.log(
                                model=judge.name, task=task.name, item_id=item.id, role="judge",
                                prompt_tokens=result.response.prompt_tokens,
                                completion_tokens=result.response.completion_tokens,
                                latency_seconds=result.response.latency_seconds,
                                estimated_cost_usd=result.response.estimated_cost_usd,
                                error=result.response.error,
                            )
                            judge_records.append({
                                "judge_name": result.judge_name,
                                "score": result.score,
                                "reason": result.reason,
                            })

                    record = {
                        "model": model.name,
                        "task": task.name,
                        "item_id": item.id,
                        "outputs": outputs,
                        "automatic": automatic,
                        "judge_records": judge_records,
                        "cost_usd": cost,
                        "latency_s": latency,
                        "tokens": prompt_tokens + completion_tokens,
                        "error": had_error,
                    }
                    raw_results.append(record)
                    _save(raw_results, raw_results_path)

                    if on_progress:
                        on_progress({
                            "task": task.name, "item_id": item.id, "model": model.name,
                            "status": "error" if had_error else "ok",
                            "completed": len(raw_results),
                            "avg_judge_score": (
                                sum(j["score"] for j in judge_records if j["score"] is not None)
                                / max(1, sum(1 for j in judge_records if j["score"] is not None))
                            ) if judge_records else None,
                        })

    return raw_results
