"""Incremental telemetry logging.

Appends one row per API call to results/run_log.csv AS IT HAPPENS (not
buffered in memory until the end) so a long benchmark run that crashes or
hits a rate limit halfway through doesn't lose everything already paid for
in API calls -- you can inspect run_log.csv immediately, and `report.py`
can work from whatever raw_results.json was saved so far.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

FIELDNAMES = [
    "timestamp", "model", "task", "item_id", "role", "prompt_tokens",
    "completion_tokens", "latency_seconds", "estimated_cost_usd", "error",
]


class TelemetryLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, model: str, task: str, item_id: str, role: str,
             prompt_tokens: int, completion_tokens: int, latency_seconds: float,
             estimated_cost_usd: float, error: Optional[str] = None) -> None:
        """`role` is one of "subject" (the model being benchmarked) or
        "judge" (an AI judge scoring a response) -- keeps judging overhead
        visible and separable from subject-model cost in run_log.csv."""
        self._writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model, "task": task, "item_id": item_id, "role": role,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "latency_seconds": round(latency_seconds, 3),
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "error": error or "",
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
