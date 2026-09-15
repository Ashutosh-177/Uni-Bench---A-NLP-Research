"""Re-issues judge calls that the provider rejected at the API (rate limits)
during a run, with the same prompt, temperature and token budget, pausing
between calls so the per-minute quota can reset. Verdicts that were returned
but could not be parsed are left as they are.

Usage:  python scripts/reissue_rejected_judges.py [--config config.paper.yaml] [--pause 65]

Updates raw_results.json in place (each re-issued verdict gets "reissued":
true), logs every attempt to reissued_judge_calls.csv, and rebuilds
human_calibration.csv from human_ratings.csv so that new verdicts are paired
with the existing human ratings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unibench.calibrate import CALIBRATION_FIELDNAMES  # noqa: E402
from unibench.evaluation.ai_judge import judge_item  # noqa: E402
from unibench.models import build_client  # noqa: E402
from unibench.tasks import build_task  # noqa: E402


def rejected_at_api(jr: dict) -> bool:
    return jr["score"] is None and jr.get("call_failed", jr["reason"].startswith("(judge call failed"))


def rebuild_calibration(results: Path, raw: list) -> int:
    ratings_path = results / "human_ratings.csv"
    if not ratings_path.exists():
        return 0
    with open(ratings_path, newline="", encoding="utf-8") as f:
        ratings = {(r["model"], r["task"], r["item_id"]): float(r["human_score"]) for r in csv.DictReader(f)}
    rows = []
    for r in raw:
        key = (r["model"], r["task"], r["item_id"])
        if key not in ratings:
            continue
        for jr in r["judge_records"]:
            if jr["score"] is not None:
                rows.append({"model": key[0], "task": key[1], "item_id": key[2], "judge_name": jr["judge_name"],
                             "ai_score": jr["score"], "ai_reason": jr["reason"], "human_score": ratings[key]})
    with open(results / "human_calibration.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.paper.yaml")
    parser.add_argument("--pause", type=float, default=65.0, help="Seconds to wait between calls.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results = ROOT / cfg.get("results_dir", "results")
    raw_path = results / "raw_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    clients = {m["name"]: build_client(m["name"], m["provider"], m["model_id"]) for m in cfg["models"]}
    tasks = {}

    todo = [(r, jr) for r in raw for jr in r["judge_records"] if rejected_at_api(jr)]
    print(f"{len(todo)} judge calls were rejected at the API; re-issuing with a {args.pause:.0f}s pause between calls.")

    log_path = results / "reissued_judge_calls.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["timestamp", "judge", "model", "task", "item_id", "outcome", "score",
                          "completion_tokens", "error"])
        for i, (record, jr) in enumerate(todo, 1):
            task = tasks.setdefault(record["task"], build_task(record["task"]))
            item = next(it for it in task.get_items() if it.id == record["item_id"])
            result = judge_item(clients[jr["judge_name"]], task, item, record["outputs"])
            outcome = "api_failed" if not result.response.ok else ("ok" if result.score is not None else "unparsed")
            jr.update(score=result.score, reason=result.reason, call_failed=not result.response.ok, reissued=True)
            record["unparsed_judges"] = [j["judge_name"] for j in record["judge_records"] if j["score"] is None]
            raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            log.writerow([datetime.now().isoformat(timespec="seconds"), jr["judge_name"], record["model"],
                          record["task"], record["item_id"], outcome, result.score,
                          result.response.completion_tokens, (result.response.error or "")[:200]])
            log_file.flush()
            print(f"[{i}/{len(todo)}] {jr['judge_name']} on {record['model']}/{record['item_id']}: {outcome} {result.score}",
                  flush=True)
            if i < len(todo):
                time.sleep(args.pause)

    print(f"Rebuilt human_calibration.csv with {rebuild_calibration(results, raw)} pairs.")


if __name__ == "__main__":
    main()
