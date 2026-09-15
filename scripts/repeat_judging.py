"""Scores every stored output with one judge several times, to measure how
much a judge's verdict varies when the same call is repeated.

Usage:
    python scripts/repeat_judging.py --judge gpt-oss-120b --passes 10
    [--config config.paper.yaml] [--results results]

Writes results/repeat_judging.csv (one row per call: pass, model, task, item,
score, whether the call failed, tokens). It does not modify raw_results.json;
`paper_stats.py` reads the CSV and reports the mean, spread, and agreement of
the repeated verdicts alongside the single-pass panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unibench.evaluation.ai_judge import judge_item  # noqa: E402
from unibench.models import build_client  # noqa: E402
from unibench.tasks import build_task  # noqa: E402

FIELDS = ["timestamp", "pass", "judge", "model", "task", "item_id", "score",
          "outcome", "completion_tokens", "reason"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, help="Model name from the config to use as judge.")
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--config", default="config.paper.yaml")
    parser.add_argument("--skip-empty", action="store_true", default=True,
                        help="Do not judge outputs with an empty response (Finding 8).")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results = ROOT / cfg.get("results_dir", "results")
    raw = json.loads((results / "raw_results.json").read_text(encoding="utf-8"))

    entry = next(m for m in cfg["models"] if m["name"] == args.judge)
    judge = build_client(entry["name"], entry["provider"], entry["model_id"])
    tasks = {}

    targets = [r for r in raw if not r.get("error")]
    if args.skip_empty:
        targets = [r for r in targets if not r.get("empty_outputs")]
    total = len(targets) * args.passes
    print(f"{args.judge}: {len(targets)} outputs x {args.passes} passes = {total} calls")

    out_path = results / "repeat_judging.csv"
    new_file = not out_path.exists()
    done = set()
    if not new_file:
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {(r["judge"], r["model"], r["task"], r["item_id"], r["pass"]) for r in csv.DictReader(f)}

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        call = 0
        for p in range(1, args.passes + 1):
            for record in targets:
                call += 1
                key = (args.judge, record["model"], record["task"], record["item_id"], str(p))
                if key in done:
                    continue
                task = tasks.setdefault(record["task"], build_task(record["task"]))
                item = next(it for it in task.get_items() if it.id == record["item_id"])
                result = judge_item(judge, task, item, record["outputs"])
                outcome = ("api_failed" if not result.response.ok
                           else "ok" if result.score is not None else "unparsed")
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"), "pass": p,
                    "judge": args.judge, "model": record["model"], "task": record["task"],
                    "item_id": record["item_id"], "score": result.score, "outcome": outcome,
                    "completion_tokens": result.response.completion_tokens,
                    "reason": (result.reason or "")[:200],
                })
                f.flush()
                if call % 10 == 0 or outcome != "ok":
                    print(f"[{call}/{total}] pass {p} {record['model']}/{record['item_id']}: {outcome} {result.score}",
                          flush=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
