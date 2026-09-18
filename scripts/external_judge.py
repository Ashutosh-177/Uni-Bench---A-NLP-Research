"""Scores every stored output with a judge drawn from OUTSIDE the model pool.

Every judge in the reported run is also one of the subject models, and two of
the three are the same family at two sizes, so the panel cannot rule out a
shared blind spot. This script adds an independent judge -- a different vendor,
a different family, a different serving stack -- and writes its verdicts to
results/external_judge.csv without touching raw_results.json.

The judge sees exactly what the in-pool judges see: the same rubric and the
same context, built by the task itself. Outputs with an empty response are
skipped by default, matching the corrected pipeline (Finding 8), so the
verdicts are directly comparable to the corrected-pipeline analysis rather
than to the uncorrected tables.

Usage:
    python scripts/external_judge.py --provider gemini --model-id gemini-3.6-flash
    [--name gemini-3.6-flash] [--config config.paper.yaml] [--results results]
    [--include-empty] [--limit N]

Re-running is safe and resumable: rows already present are not re-issued, so a
run stopped by a daily quota can be continued the next day.
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

FIELDS = ["timestamp", "judge", "provider", "model", "task", "item_id", "score",
          "outcome", "completion_tokens", "reason"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, help="mock | groq | gemini")
    parser.add_argument("--model-id", required=True, help="The provider's raw model identifier.")
    parser.add_argument("--name", help="Short name for the judge (default: --model-id).")
    parser.add_argument("--config", default="config.paper.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--include-empty", action="store_true",
                        help="Also judge outputs with an empty response (the pre-correction behavior).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N calls (0 = no limit).")
    args = parser.parse_args()
    name = args.name or args.model_id

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results = ROOT / (args.results or cfg.get("results_dir", "results"))
    raw = json.loads((results / "raw_results.json").read_text(encoding="utf-8"))

    if name in {m["name"] for m in cfg["models"]}:
        sys.exit(f"'{name}' is in the model pool; an external judge must come from outside it.")

    judge = build_client(name, args.provider, args.model_id)
    tasks: dict = {}

    targets = [r for r in raw if not r.get("error")]
    skipped_empty = 0
    if not args.include_empty:
        before = len(targets)
        targets = [r for r in targets if all((t or "").strip() for t in r["outputs"].values())]
        skipped_empty = before - len(targets)

    out_path = results / "external_judge.csv"
    new_file = not out_path.exists()
    done = set()
    if not new_file:
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {(r["judge"], r["model"], r["task"], r["item_id"]) for r in csv.DictReader(f)}

    todo = [r for r in targets if (name, r["model"], r["task"], r["item_id"]) not in done]
    print(f"{name} ({args.provider}): {len(todo)} outputs to judge "
          f"({len(targets) - len(todo)} already done, {skipped_empty} skipped as empty)")

    counts: dict[str, int] = {}
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, record in enumerate(todo, 1):
            if args.limit and i > args.limit:
                print(f"stopping at --limit {args.limit}")
                break
            task = tasks.setdefault(record["task"], build_task(record["task"]))
            item = next(it for it in task.get_items() if it.id == record["item_id"])
            result = judge_item(judge, task, item, record["outputs"])
            outcome = ("api_failed" if not result.response.ok
                       else "ok" if result.score is not None else "unparsed")
            counts[outcome] = counts.get(outcome, 0) + 1
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "judge": name, "provider": args.provider, "model": record["model"],
                "task": record["task"], "item_id": record["item_id"], "score": result.score,
                "outcome": outcome, "completion_tokens": result.response.completion_tokens,
                "reason": (result.reason or "")[:200],
            })
            f.flush()
            if i % 5 == 0 or outcome != "ok":
                print(f"[{i}/{len(todo)}] {record['model']}/{record['task']}/{record['item_id']}: "
                      f"{outcome} {result.score}", flush=True)
            if outcome == "api_failed" and "quota" in (result.reason or "").lower():
                print("provider quota reached; re-run tomorrow to resume where this stopped.")
                break

    print("outcomes:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(nothing to do)")


if __name__ == "__main__":
    main()
