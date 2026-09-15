"""Folds a repeated-judging run into raw_results.json as one verdict per output.

The judge's score for an output is the mean of its repeated passes, recorded
with the number of passes and their standard deviation so the aggregation step
treats it like any other judge while the spread stays visible.

Usage:
    python scripts/merge_repeat_judge.py --judge gpt-oss-120b [--results results]

Re-running is safe: an existing merged verdict for that judge is replaced.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    results = ROOT / args.results
    raw_path = results / "raw_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rep = pd.read_csv(results / "repeat_judging.csv")
    rep = rep[(rep["judge"] == args.judge) & (rep["outcome"] == "ok")]
    if rep.empty:
        sys.exit(f"no scored passes for {args.judge}")

    scores = rep.groupby(["model", "task", "item_id"])["score"].apply(list).to_dict()
    merged = 0
    for record in raw:
        key = (record["model"], record["task"], record["item_id"])
        if key not in scores:
            continue
        passes = scores[key]
        record["judge_records"] = [j for j in record["judge_records"] if j["judge_name"] != args.judge]
        record["judge_records"].append({
            "judge_name": args.judge,
            "score": float(statistics.fmean(passes)),
            "reason": f"(mean of {len(passes)} repeated passes)",
            "call_failed": False,
            "passes": len(passes),
            "score_sd": float(statistics.stdev(passes)) if len(passes) > 1 else 0.0,
        })
        record["unparsed_judges"] = [j["judge_name"] for j in record["judge_records"] if j["score"] is None]
        merged += 1

    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"merged {args.judge} into {merged} of {len(raw)} records")


if __name__ == "__main__":
    main()
