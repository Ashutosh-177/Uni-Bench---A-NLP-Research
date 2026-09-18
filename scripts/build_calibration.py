"""Rebuilds the judge-calibration pairs from a chosen set of human raters.

`run_benchmark.py calibrate` writes the primary rater to human_ratings.csv and
any `--rater NAME` to human_ratings_NAME.csv, and only the primary file feeds
judge calibration. That is the wrong default once independent annotators are
involved: the people best placed to calibrate the judges are the ones who did
not build the benchmark, and they arrive as `--rater` files.

This script makes any set of raters the calibration reference. It takes the
consensus (mean) human score per output across the named raters, pairs it with
every parsed judge verdict for that output, and rewrites
results/human_calibration.csv, which is what compute_bias_corrections reads.

Usage:
    python scripts/build_calibration.py --raters alice,bob
    python scripts/build_calibration.py --raters alice,bob --dry-run
    python scripts/build_calibration.py --list

A rater named `primary` (or `default`) means the unsuffixed human_ratings.csv.
The previous calibration file is backed up before being replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIELDNAMES = ["model", "task", "item_id", "judge_name", "ai_score", "ai_reason", "human_score"]
PRIMARY_ALIASES = {"primary", "default", "human_ratings"}


def rater_file(results: Path, name: str) -> Path:
    if name.lower() in PRIMARY_ALIASES:
        return results / "human_ratings.csv"
    safe = "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")
    return results / f"human_ratings_{safe}.csv"


def load_ratings(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["model"], row["task"], row["item_id"])] = float(row["human_score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def icc_2_1(matrix: list[list[float]]) -> float:
    """ICC(2,1) for n subjects x k raters, same estimator paper_stats.py uses."""
    n, k = len(matrix), len(matrix[0])
    if n < 2 or k < 2:
        return float("nan")
    grand = statistics.fmean(v for row in matrix for v in row)
    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean(row[j] for row in matrix) for j in range(k)]
    msr = k * sum((rm - grand) ** 2 for rm in row_means) / (n - 1)
    msc = n * sum((cm - grand) ** 2 for cm in col_means) / (k - 1)
    mse = sum((matrix[i][j] - row_means[i] - col_means[j] + grand) ** 2
              for i in range(n) for j in range(k)) / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denom if denom else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", help="Comma-separated rater names to use as the reference.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--list", action="store_true", help="List the rater files available.")
    args = ap.parse_args()

    results = ROOT / args.results
    available = sorted(p.name for p in results.glob("human_ratings*.csv"))
    if args.list or not args.raters:
        print("Rater files in", results)
        for name in available:
            n = len(load_ratings(results / name))
            label = "primary" if name == "human_ratings.csv" else name[len("human_ratings_"):-4]
            print(f"  {name:38s} {n:>4} ratings   (--raters {label})")
        if not args.raters:
            sys.exit("\nPass --raters NAME[,NAME...] to rebuild the calibration pairs.")
        return

    names = [n.strip() for n in args.raters.split(",") if n.strip()]
    per_rater = {}
    for name in names:
        path = rater_file(results, name)
        if not path.exists():
            sys.exit(f"no such rater file: {path.name}  (available: {', '.join(available) or 'none'})")
        per_rater[name] = load_ratings(path)
        print(f"  {name:16s} {len(per_rater[name]):>4} ratings from {path.name}")

    keys = set.intersection(*(set(r) for r in per_rater.values())) if per_rater else set()
    union = set().union(*(set(r) for r in per_rater.values())) if per_rater else set()
    print(f"\noutputs rated by all {len(names)} rater(s): {len(keys)}; by at least one: {len(union)}")

    if len(names) > 1 and keys:
        ordered = sorted(keys)
        matrix = [[per_rater[n][k] for n in names] for k in ordered]
        print(f"inter-rater ICC(2,1) over the shared {len(ordered)}: {icc_2_1(matrix):.3f}")

    # Consensus over whoever rated each output, so a partially-rated set is
    # still usable; n_raters is recorded per pair for the write-up.
    consensus = {k: statistics.fmean([r[k] for r in per_rater.values() if k in r]) for k in union}

    raw = json.loads((results / "raw_results.json").read_text(encoding="utf-8"))
    rows, judges = [], {}
    for record in raw:
        key = (record["model"], record["task"], record["item_id"])
        if key not in consensus:
            continue
        for jr in record.get("judge_records", []):
            if jr.get("score") is None:
                continue
            judges[jr["judge_name"]] = judges.get(jr["judge_name"], 0) + 1
            rows.append({
                "model": key[0], "task": key[1], "item_id": key[2],
                "judge_name": jr["judge_name"], "ai_score": jr["score"],
                "ai_reason": (jr.get("reason") or "")[:200],
                "human_score": round(consensus[key], 3),
            })

    print(f"\ncalibration pairs: {len(rows)}")
    for judge, n in sorted(judges.items()):
        print(f"  {judge:20s} {n:>4} pairs")
    if not rows:
        sys.exit("no pairs produced -- do the rated outputs match raw_results.json?")

    out = results / "human_calibration.csv"
    if args.dry_run:
        print(f"\n--dry-run: {out.name} not written")
        return
    if out.exists():
        backup = out.with_name(f"human_calibration.backup-{datetime.now():%Y%m%d-%H%M%S}.csv")
        shutil.copy2(out, backup)
        print(f"\nbacked up existing pairs to {backup.name}")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.name}  --  re-run `python run_benchmark.py report` to apply it")


if __name__ == "__main__":
    main()
