"""Checks the property the Pareto rule depends on: missing data is never an advantage.

Algorithm 1 in the paper treats missing values asymmetrically. When the model
being tested for domination lacks a value on a criterion, its rival is counted
as better on that criterion; when the rival lacks one, the rival cannot
dominate at all; when neither has one, the criterion is skipped. Written out
that asymmetry looks arbitrary, so this script verifies what it is there to
guarantee:

    Replacing any model's value with "missing" can never remove it from the
    set of dominated models.

The earlier implementation skipped a criterion whenever either side lacked a
value, which violated this: a model could be rescued from domination by having
no number where its rival had a bad one. That is how qwen-3.6-27b reached the
consistency-probe frontier on a tone-word gap averaged over the five of ten
pairs it answered, against rivals scored over all ten.

Usage:
    python scripts/test_pareto_missing.py [--trials 20000] [--seed 0]

Exits non-zero if a counterexample is found, and prints it.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unibench.evaluation.aggregator import pareto_optimal_models  # noqa: E402

VALUES = [1.0, 2.0, 3.0, float("nan")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    checked = rescued = 0
    for _ in range(args.trials):
        n_models = rng.randint(2, 5)
        cols = [f"metric{j}" for j in range(rng.randint(1, 4))]
        table = {"model": [f"M{i}" for i in range(n_models)]}
        for c in cols:
            table[c] = [rng.choice(VALUES) for _ in range(n_models)]
        df = pd.DataFrame(table)

        present = [(i, c) for i in range(n_models) for c in cols if pd.notna(df.loc[i, c])]
        if not present:
            continue
        i, c = rng.choice(present)
        name = df.loc[i, "model"]

        before = set(pareto_optimal_models(df))
        blanked = df.copy()
        blanked.loc[i, c] = float("nan")
        after = set(pareto_optimal_models(blanked))
        checked += 1

        # The model that LOST a value must not gain frontier membership.
        if name not in before and name in after:
            rescued += 1
            print("COUNTEREXAMPLE: blanking a value rescued a dominated model\n")
            print(df.to_string(index=False))
            print(f"\nfrontier before: {sorted(before)}")
            print(f"blanked {name} on {c}")
            print(f"frontier after : {sorted(after)}")
            sys.exit(1)

    print(f"{checked} tables checked, {rescued} counterexamples found")
    print("property holds: making a value missing never rescues a dominated model")


if __name__ == "__main__":
    main()
