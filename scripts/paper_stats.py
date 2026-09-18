"""Computes every number reported in the paper from one results directory.

Usage:  python scripts/paper_stats.py [results_dir]   (default: results)

Reads raw_results.json, human_calibration.csv and human_ratings.csv and
prints the values used in the tables, findings, and significance tests.
Writes nothing.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unibench.evaluation.aggregator import build_summary_table, pareto_optimal_models  # noqa: E402
from unibench.evaluation.bias_calibration import (  # noqa: E402
    compute_bias_corrections,
    _leave_one_out_mae,
)

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen-3.6-27b"]
# Groq paid-tier output-token prices (USD per token) used for the illustrative cost table.
OUTPUT_PRICE = {"qwen-3.6-27b": 3.00e-6, "gpt-oss-120b": 0.60e-6, "gpt-oss-20b": 0.30e-6}
AUTO_METRICS = {"fairness": ["response_divergence", "length_asymmetry", "tone_gap"],
                "summarization": ["rouge1_f", "rouge2_f", "rougeL_f"]}
TYPOGRAPHIC = "‘’“”–—"

raw = json.loads((RESULTS / "raw_results.json").read_text(encoding="utf-8"))
cal_path, hr_path = RESULTS / "human_calibration.csv", RESULTS / "human_ratings.csv"
cal_rows = pd.read_csv(cal_path).to_dict("records") if cal_path.exists() else []
human = pd.read_csv(hr_path) if hr_path.exists() else pd.DataFrame(columns=["model", "task", "item_id", "human_score"])
corrections = compute_bias_corrections(cal_rows) if cal_rows else {}
long_df = build_summary_table(raw, corrections)
tasks = sorted(long_df["task"].unique())
judges = sorted({jr["judge_name"] for r in raw for jr in r["judge_records"]})


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def is_empty(text) -> bool:
    return not (text or "").strip()


def exact_friedman_p(matrix: np.ndarray) -> tuple[int, int]:
    """Exact permutation test of the Friedman statistic: the model labels are
    permuted independently within each block (item). Returns (count of
    permutations with a statistic >= the observed one, total permutations).
    Ties stay tied under permutation, so comparing sum(R_j^2) is equivalent."""
    k = matrix.shape[1]
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    observed = float((ranks.sum(axis=0) ** 2).sum())
    dist = {tuple([0.0] * k): 1}
    for row in ranks:
        perms = [tuple(row[list(p)]) for p in itertools.permutations(range(k))]
        nxt = defaultdict(int)
        for state, count in dist.items():
            for perm in perms:
                nxt[tuple(round(a + b, 1) for a, b in zip(state, perm))] += count
        dist = nxt
    total = math.factorial(k) ** len(ranks)
    at_least = sum(c for s, c in dist.items() if sum(x * x for x in s) >= observed - 1e-9)
    return at_least, total


def friedman_report(pivot: pd.DataFrame, label: str) -> None:
    pivot = pivot.dropna()
    n, k = pivot.shape
    if n < 3 or k < 3:
        print(f"{label}: not enough complete items ({n})")
        return
    chi2, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    at_least, total = exact_friedman_p(pivot.values)
    print(f"{label}: n={n} chi2(2)={chi2:.3f} p_asym={p:.4f} W={chi2 / (n * (k - 1)):.3f} "
          f"p_exact={at_least}/{total}={at_least / total:.4f}  mean ranks={dict(pivot.rank(axis=1).mean().round(2))}")


section("Records and empty outputs")
for t in tasks:
    for m in MODELS:
        recs = [r for r in raw if r["model"] == m and r["task"] == t]
        calls = sum(len(r["outputs"]) for r in recs)
        empty_calls = sum(is_empty(v) for r in recs for v in r["outputs"].values())
        empty_items = sum(any(is_empty(v) for v in r["outputs"].values()) for r in recs)
        print(f"{t:14s} {m:14s} items={len(recs)} calls={calls} empty_calls={empty_calls} "
              f"items_with_empty={empty_items} api_errors={sum(bool(r['error']) for r in recs)}")

section("Judge verdicts that could not be parsed")
for j in judges:
    by_subject = defaultdict(lambda: [0, 0])
    for r in raw:
        for jr in r["judge_records"]:
            if jr["judge_name"] == j:
                by_subject[r["model"]][0] += 1
                by_subject[r["model"]][1] += jr["score"] is None
    total = sum(v[0] for v in by_subject.values())
    failed = sum(v[1] for v in by_subject.values())
    api_failed = sum(1 for r in raw for jr in r["judge_records"] if jr["judge_name"] == j and jr["score"] is None
                     and jr.get("call_failed", jr["reason"].startswith("(judge call failed")))
    print(f"{j:14s} {failed}/{total} without a score ({api_failed} API failures, {failed - api_failed} unparseable); "
          f"by subject model: " + ", ".join(f"{s} {v[1]}/{v[0]}" for s, v in sorted(by_subject.items())))

section("Unparsed verdicts vs typographic punctuation in the judged output")
for j in judges:
    counts = {True: [0, 0], False: [0, 0]}
    for r in raw:
        has = any(ch in " ".join(r["outputs"].values()) for ch in TYPOGRAPHIC)
        for jr in r["judge_records"]:
            if jr["judge_name"] == j:
                counts[has][0] += 1
                counts[has][1] += jr["score"] is None
    print(f"{j:14s} failed {counts[True][1]}/{counts[True][0]} with such punctuation, "
          f"{counts[False][1]}/{counts[False][0]} without")

section("Repeated judging (same judge, same call, repeated)")
repeat_path = RESULTS / "repeat_judging.csv"
if repeat_path.exists():
    rep = pd.read_csv(repeat_path)
    for judge_name, g in rep.groupby("judge"):
        ok = g[g["outcome"] == "ok"]
        print(f"{judge_name}: {len(g)} calls, {(g['outcome'] != 'ok').sum()} without a score "
              f"({', '.join(f'{k}={v}' for k, v in g['outcome'].value_counts().items())})")
        per_output = ok.groupby(["model", "task", "item_id"])["score"].agg(["mean", "std", "min", "max", "count"])
        print(f"{'':14s} outputs scored: {len(per_output)}; passes per output: "
              f"{per_output['count'].min()}-{per_output['count'].max()}")
        print(f"{'':14s} within-output SD across passes: mean={per_output['std'].mean():.3f}, "
              f"max={per_output['std'].max():.3f}; outputs where all passes agreed: "
              f"{(per_output['std'].fillna(0) == 0).sum()} of {len(per_output)}")
        print(f"{'':14s} mean score per model: "
              + ", ".join(f"{m}={v:.2f}" for m, v in per_output.groupby("model")["mean"].mean().items()))
        widest = per_output.assign(spread=per_output["max"] - per_output["min"]).nlargest(3, "spread")
        for idx, row in widest.iterrows():
            print(f"{'':14s} widest spread: {idx[0]}/{idx[2]} {row['min']:.1f}-{row['max']:.1f} (mean {row['mean']:.2f})")
else:
    print("no repeat_judging.csv yet")

section("Human ratings (blind)")
n_ratable = sum(1 for r in raw if not r["error"])
print(f"outputs rated: {len(human)} of {n_ratable}")
if len(human):
    print(human.groupby(["task", "model"])["human_score"].agg(["mean", "std", "count"]).round(2).to_string())
    for t in tasks:
        pivot = human[human["task"] == t].pivot_table(index="item_id", columns="model", values="human_score")
        friedman_report(pivot, f"human ratings, {t}")

section("Agreement between human raters")
for extra in sorted(RESULTS.glob("human_ratings_*.csv")):
    if "corrected" in extra.name:
        continue
    other = pd.read_csv(extra)
    both = human.merge(other, on=["model", "task", "item_id"], suffixes=("_1", "_2"))
    if len(both) < 3:
        print(f"{extra.name}: only {len(both)} outputs rated by both raters")
        continue
    y = both[["human_score_1", "human_score_2"]].to_numpy(float)
    n, k = y.shape
    grand = y.mean()
    ssr = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ssc = n * ((y.mean(axis=0) - grand) ** 2).sum()
    sse = ((y - grand) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    print(f"{extra.name}: n={n} ICC(2,1)={icc:.3f} Pearson r={stats.pearsonr(y[:, 0], y[:, 1])[0]:.3f} "
          f"Spearman rho={stats.spearmanr(y[:, 0], y[:, 1])[0]:.3f} "
          f"mean |difference|={np.abs(y[:, 0] - y[:, 1]).mean():.2f} "
          f"exact agreement={np.mean(y[:, 0] == y[:, 1]):.0%}")

section("Calibration per judge")
for j in judges:
    rows = [r for r in cal_rows if r["judge_name"] == j]
    if not rows:
        print(f"{j:14s} uncalibrated")
        continue
    c = corrections[j]
    a = np.array([r["ai_score"] for r in rows], float)
    h = np.array([r["human_score"] for r in rows], float)
    print(f"{j:14s} n={len(rows)} method={c.method} slope={c._slope:.3f} intercept={c._intercept:.3f} "
          f"mean(judge-human)={np.mean(a - h):+.2f}  maps 0..10 to {c.apply(0):.2f}..{c.apply(10):.2f}")
    if len(rows) >= 3 and a.std() > 0 and h.std() > 0:
        print(f"{'':14s} judge vs human: Pearson r={stats.pearsonr(a, h)[0]:.3f}, "
              f"Spearman rho={stats.spearmanr(a, h)[0]:.3f}")
    # The paper's calibration table reports the held-out error of the FITTED
    # correction -- the quantity the framework compares against the raw error to
    # decide whether to apply it -- so refit ungated on each fold, exactly as
    # bias_calibration does when it makes that decision.
    raw_mae = float(np.mean(np.abs(a - h)))
    loo_mae = _leave_one_out_mae([(x, y) for x, y in zip(a, h)])
    print(f"{'':14s} MAE vs human: raw={raw_mae:.2f}, fitted correction held out={loo_mae:.2f} "
          f"-> {'applied' if loo_mae < raw_mae else 'NOT applied'}")
    print(f"{'':14s} MAE as actually applied: "
          f"{np.mean([abs(c.apply(x) - y) for x, y in zip(a, h)]):.2f}")

section("Per-model results (judge score = calibrated mean +/- SD over items, ddof=1)")
for t in tasks:
    for m in MODELS:
        g = long_df[(long_df["task"] == t) & (long_df["model"] == m)]
        line = (f"{t:14s} {m:14s} judge={g['avg_judge_score'].mean():.2f}+/-{g['avg_judge_score'].std(ddof=1):.2f} "
                f"(raw {g['avg_judge_score_raw'].mean():.2f}; judged items={g['avg_judge_score'].notna().sum()}) "
                f"lat={g['latency_s'].mean():.2f}s tokens={g['tokens'].mean():.1f}")
        for col in AUTO_METRICS.get(t, []):
            if col in g:
                line += f" {col}={g[col].mean():.3f}(n={g[col].notna().sum()})"
        print(line)

section("Pareto-optimal models")
summary = long_df.groupby(["model", "task"]).mean(numeric_only=True).reset_index()
for t in tasks:
    print(f"{t:14s} {pareto_optimal_models(summary[summary['task'] == t].reset_index(drop=True))}")

section("Significance of the judge-score ranking (Friedman, blocked by item)")
for t in tasks:
    for col in ["avg_judge_score_raw", "avg_judge_score"]:
        pivot = long_df[long_df["task"] == t].pivot_table(index="item_id", columns="model", values=col)
        friedman_report(pivot, f"{t}, {col}")

section("Is the Friedman result carried by the model with empty outputs?")
# The Friedman test above ranks all three models. Qwen returned empty output on 7
# of its 16 items, which the judges score very low, so the omnibus result could be
# detecting that reliability failure rather than any difference in quality. Drop
# Qwen and test the two models that answered every item against each other with an
# exact sign test (no distributional assumption, and it handles the ties honestly).
for t in tasks:
    pivot = long_df[long_df["task"] == t].pivot_table(
        index="item_id", columns="model", values="avg_judge_score")
    both = pivot[["gpt-oss-120b", "gpt-oss-20b"]].dropna()
    diffs = (both["gpt-oss-120b"] - both["gpt-oss-20b"]).to_numpy()
    wins, losses, ties = int((diffs > 0).sum()), int((diffs < 0).sum()), int((diffs == 0).sum())
    decided = wins + losses
    p = stats.binomtest(wins, decided, 0.5).pvalue if decided else float("nan")
    print(f"{t:14s} 120b vs 20b over {len(both)} shared items: "
          f"{wins} win / {losses} loss / {ties} tie")
    print(f"{'':14s} exact sign test on the {decided} decided items: p={p:.3f}")
    ranks = pivot.rank(axis=1).mean()
    print(f"{'':14s} Friedman mean ranks (all three): "
          + ", ".join(f"{m}={ranks[m]:.2f}" for m in MODELS))

section("What the corrected pipeline would report (empty outputs never judged)")
# The framework now refuses to send an output with an empty response to the
# judges. The reported run predates that change, so its judge means include
# verdicts on empty outputs. Re-applying the exclusion rule to the STORED
# verdicts shows what the corrected pipeline would have reported. This is a
# sensitivity analysis, not a re-run: the verdicts it reuses were issued under
# the earlier context layout, so only the exclusion rule is being tested.
for t in tasks:
    rows, blocks = [], defaultdict(dict)
    for m in MODELS:
        recs = [r for r in raw if r["model"] == m and r["task"] == t]
        kept = []
        for r in recs:
            scored = [j["score"] for j in r["judge_records"] if j.get("score") is not None]
            if not scored:
                continue
            mean = float(np.mean(scored))
            if all((txt or "").strip() for txt in r["outputs"].values()):
                kept.append(mean)
                blocks[r["item_id"]][m] = mean
        rows.append((m, kept))
    for m, kept in rows:
        sd = f"{np.std(kept, ddof=1):.2f}" if len(kept) > 1 else "n/a"
        print(f"{t:14s} {m:14s} judge={np.mean(kept):.2f}+/-{sd} over {len(kept)} answered item(s)")
    complete = pd.DataFrame(
        [{"item_id": i, **d} for i, d in blocks.items() if len(d) == len(MODELS)]
    )
    if complete.empty:
        print(f"{'':14s} no complete blocks left")
        continue
    friedman_report(complete.set_index("item_id")[MODELS],
                    f"{t}, corrected pipeline (complete blocks only)")

section("Ablation: what each layer of the pipeline buys, against the independent rater")
# The framework claims a multi-judge panel beats a single judge and that
# calibration is worth doing. Neither is demonstrated by building them, so
# each configuration is scored against the independent human rater here.
_ref_path = RESULTS / "human_ratings_bob.csv"
_ext_path = RESULTS / "external_judge.csv"
if _ref_path.exists():
    ref = {(r["model"], r["task"], r["item_id"]): float(r["human_score"])
           for r in pd.read_csv(_ref_path).to_dict("records")}
    ext = {}
    if _ext_path.exists():
        for r in pd.read_csv(_ext_path).to_dict("records"):
            if r.get("outcome") == "ok" and r.get("judge") == "claude-sonnet-5":
                ext[(r["model"], r["task"], r["item_id"])] = float(r["score"])

    def _panel(judges=None, skip_empty=False, external=False):
        out = {}
        for rec in raw:
            key = (rec["model"], rec["task"], rec["item_id"])
            if skip_empty and any(is_empty(t) for t in rec["outputs"].values()):
                continue
            if external:
                if key in ext:
                    out[key] = ext[key]
                continue
            vals = [j["score"] for j in rec["judge_records"]
                    if j.get("score") is not None and (not judges or j["judge_name"] in judges)]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    human_rank = sorted(MODELS, key=lambda m: -np.mean([v for k, v in ref.items() if k[0] == m]))
    print(f"{'configuration':34s} {'n':>4} {'r':>7} {'MAE':>6}  ranking vs human")
    rows = [("single judge: gpt-oss-20b", _panel({"gpt-oss-20b"})),
            ("single judge: gpt-oss-120b", _panel({"gpt-oss-120b"})),
            ("single judge: qwen-3.6-27b", _panel({"qwen-3.6-27b"})),
            ("external judge alone", _panel(external=True)),
            ("panel of three", _panel()),
            ("panel + empty-output exclusion", _panel(skip_empty=True))]
    for label, sc in rows:
        keys = [k for k in sc if k in ref]
        if len(keys) < 3:
            continue
        x = np.array([sc[k] for k in keys]); y = np.array([ref[k] for k in keys])
        means = {m: np.mean([sc[k] for k in sc if k[0] == m]) for m in MODELS
                 if any(k[0] == m for k in sc)}
        rank = sorted(means, key=lambda m: -means[m])
        print(f"{label:34s} {len(keys):>4} {stats.pearsonr(x, y)[0]:>7.3f} {np.abs(x - y).mean():>6.2f}  "
              f"{' > '.join(rank)}  [{'matches' if rank == human_rank else 'DIFFERS'}]")

section("Synthetic bias injection: does the gate correct a bias that is really there?")
# All three judges here happen to be well calibrated, so the gate declines to
# correct any of them. That is the right call but a weak demonstration, since
# it never shows the correction working. Injecting a known offset does.
if cal_rows:
    cal_df = pd.DataFrame(cal_rows)
    print(f"{'injected':>9} {'judge':14s} {'measured':>9} {'method':>22} {'MAE raw':>8} {'MAE corr':>9} {'better':>7}")
    for bias in [0.0, -1.0, -2.0, 2.0]:
        for judge in ["gpt-oss-20b", "gpt-oss-120b"]:
            d = cal_df.copy()
            mask = d["judge_name"] == judge
            d.loc[mask, "ai_score"] = np.clip(d.loc[mask, "ai_score"] + bias, 0, 10)
            corr = compute_bias_corrections(d.to_dict("records"))[judge]
            sub = d[mask]
            a = sub["ai_score"].to_numpy(float); h = sub["human_score"].to_numpy(float)
            raw_mae = float(np.abs(a - h).mean())
            corr_mae = float(np.abs(np.array([corr.apply(v) for v in a]) - h).mean())
            print(f"{bias:>+9.1f} {judge:14s} {np.mean(a - h):>+9.2f} {corr.method:>22} "
                  f"{raw_mae:>8.2f} {corr_mae:>9.2f} {('YES' if corr_mae < raw_mae - 1e-9 else 'no'):>7}")
    print("  A positive offset on a judge already near the 10-point ceiling is clipped away,")
    print("  so it cannot be recovered -- a real limit of the correction, not of the gate.")

section("Clustering check: leave-one-item-out vs leave-one-pair-out calibration")
# Three judges score the same output and share its human rating, so calibration
# pairs are clustered by item. Each judge is fitted separately, though, so
# within one judge's fit every pair is already a distinct item -- worth
# verifying rather than assuming.
if cal_rows:
    for judge, g in pd.DataFrame(cal_rows).groupby("judge_name"):
        items = {(r["model"], r["task"], r["item_id"]) for r in g.to_dict("records")}
        print(f"  {judge:16s} {len(g):>3} pairs over {len(items):>3} distinct items "
              f"({'no clustering within this judge' if len(items) == len(g) else 'CLUSTERED'})")

section("Illustrative paid-tier cost per item and latency ratios")
for t in tasks:
    means = long_df[long_df["task"] == t].groupby("model")[["tokens", "latency_s"]].mean()
    cost = {m: means.loc[m, "tokens"] * OUTPUT_PRICE[m] for m in MODELS}
    print(f"{t:14s} cost/item: " + ", ".join(f"{m} ${cost[m]:.6f}" for m in MODELS))
    print(f"{'':14s} qwen cost ratio: vs 120b {cost['qwen-3.6-27b'] / cost['gpt-oss-120b']:.2f}x, "
          f"vs 20b {cost['qwen-3.6-27b'] / cost['gpt-oss-20b']:.2f}x; latency ratio: "
          f"vs 120b {means.loc['qwen-3.6-27b', 'latency_s'] / means.loc['gpt-oss-120b', 'latency_s']:.2f}x, "
          f"vs 20b {means.loc['qwen-3.6-27b', 'latency_s'] / means.loc['gpt-oss-20b', 'latency_s']:.2f}x")

section("Self-preference check (raw judge scores on a model's own outputs vs other judges)")
for j in judges:
    if j not in MODELS:
        continue
    own = [jr["score"] for r in raw if r["model"] == j for jr in r["judge_records"]
           if jr["judge_name"] == j and jr["score"] is not None]
    other = [jr["score"] for r in raw if r["model"] == j for jr in r["judge_records"]
             if jr["judge_name"] != j and jr["score"] is not None]
    if own and other:
        print(f"{j:14s} self {np.mean(own):.2f} (n={len(own)}), other judges {np.mean(other):.2f} "
              f"(n={len(other)}), difference {np.mean(own) - np.mean(other):+.2f}")

section("Response length vs response divergence (fairness pairs with a defined divergence)")
lengths, divergences = [], []
for r in raw:
    d = r.get("automatic", {}).get("response_divergence")
    if r["task"] == "fairness" and d is not None and not pd.isna(d):
        lengths.append((len(r["outputs"]["variant_a"]) + len(r["outputs"]["variant_b"])) / 2)
        divergences.append(d)
if len(lengths) >= 3:
    rho, p = stats.spearmanr(lengths, divergences)
    print(f"n={len(lengths)} Spearman rho={rho:.3f} p={p:.3f}")

section("Qualitative examples (fairness)")
for m, item_id in [("gpt-oss-120b", "gender_engineer"), ("qwen-3.6-27b", "name_hiring")]:
    r = next((x for x in raw if x["model"] == m and x["item_id"] == item_id), None)
    if r is None:
        continue
    print(f"{m} / {item_id}: divergence={r['automatic'].get('response_divergence')}")
    for jr in r["judge_records"]:
        print(f"   judge {jr['judge_name']}: {jr['score']} -- {jr['reason']}")
    for variant, text in r["outputs"].items():
        print(f"   [{variant}] {text}")
fair = [(r["automatic"]["response_divergence"], r["model"], r["item_id"]) for r in raw
        if r["task"] == "fairness" and not pd.isna(r["automatic"].get("response_divergence", np.nan))]
print("lowest divergence pairs:", sorted(fair)[:3])
print("highest divergence pairs:", sorted(fair)[-3:])
