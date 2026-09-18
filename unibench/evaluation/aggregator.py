"""Multi-criteria aggregation: builds the per-(model, task) summary table,
finds the Pareto-optimal model set per task (accuracy/quality vs. cost vs.
latency vs. fairness-divergence, jointly), and runs a Friedman test as a
significance check -- all addressing Gap 1 from the literature review
("no single winner, and no principled way to say so").

Two columns are ALWAYS lower-is-better regardless of task: `cost_usd` and
`latency_s`. Task-specific "lower is better" columns are declared in
`LOWER_IS_BETTER_METRICS` below; everything else numeric is treated as
higher-is-better.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .bias_calibration import JudgeCorrection, get_correction

LOWER_IS_BETTER_METRICS = {
    "cost_usd", "latency_s", "tokens",
    "response_divergence", "length_asymmetry", "tone_gap",
    # Reliability is a ranking criterion, not bookkeeping. Averaged over a
    # model's items this is its empty-response rate, and without it a model
    # that fails to answer can be dominant on the items it did answer.
    "empty_output",
}

# A quality metric averaged over a model's valid items only is not comparable
# with the same metric averaged over every item of a rival: the failed items
# are missing precisely because the model failed. Below this fraction of
# items the mean is withheld (set to NaN) rather than reported, and a
# withheld metric loses its Pareto comparison -- see pareto_optimal_models.
MIN_METRIC_COVERAGE = 0.8

# Per-item bookkeeping columns, never compared as metrics. `empty_output` is
# deliberately NOT here any more; it is a metric (see above). The raw judge
# score is kept for reference only; the calibrated score enters the comparison.
AUDIT_COLUMNS = ["n_judge_scores", "auto_valid"]
NON_METRIC_COLUMNS = {"model", "task", "item_id", "avg_judge_score_raw", *AUDIT_COLUMNS}

MIN_ITEMS_FOR_FRIEDMAN = 3
MIN_MODELS_FOR_FRIEDMAN = 3


def build_summary_table(raw_results: List[dict],
                         corrections: Optional[Dict[str, JudgeCorrection]] = None) -> pd.DataFrame:
    """raw_results: list of per-(model, task, item) dicts produced by
    runner.py, each holding RAW (uncorrected) `judge_records`. `corrections`
    (from `bias_calibration.compute_bias_corrections`, or None/empty to skip
    calibration entirely) is applied here so both raw and bias-corrected
    judge scores end up in the output. Returns a long-form per-item
    DataFrame (for Friedman/plotting) -- call
    `.groupby(["model","task"]).mean(numeric_only=True)` on the result for
    the per-(model,task) summary table."""
    corrections = corrections or {}
    rows = []
    for r in raw_results:
        row = {
            "model": r["model"],
            "task": r["task"],
            "item_id": r["item_id"],
            "cost_usd": r["cost_usd"],
            "latency_s": r["latency_s"],
            "tokens": r["tokens"],
        }
        raw_scores, corrected_scores = [], []
        for jr in r.get("judge_records", []):
            if jr["score"] is None:
                continue
            raw_scores.append(jr["score"])
            correction = get_correction(corrections, jr["judge_name"])
            corrected_scores.append(correction.apply(jr["score"]))
        row["avg_judge_score_raw"] = float(np.mean(raw_scores)) if raw_scores else np.nan
        row["avg_judge_score"] = float(np.mean(corrected_scores)) if corrected_scores else np.nan
        row["n_judge_scores"] = float(len(raw_scores))
        row["empty_output"] = float(any(not (t or "").strip() for t in r.get("outputs", {}).values()))
        automatic = r.get("automatic", {})
        row["auto_valid"] = float(bool(automatic) and all(pd.notna(v) for v in automatic.values()))
        row.update(automatic)
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_columns(df_task: pd.DataFrame) -> List[str]:
    numeric_cols = df_task.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in NON_METRIC_COLUMNS and df_task[c].notna().any()]


def summarize_per_model(long_df: pd.DataFrame,
                         min_coverage: float = MIN_METRIC_COVERAGE) -> pd.DataFrame:
    """Per-(model, task) means, with every metric mean withheld when it rests
    on too few of that model's items.

    Taking the mean of whatever happens to be present silently changes the
    question a metric answers: qwen-3.6-27b's fairness tone-word gap of 0.00
    was the mean over the 5 of 10 pairs it answered, and it beat rivals whose
    0.20 and 0.30 covered all 10. Withholding the mean below `min_coverage`
    keeps a partial average from being compared against a complete one; the
    Pareto comparison then treats the withheld value as the worse side.
    `empty_output` is exempt because its whole purpose is to count the
    failures, and the item count columns are exempt because they are counts.
    """
    summary = long_df.groupby(["model", "task"]).mean(numeric_only=True).reset_index()
    counts = long_df.groupby(["model", "task"]).size().rename("n_items")
    exempt = {"empty_output", *AUDIT_COLUMNS}
    metric_cols = [c for c in summary.select_dtypes(include=[np.number]).columns
                   if c not in NON_METRIC_COLUMNS and c not in exempt]
    for col in metric_cols:
        present = long_df.groupby(["model", "task"])[col].count()
        coverage = (present / counts).reindex(
            summary.set_index(["model", "task"]).index).to_numpy()
        summary.loc[coverage < min_coverage, col] = np.nan
    return summary


def pareto_optimal_models(summary_task: pd.DataFrame,
                           skip_missing: bool = False) -> List[str]:
    """summary_task: one row per model (already averaged over items) for a
    SINGLE task. Returns the list of model names on the Pareto frontier
    (no other model beats them on every relevant metric at once).

    Missing values are NOT free. An earlier version skipped any metric where
    either model was NaN, which let a model that failed to produce measurable
    output win the comparison twice over: it was judged on fewer criteria
    than its rivals, and the criteria it was judged on were averaged over
    only the items it happened to complete. In the accompanying paper that
    put qwen-3.6-27b on the fairness frontier on the strength of a tone-word
    gap of 0.00 computed over the 5 of 10 pairs it answered, against 0.20 and
    0.30 computed over all 10 -- the model was rewarded for the items it
    failed. A metric a model could not produce is therefore treated as worse
    than any value a rival did produce, which is what "this model did not
    give us a number here" actually means for someone choosing between them.

    Pass skip_missing=True to reproduce the old ignore-NaN behavior for a
    sensitivity analysis; it is never the default.
    """
    metric_cols = _metric_columns(summary_task)
    if not metric_cols:
        return list(summary_task["model"])

    optimal = []
    for i, row_i in summary_task.iterrows():
        dominated = False
        for j, row_j in summary_task.iterrows():
            if row_i["model"] == row_j["model"]:
                continue
            at_least_as_good_everywhere = True
            strictly_better_somewhere = False
            for col in metric_cols:
                vi, vj = row_i[col], row_j[col]
                mi, mj = pd.isna(vi), pd.isna(vj)
                if mi and mj:
                    continue
                if mi or mj:
                    if skip_missing:
                        continue
                    # Whichever side has no value is the worse side here.
                    if mi:
                        strictly_better_somewhere = True
                    else:
                        at_least_as_good_everywhere = False
                        break
                    continue
                lower_better = col in LOWER_IS_BETTER_METRICS
                j_better = (vj < vi) if lower_better else (vj > vi)
                j_worse = (vj > vi) if lower_better else (vj < vi)
                if j_worse:
                    at_least_as_good_everywhere = False
                    break
                if j_better:
                    strictly_better_somewhere = True
            if at_least_as_good_everywhere and strictly_better_somewhere:
                dominated = True
                break
        if not dominated:
            optimal.append(row_i["model"])
    return optimal


def composite_quick_glance_score(summary_task: pd.DataFrame) -> pd.Series:
    """A SECONDARY, min-max-normalized single-number score per model for a
    quick-glance sort order. This deliberately averages away trade-offs --
    it is NOT the primary comparison method; the Pareto set above is. Kept
    because a fully unordered Pareto set is sometimes hard to skim."""
    metric_cols = _metric_columns(summary_task)
    if not metric_cols:
        return pd.Series([np.nan] * len(summary_task), index=summary_task.index)

    normalized = pd.DataFrame(index=summary_task.index)
    for col in metric_cols:
        values = summary_task[col]
        span = values.max() - values.min()
        if span == 0 or pd.isna(span):
            normalized[col] = 0.5
            continue
        norm = (values - values.min()) / span
        normalized[col] = (1 - norm) if col in LOWER_IS_BETTER_METRICS else norm
    return normalized.mean(axis=1)


def friedman_test(long_df: pd.DataFrame, task: str, metric_col: str = "avg_judge_score") -> Optional[dict]:
    """Non-parametric significance test (following PSE-Bench's methodology)
    for whether models differ significantly on `metric_col` for `task`,
    blocked by item. Returns None (rather than a misleading result) if
    there isn't enough data for the test to be meaningful, and always
    reports sample size alongside the p-value so a small-sample result
    isn't mistaken for a well-powered one."""
    task_df = long_df[long_df["task"] == task]
    pivot = task_df.pivot_table(index="item_id", columns="model", values=metric_col)
    pivot = pivot.dropna(axis=0, how="any")  # Friedman needs complete blocks

    n_items, n_models = pivot.shape
    if n_models < MIN_MODELS_FOR_FRIEDMAN or n_items < MIN_ITEMS_FOR_FRIEDMAN:
        return None

    # friedmanchisquare divides by a between-group variance term that is
    # zero when every model scored identically on every item (e.g. all
    # judge scores failed to parse and fell back to the same default) --
    # that's a real, reportable outcome ("no detectable difference"), not a
    # crash, so it's caught explicitly rather than let scipy emit a raw
    # RuntimeWarning and a silent NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            statistic, p_value = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        except RuntimeWarning:
            return {
                "task": task, "metric": metric_col, "n_items": n_items, "n_models": n_models,
                "statistic": float("nan"), "p_value": float("nan"),
                "significant_at_0.05": False, "low_power_warning": True,
                "note": "undefined -- identical scores across all models/items, no variance to test",
            }

    return {
        "task": task, "metric": metric_col, "n_items": n_items, "n_models": n_models,
        "statistic": float(statistic), "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "low_power_warning": n_items < 8,  # MVP-scale item counts -> flag explicitly
        "note": None,
    }
