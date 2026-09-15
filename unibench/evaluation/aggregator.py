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
}

# Per-item bookkeeping columns: summed into valid-item counts by the report,
# never compared as metrics. The raw judge score is kept for reference only;
# the calibrated score is the one that enters the Pareto comparison.
AUDIT_COLUMNS = ["n_judge_scores", "empty_output", "auto_valid"]
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


def pareto_optimal_models(summary_task: pd.DataFrame) -> List[str]:
    """summary_task: one row per model (already averaged over items) for a
    SINGLE task. Returns the list of model names on the Pareto frontier
    (no other model beats them on every relevant metric at once)."""
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
                if pd.isna(vi) or pd.isna(vj):
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
