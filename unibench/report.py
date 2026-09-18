"""Generates the final cross-domain report: leaderboard.csv, a Pareto /
Friedman summary, an accuracy-vs-cost chart, and one combined report.html
-- the "Reporting Dashboard" stage of the architecture in the literature
review (Figure 1).

`compute_report_data()` is the pure, file-write-free core (summary table +
Pareto flags + composite score + Friedman test, as JSON-serializable
records) -- both the CLI's `generate_report()` AND the web dashboard's
`GET /api/results/summary` endpoint call it, so the two surfaces can never
silently drift apart into different numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed to save PNGs
import matplotlib.pyplot as plt
import pandas as pd

from .evaluation.aggregator import (
    AUDIT_COLUMNS, build_summary_table, summarize_per_model, pareto_optimal_models,
    composite_quick_glance_score,
    friedman_test,
)
from .evaluation.bias_calibration import compute_bias_corrections


def _load_raw_results(results_dir: Path) -> List[dict]:
    path = results_dir / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python run_benchmark.py run` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(results_dir: Path) -> List[dict]:
    path = results_dir / "human_calibration.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _json_safe(value):
    """Bare NaN is not valid JSON (Starlette's JSONResponse explicitly
    rejects it, allow_nan=False) and will 500 the API / fail
    `fetch().json()` in the browser -- convert to None instead.

    NOTE: assigning None into a pandas DataFrame's float64 column gets
    silently re-cast back to NaN by pandas, so this cleanup can't happen
    via `DataFrame.where(...)` before `.to_dict()` -- it has to run on the
    plain Python dicts AFTER `.to_dict("records")`, which is why this is a
    scalar helper applied post-conversion, not a DataFrame method."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame) -> List[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN -> None)."""
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict("records")]


def compute_report_data(results_dir: Path) -> dict:
    """Returns everything the dashboard needs, computed fresh from
    raw_results.json + human_calibration.csv (if present) -- no file writes.
    Safe to call on every dashboard page load; recomputation is cheap
    (no API calls, just pandas over already-collected data)."""
    results_dir = Path(results_dir)
    raw_results = _load_raw_results(results_dir)
    calibration_rows = _load_calibration(results_dir)
    corrections = compute_bias_corrections(calibration_rows) if calibration_rows else {}

    judge_names = sorted({jr["judge_name"] for r in raw_results for jr in r.get("judge_records", [])})

    long_df = build_summary_table(raw_results, corrections)
    counts = (long_df.groupby(["model", "task"])
              .agg(n_items=("item_id", "count"), n_empty_items=("empty_output", "sum"),
                   n_valid_auto=("auto_valid", "sum"), n_judge_scores=("n_judge_scores", "sum"))
              .astype(int).reset_index())
    summary = summarize_per_model(long_df)
    summary = summary.drop(columns=AUDIT_COLUMNS)
    summary = summary.sort_values(["task", "model"]).reset_index(drop=True)

    judge_audit: Dict[str, dict] = {}
    for r in raw_results:
        for jr in r.get("judge_records", []):
            entry = judge_audit.setdefault(jr["judge_name"], {"judge": jr["judge_name"], "n_calls": 0,
                                                              "n_call_failed": 0, "n_unparsed": 0})
            entry["n_calls"] += 1
            if jr["score"] is None:
                # Older raw results lack `call_failed`; ai_judge marks API failures in the reason.
                failed = jr.get("call_failed", jr["reason"].startswith("(judge call failed"))
                entry["n_call_failed" if failed else "n_unparsed"] += 1

    pareto_flags, composite_vals, friedman_results = [], [], []
    for task in summary["task"].unique():
        task_slice = summary[summary["task"] == task].reset_index(drop=True)
        pareto = set(pareto_optimal_models(task_slice))
        composite = composite_quick_glance_score(task_slice)
        for i in range(len(task_slice)):
            pareto_flags.append(task_slice.iloc[i]["model"] in pareto)
            composite_vals.append(composite.iloc[i])
        f = friedman_test(long_df, task, "avg_judge_score")
        if f:
            friedman_results.append(f)
    summary["pareto_optimal"] = pareto_flags
    summary["composite_quick_glance"] = composite_vals
    summary = summary.merge(counts, on=["model", "task"], how="left")

    corrections_json = {
        name: {"method": c.method, "n_points": c.n_points, "detail": c.detail}
        for name, c in corrections.items()
    }
    for name in judge_names:
        corrections_json.setdefault(name, {
            "method": "uncalibrated", "n_points": 0,
            "detail": "NO human calibration data -- raw, unchecked AI-judge scores used as-is.",
        })

    return {
        "n_records": len(raw_results),
        "n_errors": sum(1 for r in raw_results if r.get("error")),
        "n_empty_records": int(long_df["empty_output"].sum()),
        "judge_audit": list(judge_audit.values()),
        "judge_names": judge_names,
        "corrections": corrections_json,
        "summary": _records(summary),
        "friedman": [{k: _json_safe(v) for k, v in f.items()} for f in friedman_results],
        "tasks": sorted(summary["task"].unique().tolist()),
        "models": sorted(summary["model"].unique().tolist()),
    }


def _make_chart(summary: pd.DataFrame, out_path: Path) -> None:
    tasks = sorted(summary["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_df = summary[summary["task"] == task]
        for _, row in task_df.iterrows():
            ax.scatter(row["cost_usd"], row.get("avg_judge_score", float("nan")), s=90)
            ax.annotate(row["model"], (row["cost_usd"], row.get("avg_judge_score", 0)),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_title(f"Task: {task}")
        ax.set_xlabel("Avg. estimated cost per query (USD)")
        ax.set_ylabel("Bias-corrected AI-judge score (0-10)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Accuracy--Efficiency Trade-off (Pareto view)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _corrections_note(corrections: Dict[str, dict]) -> str:
    lines = []
    for name, c in corrections.items():
        lines.append(f"- **{name}**: {c['detail']}")
    return "\n".join(lines)


def generate_report(results_dir: Path) -> None:
    """CLI entrypoint: computes the same data as `compute_report_data()`,
    then additionally writes leaderboard.csv, the PNG chart, report.md, and
    report.html to disk."""
    results_dir = Path(results_dir)
    data = compute_report_data(results_dir)
    summary = pd.DataFrame(data["summary"])

    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "leaderboard.csv", index=False)
    _make_chart(summary, results_dir / "accuracy_vs_cost.png")

    md = ["# UniBench-NLP Report\n"]
    md.append("## Judge bias calibration\n")
    md.append(_corrections_note(data["corrections"]) or "_No judges found._")
    md.append("\n\n## Data completeness audit\n")
    for row in data["summary"]:
        md.append(f"- **{row['model']}** / {row['task']}: {row['n_items']} items, "
                  f"{row['n_empty_items']} with an empty response, "
                  f"{row['n_valid_auto']} with every automatic metric defined, "
                  f"{row['n_judge_scores']} parsed judge scores")
    for ja in data["judge_audit"]:
        md.append(f"- judge **{ja['judge']}**: of {ja['n_calls']} calls, {ja['n_call_failed']} failed at the API "
                  f"and {ja['n_unparsed']} returned a verdict that could not be parsed")
    md.append("\n\n## Per-task results\n")
    for task in data["tasks"]:
        task_df = summary[summary["task"] == task]
        md.append(f"### Task: {task}\n")
        md.append(task_df.drop(columns=["task"]).to_markdown(index=False))
        pareto_models = task_df[task_df["pareto_optimal"]]["model"].tolist()
        md.append(f"\n**Pareto-optimal model(s) for `{task}`:** {', '.join(pareto_models)}\n")
    md.append("\n## Statistical significance (Friedman test, blocked by item)\n")
    if data["friedman"]:
        for f in data["friedman"]:
            if f.get("note"):
                md.append(f"- **{f['task']}** ({f['metric']}, n_items={f['n_items']}, "
                          f"n_models={f['n_models']}): {f['note']}.")
                continue
            warn = " [LOW STATISTICAL POWER -- few items, treat as directional only]" if f["low_power_warning"] else ""
            md.append(
                f"- **{f['task']}** ({f['metric']}, n_items={f['n_items']}, n_models={f['n_models']}): "
                f"chi2={f['statistic']:.3f}, p={f['p_value']:.4f} "
                f"({'significant' if f['significant_at_0.05'] else 'not significant'} at 0.05).{warn}"
            )
    else:
        md.append("_Not enough items/models yet for a meaningful Friedman test "
                   "(need >= 3 models and >= 3 fully-scored items per task)._")
    md.append("\n![accuracy vs cost](accuracy_vs_cost.png)\n")

    report_md = "\n".join(md)
    (results_dir / "report.md").write_text(report_md, encoding="utf-8")

    html = (
        "<html><head><meta charset='utf-8'><title>UniBench-NLP Report</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.5}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
        "</head><body>"
        + summary.to_html(index=False)
        + "<h2>Full markdown report</h2><pre>" + report_md.replace("<", "&lt;") + "</pre>"
        + "<img src='accuracy_vs_cost.png' style='max-width:100%'>"
        + "</body></html>"
    )
    (results_dir / "report.html").write_text(html, encoding="utf-8")

    print(report_md)
    print(f"\nSaved: {results_dir/'leaderboard.csv'}, {results_dir/'report.md'}, "
          f"{results_dir/'report.html'}, {results_dir/'accuracy_vs_cost.png'}")
