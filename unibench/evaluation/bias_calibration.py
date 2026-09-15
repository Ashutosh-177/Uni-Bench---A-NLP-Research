"""Bias-calibration module.

PSE-Bench (Sukpancharoen & Srinophakun, 2026, reviewed in the literature
review) found AI judges run about +0.85 points too lenient compared with
human raters -- and, crucially, they only found this out because they
scored a human-rated subset and checked. This module generalizes that
check: for every judge model, it fits a correction from a small
human-rated calibration subset (collected via `run_benchmark.py calibrate`)
and applies it to every AI-judge score before the final report.

- With >= MIN_POINTS_FOR_REGRESSION calibration points for a judge, fits a
  simple linear regression (ai_score -> human_score) so it corrects both
  bias (offset) and scale (a judge that's not just lenient but also
  compresses its range gets corrected for that too).
- With fewer points, falls back to a constant mean-offset correction.
- Either way the fit is then VALIDATED by leave-one-out cross-validation
  and applied only if it lowers the mean absolute error against the human
  ratings. A judge that already agrees closely with its raters gains
  nothing from a fitted correction: the fit adds estimation noise instead
  of removing bias, which is what happened to all three judges in the
  accompanying paper. Such a judge is reported as measured-but-uncorrected,
  with the measured offset kept in the report.
- With zero points for a judge, applies NO correction and the report
  explicitly marks that judge's scores as "uncalibrated" -- silently
  trusting an unchecked judge is exactly the failure mode this project
  is trying to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

MIN_POINTS_FOR_REGRESSION = 5
MIN_POINTS_FOR_VALIDATION = 3


@dataclass
class JudgeCorrection:
    judge_name: str
    method: str            # "regression" | "mean_offset" | "uncalibrated"
    n_points: int
    detail: str             # human-readable summary, e.g. "+0.85 pt lenient"
    _slope: float = 1.0
    _intercept: float = 0.0

    def apply(self, ai_score: float) -> float:
        if ai_score is None:
            return None
        corrected = self._slope * ai_score + self._intercept
        return float(np.clip(corrected, 0.0, 10.0))


def _fit(pairs: List[tuple]) -> tuple:
    """Returns (slope, intercept, method) for one judge's (ai, human) pairs."""
    ai = np.array([p[0] for p in pairs])
    human = np.array([p[1] for p in pairs])
    if len(pairs) >= MIN_POINTS_FOR_REGRESSION:
        model = LinearRegression().fit(ai.reshape(-1, 1), human)
        return float(model.coef_[0]), float(model.intercept_), "regression"
    return 1.0, float(-np.mean(ai - human)), "mean_offset"


def _apply(slope: float, intercept: float, ai_score: float) -> float:
    return float(np.clip(slope * ai_score + intercept, 0.0, 10.0))


def _leave_one_out_mae(pairs: List[tuple]) -> float:
    """Mean absolute error of the fitted correction on held-out points: refit
    without each point, then score that point. This is what decides whether a
    correction is worth applying at all."""
    errors = []
    for i in range(len(pairs)):
        rest = pairs[:i] + pairs[i + 1:]
        slope, intercept, _ = _fit(rest)
        errors.append(abs(_apply(slope, intercept, pairs[i][0]) - pairs[i][1]))
    return float(np.mean(errors))


def compute_bias_corrections(calibration_rows: List[dict]) -> Dict[str, JudgeCorrection]:
    """calibration_rows: list of {"judge_name": str, "ai_score": float,
    "human_score": float} collected by the `calibrate` CLI step.
    Returns {judge_name: JudgeCorrection}."""
    by_judge: Dict[str, List[tuple]] = {}
    for row in calibration_rows:
        by_judge.setdefault(row["judge_name"], []).append(
            (float(row["ai_score"]), float(row["human_score"]))
        )

    corrections: Dict[str, JudgeCorrection] = {}
    for judge_name, pairs in by_judge.items():
        ai_scores = np.array([p[0] for p in pairs])
        human_scores = np.array([p[1] for p in pairs])
        mean_gap = float(np.mean(ai_scores - human_scores))

        slope, intercept, method = _fit(pairs)
        raw_mae = float(np.mean(np.abs(ai_scores - human_scores)))
        loo_mae = (_leave_one_out_mae(pairs) if len(pairs) >= MIN_POINTS_FOR_VALIDATION
                   else float("nan"))

        if loo_mae == loo_mae and loo_mae >= raw_mae:
            # The fit does not survive validation: keep the measurement, drop the correction.
            corrections[judge_name] = JudgeCorrection(
                judge_name=judge_name, method="measured_uncorrected", n_points=len(pairs),
                detail=f"measured {mean_gap:+.2f} pt offset from {len(pairs)} human-rated points; "
                       f"correction NOT applied because it did not reduce held-out error "
                       f"({raw_mae:.2f} -> {loo_mae:.2f} MAE)",
                _slope=1.0, _intercept=0.0,
            )
        elif method == "regression":
            corrections[judge_name] = JudgeCorrection(
                judge_name=judge_name, method="regression", n_points=len(pairs),
                detail=f"regression-corrected from {len(pairs)} human-rated points "
                       f"(raw judge was {mean_gap:+.2f} pt on average; held-out MAE "
                       f"{raw_mae:.2f} -> {loo_mae:.2f})",
                _slope=slope, _intercept=intercept,
            )
        else:
            corrections[judge_name] = JudgeCorrection(
                judge_name=judge_name, method="mean_offset", n_points=len(pairs),
                detail=f"mean-offset corrected from only {len(pairs)} human-rated "
                       f"point(s) ({mean_gap:+.2f} pt) -- add more calibration "
                       f"samples for a more reliable correction",
                _slope=slope, _intercept=intercept,
            )
    return corrections


def get_correction(corrections: Dict[str, JudgeCorrection], judge_name: str) -> JudgeCorrection:
    if judge_name in corrections:
        return corrections[judge_name]
    return JudgeCorrection(judge_name=judge_name, method="uncalibrated", n_points=0,
                            detail="NO human calibration data for this judge -- "
                                   "raw, unchecked AI-judge score used as-is")
