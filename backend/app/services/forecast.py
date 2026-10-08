"""Time-series forecasting with holdout-based model selection.

Candidates are fitted on all but the last `h` periods, scored on that holdout,
and the winner is refitted on the full history to produce the forecast.
Always comparing against a seasonal-naive baseline keeps the reported
accuracy honest: if a fancy model can't beat "same as last season", we say so.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

Z_80 = 1.2816
Z_95 = 1.96


def _seasonal_naive(train: pd.Series, h: int, season: int) -> tuple[np.ndarray, np.ndarray]:
    if len(train) >= season:
        last = train.values[-season:]
        pred = np.array([last[i % season] for i in range(h)])
        fitted_resid = train.values[season:] - train.values[:-season]
    else:
        pred = np.repeat(train.values[-1], h)
        fitted_resid = np.diff(train.values)
    return pred, fitted_resid


def _holt(train: pd.Series, h: int, season: int) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(train.values, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
    return model.forecast(h), model.resid


def _holt_winters(train: pd.Series, h: int, season: int) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(train) < 2 * season + 2:
        raise ValueError("not enough history for seasonal model")
    model = ExponentialSmoothing(train.values, trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=season,
                                 initialization_method="estimated").fit()
    return model.forecast(h), model.resid


CANDIDATES: dict[str, Callable] = {
    "Seasonal naive (baseline)": _seasonal_naive,
    "Holt (damped trend)": _holt,
    "Holt-Winters (trend + seasonality)": _holt_winters,
}


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None
    denom = np.abs(actual) + np.abs(pred)
    smape = float(np.mean(np.where(denom == 0, 0, 2 * np.abs(err) / np.where(denom == 0, 1, denom))) * 100)
    return {"mae": round(mae, 3), "mape": None if mape is None else round(mape, 2), "smape": round(smape, 2)}


def forecast(series: pd.Series, horizon: int, season: int) -> dict:
    n = len(series)
    if n < 8:
        raise ValueError("Need at least 8 periods of history to forecast.")
    holdout = max(2, min(horizon, n // 4))
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    scores: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in CANDIDATES.items():
            try:
                pred, _ = fn(train, holdout, season)
                if not np.all(np.isfinite(pred)):
                    raise ValueError("non-finite forecast")
                scores.append({"model": name, **_metrics(test.values, np.asarray(pred))})
            except Exception as exc:
                scores.append({"model": name, "skipped": str(exc)})

        valid = [s for s in scores if "mae" in s]
        best = min(valid, key=lambda s: s["mae"])
        baseline = next(s for s in scores if s["model"].startswith("Seasonal naive"))

        pred, resid = CANDIDATES[best["model"]](series, horizon, season)

    pred = np.asarray(pred, dtype=float)
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    sigma = float(np.std(resid)) if resid.size > 1 else float(series.std())
    steps = np.sqrt(np.arange(1, horizon + 1))  # uncertainty widens with horizon
    non_negative = bool((series >= 0).all())

    def clip(a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0) if non_negative else a

    future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=series.index.freq)[1:]
    points = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": round(float(clip(np.array([p]))[0]), 2),
            "lower_80": round(float(clip(np.array([p - Z_80 * sigma * s]))[0]), 2),
            "upper_80": round(float(p + Z_80 * sigma * s), 2),
            "lower_95": round(float(clip(np.array([p - Z_95 * sigma * s]))[0]), 2),
            "upper_95": round(float(p + Z_95 * sigma * s), 2),
        }
        for d, p, s in zip(future_index, pred, steps)
    ]

    improvement = None
    if "mae" in baseline and baseline["mae"] > 0:
        improvement = round((baseline["mae"] - best["mae"]) / baseline["mae"] * 100, 1)

    return {
        "model": best["model"],
        "horizon": horizon,
        "holdout_periods": holdout,
        "backtest": scores,
        "improvement_vs_baseline_pct": improvement,
        "expected_total": round(float(sum(p["forecast"] for p in points)), 2),
        "points": points,
    }
