"""Shared seasonal-trend decomposition.

statsmodels' STL defaults use the smallest allowed windows, which lets the
trend and seasonal components soak up ordinary noise. That makes residuals
artificially small and turns normal days into "anomalies". We widen both
windows and, for strictly positive data, decompose in log space so that
multiplicative seasonality (e.g. "weekends are +18%") is handled correctly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _odd(x: float) -> int:
    n = int(np.ceil(x))
    return n if n % 2 else n + 1


@dataclass
class Decomposition:
    expected: pd.Series      # trend + seasonal, in original units
    trend: pd.Series         # in original units
    resid: pd.Series         # in model space (log or linear)
    seasonal_strength: float | None
    log_space: bool


def decompose(series: pd.Series, season: int) -> Decomposition | None:
    if len(series) < 2 * season + 1:
        return None
    from statsmodels.tsa.seasonal import STL

    log_space = bool((series > 0).all())
    y = np.log(series) if log_space else series

    seasonal_win = 13 if len(series) >= 13 * season // 2 else 7
    default_trend = 1.5 * season / (1 - 1.5 / seasonal_win)
    trend_win = _odd(max(default_trend, min(31, len(series) // 4)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(y, period=season, seasonal=seasonal_win, trend=trend_win, robust=True).fit()

    var_r = float(np.var(res.resid))
    var_sr = float(np.var(res.seasonal + res.resid))
    strength = None if var_sr == 0 else round(max(0.0, 1 - var_r / var_sr), 3)

    fitted = res.trend + res.seasonal
    expected = np.exp(fitted) if log_space else fitted
    trend = np.exp(res.trend) if log_space else res.trend
    return Decomposition(expected=expected, trend=trend, resid=res.resid,
                         seasonal_strength=strength, log_space=log_space)
