"""Aggregation, KPI and trend analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import decompose

FREQ_ALIASES = {"D": "D", "W": "W-MON", "M": "MS"}
SEASON_LENGTH = {"D": 7, "W": 52, "M": 12}   # dominant cycle used for decomposition
YEAR_PERIODS = {"D": 365, "W": 52, "M": 12}
PERIOD_LABEL = {"D": "day", "W": "week", "M": "month"}


def infer_freq(dates: pd.Series) -> str:
    """Pick a sensible aggregation grain from the span and density of the data."""
    span_days = (dates.max() - dates.min()).days
    n_unique_days = dates.dt.normalize().nunique()
    if span_days > 3 * 365 or (span_days > 365 and n_unique_days < span_days * 0.3):
        return "M"
    if span_days > 120:
        return "W"
    return "D"


def _resample(obj, freq: str):
    # Weeks run Monday-Sunday and are labelled by their Monday.
    if freq == "W":
        return obj.resample("W-MON", label="left", closed="left")
    return obj.resample(FREQ_ALIASES[freq])


def aggregate(df: pd.DataFrame, date_col: str, metric_col: str, freq: str, agg: str = "sum") -> pd.Series:
    indexed = df.set_index(date_col)[metric_col]
    series = _resample(indexed, freq).agg(agg)
    if agg == "sum":
        series = series.fillna(0.0)
    else:
        series = series.interpolate(limit_direction="both")

    # Drop partial first/last periods (e.g. data starting mid-week) so they
    # don't show up as fake drops in the chart, trend or anomaly detection.
    if freq != "D" and len(series) > 3:
        first, last = df[date_col].min().normalize(), df[date_col].max().normalize()
        step = pd.Timedelta(days=7) if freq == "W" else None
        start_of_first = series.index[0]
        end_of_last = series.index[-1] + (step or pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        if first > start_of_first:
            series = series.iloc[1:]
        if last < end_of_last:
            series = series.iloc[:-1]
    series.name = metric_col
    series.index.freq = series.index.inferred_freq and pd.tseries.frequencies.to_offset(series.index.inferred_freq)
    return series.astype(float)


def _pct(a: float, b: float) -> float | None:
    if b == 0 or np.isnan(b):
        return None
    return round((a - b) / abs(b) * 100, 2)


GROWTH_THRESHOLD_PCT = 3.0


def trend_stats(series: pd.Series) -> dict:
    """Linear trend: slope, % change per period, % change across the span, R²."""
    y = series.values
    x = np.arange(len(y))
    if len(y) < 3:
        return {"slope": 0.0, "pct_per_period": 0.0, "span_change_pct": 0.0, "r2": 0.0}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = max(0.0, 1 - ss_res / ss_tot)
    start = abs(float(fitted[0])) or 1.0
    return {"slope": round(float(slope), 4),
            "pct_per_period": round(float(slope / start * 100), 3),
            "span_change_pct": round(float((fitted[-1] - fitted[0]) / start * 100), 2),
            "r2": round(r2, 3)}


def growth(series: pd.Series, freq: str, dec_trend: pd.Series | None) -> dict:
    """Headline growth and direction.

    With two full years of history, year-over-year is the most honest measure
    because it compares like-for-like seasons. Otherwise fall back to a linear
    fit on the deseasonalised trend (or raw series if we couldn't decompose).
    """
    year = YEAR_PERIODS[freq]
    lin = trend_stats(dec_trend if dec_trend is not None else series)
    if len(series) >= 2 * year:
        last, prior = series.iloc[-year:].mean(), series.iloc[-2 * year:-year].mean()
        pct, basis, confident = _pct(last, prior), "year over year", True
    else:
        pct = lin["span_change_pct"]
        basis = ("linear trend on deseasonalised data" if dec_trend is not None
                 else "linear trend on raw data")
        confident = lin["r2"] >= 0.3
    direction = "flat"
    if pct is not None and confident and abs(pct) >= GROWTH_THRESHOLD_PCT:
        direction = "up" if pct > 0 else "down"
    return {**lin, "growth_pct": pct, "basis": basis, "direction": direction}


def compute_kpis(series: pd.Series, freq: str) -> dict:
    season = SEASON_LENGTH[freq]
    dec = decompose(series, season)
    trend = growth(series, freq, dec.trend if dec is not None else None)
    latest = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) > 1 else float("nan")
    same_last_season = float(series.iloc[-1 - season]) if len(series) > season else float("nan")
    window = min(len(series) // 2, season) or 1
    recent = series.iloc[-window:].sum()
    prior = series.iloc[-2 * window:-window].sum() if len(series) >= 2 * window else float("nan")

    return {
        "period": PERIOD_LABEL[freq],
        "total": round(float(series.sum()), 2),
        "average": round(float(series.mean()), 2),
        "latest": round(latest, 2),
        "latest_period": series.index[-1].strftime("%Y-%m-%d"),
        "change_vs_previous_pct": _pct(latest, prev),
        "change_vs_last_season_pct": _pct(latest, same_last_season),
        "recent_window_periods": int(window),
        "recent_vs_prior_window_pct": _pct(recent, prior),
        "peak": {"value": round(float(series.max()), 2), "date": series.idxmax().strftime("%Y-%m-%d")},
        "trough": {"value": round(float(series.min()), 2), "date": series.idxmin().strftime("%Y-%m-%d")},
        "volatility_cv": round(float(series.std() / series.mean()), 3) if series.mean() else None,
        "trend": trend,
        # With fewer than three full cycles STL can fit seasonality almost
        # perfectly, so the strength score isn't meaningful; don't report it.
        "seasonality_strength": (dec.seasonal_strength
                                 if dec is not None and len(series) >= 3 * season else None),
    }


def top_movers(df: pd.DataFrame, date_col: str, metric_col: str, category_col: str,
               freq: str, limit: int = 5) -> list[dict]:
    """Which categories drove the change in the most recent window.

    Compares against the same window one season earlier when history allows
    (e.g. this quarter vs the same quarter last year), so normal seasonality
    isn't mistaken for a category declining.
    """
    grouped = _resample(df.set_index(date_col).groupby(category_col)[metric_col], freq).sum()
    grouped = grouped.unstack(level=0, fill_value=0.0) if grouped.index.nlevels > 1 else grouped
    if len(grouped) < 2:
        return []
    season = SEASON_LENGTH[freq]
    window = max(1, min(len(grouped) // 2, season // 4 or 1))
    recent = grouped.iloc[-window:].sum()
    if len(grouped) >= season + window:
        prior = grouped.iloc[-season - window:-season].sum()
        basis = "same window last season"
    else:
        prior = grouped.iloc[-2 * window:-window].sum()
        basis = "previous window"
    delta = (recent - prior).sort_values(key=np.abs, ascending=False).head(limit)
    return [
        {
            "category": str(cat),
            "recent": round(float(recent[cat]), 2),
            "prior": round(float(prior[cat]), 2),
            "change": round(float(delta[cat]), 2),
            "change_pct": _pct(float(recent[cat]), float(prior[cat])),
            "window_periods": int(window),
            "compared_to": basis,
        }
        for cat in delta.index
    ]
