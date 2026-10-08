"""Anomaly detection: statistical + machine-learning, combined.

1. Decompose the series (STL) so normal seasonality isn't flagged.
2. Score each residual with a robust z-score (median / MAD), which a few
   extreme points can't inflate the way a plain mean / std would.
3. Run an Isolation Forest over engineered features to catch points that are
   unusual in combination (e.g. a sudden jump that's only moderately large).
4. Flag a point if the robust z is extreme, or if both methods agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .decompose import decompose

# Thresholds were tuned on synthetic seeds 0-99 and validated on held-out
# seeds 1000-1199 (see scripts/evaluate.py), not on the reported test set.
Z_HIGH = 10.0        # severity "high"
Z_SIGNIFICANT = 5.5  # severity "medium": counted as a real anomaly
Z_WATCH = 4.5        # severity "low" on statistics alone
Z_SUPPORTED = 3.5    # severity "low" if the Isolation Forest also agrees


def _decompose(series: pd.Series, season: int) -> tuple[pd.Series, pd.Series]:
    """Return (expected in original units, residual in model space)."""
    dec = decompose(series, season)
    if dec is not None:
        return dec.expected, dec.resid
    window = max(3, min(7, len(series) // 3))
    expected = series.rolling(window, center=True, min_periods=1).median()
    return expected, series - expected


def _robust_z(resid: pd.Series) -> pd.Series:
    med = resid.median()
    mad = (resid - med).abs().median()
    scale = 1.4826 * mad
    if scale == 0:
        scale = resid.std() or 1.0
    return (resid - med) / scale


def detect_anomalies(series: pd.Series, season: int, contamination: float = 0.02) -> dict:
    if len(series) < 10:
        return {"method": "insufficient data", "count": 0, "points": []}

    expected, resid = _decompose(series, season)
    z = _robust_z(resid)

    features = pd.DataFrame({
        "resid": resid,
        "diff": series.diff().fillna(0.0),
        "pct_from_expected": ((series - expected) / expected.abs().replace(0, np.nan)).fillna(0.0),
    })
    features = (features - features.mean()) / features.std().replace(0, 1)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    forest_flag = forest.fit_predict(features.values) == -1
    forest_score = -forest.score_samples(features.values)

    flagged = (z.abs() >= Z_WATCH) | (forest_flag & (z.abs() >= Z_SUPPORTED))

    points = []
    for ts in series.index[flagged.values]:
        i = series.index.get_loc(ts)
        zi = float(z.iloc[i])
        actual = float(series.iloc[i])
        exp = float(expected.iloc[i])
        points.append({
            "date": ts.strftime("%Y-%m-%d"),
            "actual": round(actual, 2),
            "expected": round(exp, 2),
            "deviation_pct": round((actual - exp) / abs(exp) * 100, 1) if exp else None,
            "z_score": round(zi, 2),
            "isolation_score": round(float(forest_score[i]), 3),
            "direction": "spike" if zi > 0 else "drop",
            "severity": "high" if abs(zi) >= Z_HIGH else "medium" if abs(zi) >= Z_SIGNIFICANT else "low",
            "detected_by": ("both" if forest_flag[i] and abs(zi) >= Z_WATCH
                            else "statistical" if abs(zi) >= Z_WATCH else "isolation forest"),
        })
    points.sort(key=lambda p: abs(p["z_score"]), reverse=True)

    return {
        "method": "STL residual robust z-score + Isolation Forest",
        "count": len(points),
        "by_severity": {s: sum(p["severity"] == s for p in points) for s in ("high", "medium", "low")},
        "points": points,
        "expected": [
            {"date": ts.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for ts, v in expected.items()
        ],
    }
