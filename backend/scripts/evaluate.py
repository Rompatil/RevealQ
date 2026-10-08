"""Reproducible evaluation of the forecasting and anomaly-detection engines.

Generates synthetic business series with known structure (trend, weekly and
yearly seasonality, noise) and known injected anomalies, then measures:

* Anomaly detection precision / recall / F1 (medium + high severity flags).
* Forecast accuracy on a final holdout that the model-selection step never
  sees, compared with a seasonal-naive baseline.

Run:  python scripts/evaluate.py [n_series]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services import anomaly, forecast  # noqa: E402

warnings.simplefilter("ignore")


def make_series(seed: int) -> tuple[pd.Series, set[int]]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(300, 750))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    growth = rng.uniform(-0.3, 0.6)
    weekly = 1 + rng.uniform(0.05, 0.3) * (idx.dayofweek >= rng.integers(4, 6))
    yearly = 1 + rng.uniform(0, 0.2) * np.cos(2 * np.pi * (idx.dayofyear - rng.integers(0, 365)) / 365.25)
    noise = rng.normal(1, rng.uniform(0.03, 0.09), n)
    y = np.asarray(rng.uniform(500, 20000) * (1 + growth * t / n) * weekly * yearly * noise, dtype=float)

    k = int(rng.integers(2, 6))
    spots = set(int(i) for i in rng.choice(np.arange(20, n - 5), size=k, replace=False))
    for i in spots:  # mix of obvious and subtle spikes / drops
        y[i] *= rng.uniform(1.4, 3.0) if rng.random() < 0.5 else rng.uniform(0.25, 0.7)
    s = pd.Series(y, index=idx)
    s.index.freq = "D"
    return s, spots


HELD_OUT_SEED_START = 1000  # thresholds were tuned on seeds 0-99; report on unseen seeds


def main(n_series: int = 200) -> None:
    tp = fp = fn = 0
    model_mape, naive_mape, wins = [], [], 0
    chosen: dict[str, int] = {}
    started = time.perf_counter()

    for seed in range(HELD_OUT_SEED_START, HELD_OUT_SEED_START + n_series):
        s, spots = make_series(seed)

        res = anomaly.detect_anomalies(s, season=7)
        flagged = {s.index.get_loc(pd.Timestamp(p["date"])) for p in res["points"]
                   if p["severity"] in ("high", "medium")}
        tp += len(flagged & spots)
        fp += len(flagged - spots)
        fn += len(spots - flagged)

        # True out-of-sample: hide the last 28 days entirely from forecast().
        h = 28
        train, test = s.iloc[:-h], s.iloc[-h:]
        # Don't score a holdout that contains an injected anomaly.
        if any(i >= len(s) - h for i in spots):
            continue
        fc = forecast.forecast(train, horizon=h, season=7)
        pred = np.array([p["forecast"] for p in fc["points"]])
        naive = np.array([train.values[-7:][i % 7] for i in range(h)])
        m = float(np.mean(np.abs((test.values - pred) / test.values)) * 100)
        nv = float(np.mean(np.abs((test.values - naive) / test.values)) * 100)
        model_mape.append(m)
        naive_mape.append(nv)
        wins += m <= nv
        chosen[fc["model"]] = chosen.get(fc["model"], 0) + 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    elapsed = time.perf_counter() - started

    print(f"Series evaluated: {n_series}  ({elapsed:.1f}s, {elapsed / n_series * 1000:.0f} ms/series)\n")
    print("Anomaly detection (medium + high severity)")
    print(f"  injected: {tp + fn}   true positives: {tp}   false positives: {fp}")
    print(f"  precision {precision:.1%}   recall {recall:.1%}   F1 {f1:.3f}\n")
    print(f"Forecasting, 28-day out-of-sample holdout ({len(model_mape)} series)")
    print(f"  median MAPE  selected model {np.median(model_mape):.2f}%   "
          f"seasonal naive {np.median(naive_mape):.2f}%")
    print(f"  mean   MAPE  selected model {np.mean(model_mape):.2f}%   "
          f"seasonal naive {np.mean(naive_mape):.2f}%")
    rel = (np.mean(naive_mape) - np.mean(model_mape)) / np.mean(naive_mape) * 100
    print(f"  error reduction vs baseline: {rel:.1f}%   selected model at least as good on "
          f"{wins}/{len(model_mape)} series")
    print(f"  models chosen: {chosen}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
