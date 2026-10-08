import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import anomaly, forecast, ingest, insights, kpis

client = TestClient(app)


def synthetic_daily(n=500, seed=0, growth=0.3, noise=0.05):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = 1000 * (1 + growth * t / n) * (1 + 0.2 * (idx.dayofweek >= 5)) * rng.normal(1, noise, n)
    s = pd.Series(y, index=idx)
    s.index.freq = "D"
    return s


# ---------- ingestion ----------

def test_detects_schema_and_cleans_currency_strings():
    csv = "Order Date,Revenue,Region\n" + "\n".join(
        f"2025-01-{d:02d},\"${1000 + d * 10:,}\",{'North' if d % 2 else 'South'}" for d in range(1, 29))
    df = ingest.read_csv(csv.encode())
    schema = ingest.detect_schema(df)
    assert schema.suggested_date == "Order Date"
    assert schema.suggested_metric == "Revenue"
    assert "Region" in schema.category_columns
    clean = ingest.prepare(df, "Order Date", "Revenue")
    assert clean["Revenue"].iloc[0] == 1010


def test_rejects_empty_and_single_column_files():
    with pytest.raises(ingest.IngestError):
        ingest.read_csv(b"")
    with pytest.raises(ingest.IngestError):
        ingest.read_csv(b"just_one\n1\n2\n")


def test_semicolon_delimited_csv():
    csv = "date;sales\n" + "\n".join(f"2025-02-{d:02d};{d * 3}" for d in range(1, 20))
    df = ingest.read_csv(csv.encode())
    assert list(df.columns) == ["date", "sales"]


# ---------- KPIs / trend ----------

def test_growth_direction_detected():
    s = synthetic_daily(n=800, growth=0.4)
    k = kpis.compute_kpis(s, "D")
    assert k["trend"]["direction"] == "up"
    assert k["trend"]["basis"] == "year over year"


def test_flat_series_is_flat():
    s = synthetic_daily(n=400, growth=0.0, seed=3)
    assert kpis.compute_kpis(s, "D")["trend"]["direction"] == "flat"


def test_trailing_partial_period_dropped():
    idx = pd.date_range("2025-01-01", "2025-06-10", freq="D")
    df = pd.DataFrame({"d": idx, "v": 10.0})
    monthly = kpis.aggregate(df, "d", "v", "M")
    assert monthly.index[-1] == pd.Timestamp("2025-05-01")


def test_weeks_are_monday_based_and_partial_weeks_dropped():
    # Data starts on a Wednesday and ends on a Tuesday.
    idx = pd.date_range("2025-01-01", "2025-03-04", freq="D")
    df = pd.DataFrame({"d": idx, "v": 10.0})
    weekly = kpis.aggregate(df, "d", "v", "W")
    assert weekly.index[0] == pd.Timestamp("2025-01-06")  # first full Monday
    assert (weekly.index.dayofweek == 0).all()
    assert (weekly == 70.0).all(), "every remaining week should be a full 7 days"


def test_seasonality_strength_needs_three_cycles():
    s = synthetic_daily(n=18)  # under three weekly cycles
    assert kpis.compute_kpis(s, "D")["seasonality_strength"] is None


# ---------- forecasting ----------

def test_forecast_shape_and_intervals():
    s = synthetic_daily(n=300)
    fc = forecast.forecast(s, horizon=14, season=7)
    assert len(fc["points"]) == 14
    for p in fc["points"]:
        assert p["lower_95"] <= p["lower_80"] <= p["forecast"] <= p["upper_80"] <= p["upper_95"]
    widths = [p["upper_95"] - p["lower_95"] for p in fc["points"]]
    assert widths[-1] > widths[0], "uncertainty should widen with horizon"


def test_forecast_never_negative_for_positive_data():
    s = synthetic_daily(n=200, growth=-0.9)
    fc = forecast.forecast(s, horizon=60, season=7)
    assert all(p["forecast"] >= 0 and p["lower_95"] >= 0 for p in fc["points"])


def test_forecast_requires_history():
    with pytest.raises(ValueError):
        forecast.forecast(synthetic_daily(n=5), horizon=3, season=7)


# ---------- anomaly detection ----------

@pytest.mark.parametrize("seed", range(5))
def test_injected_anomalies_are_found_without_flooding(seed):
    s = synthetic_daily(n=400, seed=seed)
    rng = np.random.default_rng(100 + seed)
    spots = rng.choice(np.arange(30, 370), size=4, replace=False)
    for i, pos in enumerate(spots):
        s.iloc[pos] *= 2.2 if i % 2 else 0.35
    res = anomaly.detect_anomalies(s, season=7)
    major = {p["date"] for p in res["points"] if p["severity"] in ("high", "medium")}
    injected = {s.index[p].strftime("%Y-%m-%d") for p in spots}
    assert injected <= major, f"missed {injected - major}"
    assert len(major - injected) <= 2, f"too many false alarms: {major - injected}"


def test_clean_series_has_no_high_severity_flags():
    res = anomaly.detect_anomalies(synthetic_daily(n=400, seed=11), season=7)
    assert res["by_severity"]["high"] == 0


# ---------- insights ----------

def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(insights.httpx, "post", boom)
    s = synthetic_daily(n=400)
    k = kpis.compute_kpis(s, "D")
    facts = insights.build_facts("revenue", k, None, anomaly.detect_anomalies(s, 7), [])
    out = insights.generate(facts)
    assert out["source"] == "rules"
    assert out["headline"] and out["insights"]


# ---------- API ----------

def test_api_end_to_end_with_sample():
    up = client.post("/api/datasets/sample")
    assert up.status_code == 200
    body = up.json()
    r = client.post("/api/analyze", json={
        "dataset_id": body["dataset_id"],
        "date_column": body["schema"]["suggested_date"],
        "metric_column": body["schema"]["suggested_metric"],
        "category_column": "category",
        "frequency": "D",
    })
    assert r.status_code == 200
    a = r.json()
    top3 = {p["date"] for p in a["anomalies"]["points"][:3]}
    assert top3 == {"2025-03-14", "2025-11-28", "2026-02-09"}  # the anomalies baked into the sample
    assert a["kpis"]["trend"]["direction"] == "up"
    assert a["forecast"]["points"]


def test_api_upload_csv_and_errors():
    csv = "date,amount\n" + "\n".join(f"2025-03-{d:02d},{100 + d}" for d in range(1, 31))
    r = client.post("/api/datasets", files={"file": ("x.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    assert client.post("/api/datasets", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 415
    assert client.post("/api/analyze", json={"dataset_id": "nope", "date_column": "a",
                                              "metric_column": "b"}).status_code == 404
