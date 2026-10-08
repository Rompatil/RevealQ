"""Signalboard API."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services import anomaly, forecast, insights, ingest, kpis
from .store import store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signalboard")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "retail_sales.csv"

app = FastAPI(title="Signalboard API", version="1.0.0",
              description="Upload business data; get trends, forecasts, anomalies and insights.")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    dataset_id: str
    date_column: str
    metric_column: str
    category_column: str | None = None
    frequency: Literal["auto", "D", "W", "M"] = "auto"
    aggregation: Literal["sum", "mean"] = "sum"
    horizon: int = Field(default=0, ge=0, le=104, description="0 = pick automatically")


def _register(name: str, raw: bytes) -> dict:
    try:
        df = ingest.read_csv(raw)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    schema = ingest.detect_schema(df)
    if not schema.date_columns:
        raise HTTPException(status_code=422, detail="No date column detected. Include a column of dates.")
    if not schema.metric_columns:
        raise HTTPException(status_code=422, detail="No numeric metric column detected.")
    dataset_id = store.put(name, df, schema)
    return {"dataset_id": dataset_id, "name": name, "schema": schema.to_dict(),
            "profile": ingest.profile(df, schema)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_enabled": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/api/datasets")
async def upload(file: UploadFile = File(...)) -> dict:
    if file.filename and not file.filename.lower().endswith((".csv", ".txt", ".tsv")):
        raise HTTPException(status_code=415, detail="Please upload a CSV file.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 15 MB.")
    return _register(file.filename or "upload.csv", raw)


@app.post("/api/datasets/sample")
def load_sample() -> dict:
    return _register("retail_sales.csv (sample)", SAMPLE_PATH.read_bytes())


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    started = time.perf_counter()
    item = store.get(req.dataset_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Dataset expired or not found. Please upload again.")
    name, df, schema = item

    try:
        clean = ingest.prepare(df, req.date_column, req.metric_column)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    freq = kpis.infer_freq(clean[req.date_column]) if req.frequency == "auto" else req.frequency
    series = kpis.aggregate(clean, req.date_column, req.metric_column, freq, req.aggregation)
    if len(series) < 8:
        raise HTTPException(status_code=422,
                            detail=f"Only {len(series)} {kpis.PERIOD_LABEL[freq]}s of data after aggregation; "
                                   "need at least 8. Try a finer frequency.")
    season = kpis.SEASON_LENGTH[freq]
    horizon = req.horizon or {"D": 30, "W": 12, "M": 6}[freq]

    kpi = kpis.compute_kpis(series, freq)
    try:
        fc = forecast.forecast(series, horizon, season)
    except ValueError as exc:
        log.info("forecast skipped: %s", exc)
        fc = None
    anomalies = anomaly.detect_anomalies(series, season)

    movers: list[dict] = []
    if req.category_column and req.category_column in clean.columns:
        movers = kpis.top_movers(clean, req.date_column, req.metric_column, req.category_column, freq)

    facts = insights.build_facts(req.metric_column, kpi, fc, anomalies, movers)
    summary = insights.generate(facts)

    return {
        "dataset": name,
        "metric": req.metric_column,
        "frequency": freq,
        "history": [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in series.items()],
        "kpis": kpi,
        "forecast": fc,
        "anomalies": anomalies,
        "top_movers": movers,
        "insights": summary,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
