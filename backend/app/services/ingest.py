"""CSV ingestion and schema detection.

Takes an arbitrary business CSV and works out which column is the date,
which columns are numeric metrics, and which are low-cardinality categories
(useful for "top movers" breakdowns).
"""
from __future__ import annotations

import csv
import io
import warnings
from dataclasses import dataclass, field

import pandas as pd

MAX_ROWS = 500_000
DATE_NAME_HINTS = ("date", "time", "day", "month", "period", "timestamp", "created", "ordered")


class IngestError(ValueError):
    """Raised when an upload can't be turned into an analysable dataset."""


@dataclass
class Schema:
    date_columns: list[str] = field(default_factory=list)
    metric_columns: list[str] = field(default_factory=list)
    category_columns: list[str] = field(default_factory=list)
    suggested_date: str | None = None
    suggested_metric: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def read_csv(raw: bytes) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise IngestError("The file is empty.")
    head = raw[:20_000].decode("utf-8", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, nrows=MAX_ROWS, encoding_errors="replace")
    except Exception as exc:  # pandas raises many different parser errors
        raise IngestError(f"Could not parse CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty or len(df.columns) < 2:
        raise IngestError("CSV needs at least two columns (a date and a numeric metric).")
    return df


def _looks_like_date(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    hit_rate = parsed.notna().mean()
    threshold = 0.6 if any(h in name.lower() for h in DATE_NAME_HINTS) else 0.9
    return hit_rate >= threshold


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce strings like '$1,234.50' or '12%' to numbers."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace(r"[,\s$€£₹%]", "", regex=True)
        .str.replace(r"^AED", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def detect_schema(df: pd.DataFrame) -> Schema:
    schema = Schema()
    for col in df.columns:
        s = df[col]
        if _looks_like_date(s, col):
            schema.date_columns.append(col)
            continue
        numeric = _to_numeric(s)
        if numeric.notna().mean() >= 0.8 and numeric.nunique() > 2:
            schema.metric_columns.append(col)
            continue
        nunique = s.nunique(dropna=True)
        if 1 < nunique <= min(50, max(2, len(df) // 5)):
            schema.category_columns.append(col)

    if schema.date_columns:
        named = [c for c in schema.date_columns if any(h in c.lower() for h in DATE_NAME_HINTS)]
        schema.suggested_date = (named or schema.date_columns)[0]
    if schema.metric_columns:
        # Prefer money-ish columns, otherwise the one with the largest total.
        money = [c for c in schema.metric_columns
                 if any(k in c.lower() for k in ("revenue", "sales", "amount", "total", "gmv", "value"))]
        if money:
            schema.suggested_metric = money[0]
        else:
            totals = {c: _to_numeric(df[c]).abs().sum() for c in schema.metric_columns}
            schema.suggested_metric = max(totals, key=totals.get)
    return schema


def prepare(df: pd.DataFrame, date_col: str, metric_col: str) -> pd.DataFrame:
    """Return a clean frame with parsed dates and a numeric metric column."""
    for col in (date_col, metric_col):
        if col not in df.columns:
            raise IngestError(f"Column '{col}' not found.")
    out = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", format="mixed")
    out[metric_col] = _to_numeric(out[metric_col])
    out = out.dropna(subset=[date_col, metric_col])
    if out.empty:
        raise IngestError("No rows had both a valid date and a numeric value.")
    return out.sort_values(date_col)


def profile(df: pd.DataFrame, schema: Schema) -> dict:
    return {
        "rows": int(len(df)),
        "columns": [
            {
                "name": c,
                "role": ("date" if c in schema.date_columns
                         else "metric" if c in schema.metric_columns
                         else "category" if c in schema.category_columns
                         else "other"),
                "null_pct": round(float(df[c].isna().mean() * 100), 1),
                "unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
        "preview": df.head(8).astype(str).to_dict(orient="records"),
    }
