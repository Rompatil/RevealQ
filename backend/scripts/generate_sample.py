"""Generate the bundled sample dataset: ~2 years of daily retail orders.

Built-in structure (so the demo has something real to find):
- upward trend (~+25% over two years)
- weekly seasonality (weekend peak) and yearly seasonality (Q4 peak)
- injected anomalies: a payment outage, a flash-sale spike, a data-entry glitch
- 4 product categories with different growth rates
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
dates = pd.date_range("2024-07-01", "2026-06-30", freq="D")
categories = {"Electronics": 1.35, "Home & Kitchen": 1.15, "Fashion": 1.05, "Groceries": 0.92}
base = {"Electronics": 4200, "Home & Kitchen": 2600, "Fashion": 3100, "Groceries": 1900}

rows = []
n = len(dates)
for i, d in enumerate(dates):
    progress = i / n
    weekly = 1 + 0.18 * (d.dayofweek >= 4)                    # Fri-Sun bump
    yearly = 1 + 0.22 * np.cos(2 * np.pi * (d.dayofyear - 340) / 365.25)  # peaks mid-Dec
    for cat, growth in categories.items():
        trend = 1 + (growth - 1) * progress
        revenue = base[cat] * trend * weekly * yearly * rng.normal(1, 0.07)
        if d == pd.Timestamp("2025-03-14"):          # payment provider outage
            revenue *= 0.25
        if d == pd.Timestamp("2025-11-28") and cat == "Electronics":  # flash sale
            revenue *= 3.2
        if d == pd.Timestamp("2026-02-09") and cat == "Fashion":      # duplicated import
            revenue *= 2.6
        aov = {"Electronics": 310, "Home & Kitchen": 95, "Fashion": 72, "Groceries": 38}[cat]
        orders = max(1, int(revenue / (aov * rng.normal(1, 0.05))))
        rows.append({"order_date": d.strftime("%Y-%m-%d"), "category": cat,
                     "revenue": round(revenue, 2), "orders": orders,
                     "region": rng.choice(["Dubai", "Abu Dhabi", "Sharjah"], p=[0.55, 0.3, 0.15])})

out = Path(__file__).resolve().parent.parent / "sample_data" / "retail_sales.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {len(rows)} rows to {out}")
