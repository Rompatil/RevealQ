"""Turn analysis results into plain-English, actionable insights.

The numbers are always computed deterministically first. If an LLM key is
configured, the model only *phrases* those computed facts (it is told not to
invent figures); otherwise a rule-based writer produces the same structure.
Either way the response has the same shape, so the UI doesn't care which ran.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a business analyst writing for a busy operator.
You receive JSON with pre-computed analysis of one business metric.
Rules:
- Use ONLY numbers that appear in the JSON. Never invent or extrapolate figures.
- Be specific and concrete; no generic advice like "monitor closely".
- If forecast accuracy is weak or the model barely beats the baseline, say so.
Return ONLY valid JSON: {"headline": str, "insights": [str, ...3-5], "actions": [str, ...2-3]}"""


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v:+.1f}%"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.2f}"


def build_facts(metric: str, kpis: dict, fc: dict | None, anomalies: dict, movers: list[dict]) -> dict:
    facts = {"metric": metric, "kpis": kpis, "anomalies": {
        "count": anomalies.get("count", 0), "by_severity": anomalies.get("by_severity", {}),
        "top": anomalies.get("points", [])[:3]}}
    if fc:
        best = next((s for s in fc["backtest"] if s["model"] == fc["model"]), {})
        facts["forecast"] = {
            "model": fc["model"], "horizon": fc["horizon"], "expected_total": fc["expected_total"],
            "backtest_mape": best.get("mape"),
            "improvement_vs_baseline_pct": fc["improvement_vs_baseline_pct"],
            "first_period": fc["points"][0], "last_period": fc["points"][-1],
        }
    if movers:
        facts["top_movers"] = movers[:3]
    return facts


def rule_based(facts: dict) -> dict:
    k, metric, period = facts["kpis"], facts["metric"], facts["kpis"]["period"]
    trend = k["trend"]
    insights: list[str] = []
    actions: list[str] = []

    word = {"up": "growing", "down": "declining", "flat": "broadly flat"}[trend["direction"]]
    headline = f"{metric} is {word}"
    g = trend.get("growth_pct")
    if trend["direction"] != "flat":
        headline += f" ({_fmt(g, pct=True)} {trend['basis']})"
        insights.append(f"{metric} is {word}: {_fmt(g, pct=True)} measured {trend['basis']}.")
    else:
        insights.append(f"No clear directional change in {metric} "
                        f"({_fmt(g, pct=True)}, {trend['basis']}).")

    if k["recent_vs_prior_window_pct"] is not None and trend["basis"] != "year over year":
        insights.append(
            f"The last {k['recent_window_periods']} {period}s are "
            f"{_fmt(k['recent_vs_prior_window_pct'], pct=True)} versus the {k['recent_window_periods']} before.")

    s = k.get("seasonality_strength")
    if s is not None and s >= 0.4:
        insights.append(f"Seasonality is {'strong' if s >= 0.64 else 'moderate'} (strength {s:.2f}); "
                        f"compare against the same point last cycle, not just the previous {period}.")

    a = facts["anomalies"]
    if a["count"]:
        top = a["top"][0]
        sev = a.get("by_severity", {})
        major = sev.get("high", 0) + sev.get("medium", 0)
        minor = sev.get("low", 0)
        lead = (f"{major} significant anomal{'y' if major == 1 else 'ies'}" if major
                else f"{minor} minor anomal{'y' if minor == 1 else 'ies'}")
        tail = f" (plus {minor} minor)" if major and minor else ""
        insights.append(
            f"{lead}{tail} detected; the largest was a {top['direction']} on {top['date']} "
            f"({_fmt(top['actual'])} vs ~{_fmt(top['expected'])} expected, "
            f"{_fmt(top['deviation_pct'], pct=True)}).")
        actions.append(f"Investigate what happened around {top['date']} — check promotions, outages or data errors.")

    fc = facts.get("forecast")
    if fc:
        insights.append(
            f"Forecast ({fc['model']}): about {_fmt(fc['expected_total'])} over the next {fc['horizon']} {period}s.")
        imp = fc["improvement_vs_baseline_pct"]
        if fc["model"].startswith("Seasonal naive"):
            actions.append("No model beat a simple 'same as last season' baseline yet — "
                           "plan off last season's numbers and revisit as history grows.")
        elif imp is not None and imp < 5:
            actions.append("Treat the forecast as a rough guide — it barely beats a seasonal-naive baseline.")
        elif fc.get("backtest_mape") is not None:
            actions.append(f"Use the forecast for planning; backtest error was {fc['backtest_mape']:.1f}% MAPE.")

    movers = facts.get("top_movers") or []
    if movers:
        m = movers[0]
        insights.append(
            f"Biggest driver of recent change: {m['category']} ({_fmt(m['change'])}, "
            f"{_fmt(m['change_pct'], pct=True)} vs {m['compared_to']}).")
        actions.append(f"Dig into {m['category']} first — it explains the most movement.")

    if trend["direction"] == "down":
        actions.append(f"Prioritise a root-cause review of the decline before the next {period} closes.")

    def cap(s: str) -> str:
        return s[:1].upper() + s[1:]

    return {"headline": cap(headline), "insights": [cap(s) for s in insights[:5]],
            "actions": [cap(s) for s in actions[:3]] or
            ["Keep collecting data; more history will sharpen the forecast."], "source": "rules"}


def llm(facts: dict, api_key: str, timeout: float = 20.0) -> dict:
    resp = httpx.post(
        ANTHROPIC_URL,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={
            "model": DEFAULT_MODEL,
            "max_tokens": 700,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(facts, default=str)}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = "".join(b.get("text", "") for b in resp.json().get("content", []))
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    if not isinstance(data.get("insights"), list) or not data.get("headline"):
        raise ValueError("LLM returned unexpected shape")
    return {"headline": str(data["headline"]), "insights": [str(x) for x in data["insights"]][:5],
            "actions": [str(x) for x in data.get("actions", [])][:3], "source": "llm"}


def generate(facts: dict) -> dict:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            return llm(facts, key)
        except Exception as exc:  # never let the LLM take the dashboard down
            log.warning("LLM insights failed, using rule-based fallback: %s", exc)
    return rule_based(facts)
