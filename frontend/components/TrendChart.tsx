"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Analysis } from "@/lib/types";
import { compact, full, shortDate } from "@/lib/format";
import styles from "./dashboard.module.css";

type Row = {
  date: string;
  value?: number;
  forecast?: number;
  band80?: [number, number];
  band95?: [number, number];
  anomaly?: number;
  severity?: "high" | "medium" | "low";
};

type Props = {
  analysis: Analysis;
  activeAnomaly: string | null;
  onSelectAnomaly: (date: string | null) => void;
};

export default function TrendChart({ analysis, activeAnomaly, onSelectAnomaly }: Props) {
  const { history, forecast, anomalies, frequency, metric } = analysis;

  const rows = useMemo<Row[]>(() => {
    const byDate = new Map<string, Row>();
    for (const h of history) byDate.set(h.date, { date: h.date, value: h.value });
    for (const a of anomalies.points) {
      const row = byDate.get(a.date);
      if (row) {
        row.anomaly = a.actual;
        row.severity = a.severity;
      }
    }
    if (forecast && history.length) {
      // Join the forecast to the last actual so the lines connect.
      const last = byDate.get(history[history.length - 1].date)!;
      last.forecast = last.value;
      last.band80 = [last.value!, last.value!];
      last.band95 = [last.value!, last.value!];
      for (const p of forecast.points) {
        byDate.set(p.date, {
          date: p.date,
          forecast: p.forecast,
          band80: [p.lower_80, p.upper_80],
          band95: [p.lower_95, p.upper_95],
        });
      }
    }
    return Array.from(byDate.values());
  }, [history, forecast, anomalies.points]);

  const lastActual = history[history.length - 1]?.date;

  return (
    <figure className={styles.chartFigure}>
      <figcaption className="visually-hidden">
        {metric} by {frequency === "D" ? "day" : frequency === "W" ? "week" : "month"}, with forecast and
        flagged anomalies.
      </figcaption>
      <div className={styles.chartBox}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 16, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={(d) => shortDate(d, frequency === "D" ? "M" : frequency)}
              tick={{ fill: "var(--graphite)", fontSize: 12 }}
              axisLine={{ stroke: "var(--rule-strong)" }}
              tickLine={false}
              minTickGap={48}
            />
            <YAxis
              tickFormatter={(v) => compact(v)}
              tick={{ fill: "var(--graphite)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              width={52}
            />
            <Tooltip content={<ChartTooltip frequency={frequency} />} />
            {lastActual && forecast && (
              <ReferenceLine x={lastActual} stroke="var(--rule-strong)" strokeDasharray="3 3" />
            )}
            <Area dataKey="band95" stroke="none" fill="var(--blue-wash-2)" isAnimationActive={false} />
            <Area dataKey="band80" stroke="none" fill="var(--blue-wash)" isAnimationActive={false} />
            <Line
              dataKey="value"
              stroke="var(--ink)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              name="Actual"
            />
            <Line
              dataKey="forecast"
              stroke="var(--blue)"
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              isAnimationActive={false}
              name="Forecast"
            />
            <Scatter
              dataKey="anomaly"
              isAnimationActive={false}
              shape={(props: { cx?: number; cy?: number; payload?: Row }) => (
                <PencilRing
                  cx={props.cx}
                  cy={props.cy}
                  row={props.payload}
                  active={props.payload?.date === activeAnomaly}
                  onSelect={onSelectAnomaly}
                />
              )}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className={styles.legend} aria-hidden="true">
        <span><i className={styles.keyInk} />Actual</span>
        {forecast && <span><i className={styles.keyBlue} />Forecast, 80% and 95% ranges</span>}
        {anomalies.points.length > 0 && <span><i className={styles.keyRing} />Flagged anomaly</span>}
      </div>
    </figure>
  );
}

/** A slightly uneven red ring, like a pencil circle around a number. */
function PencilRing({
  cx,
  cy,
  row,
  active,
  onSelect,
}: {
  cx?: number;
  cy?: number;
  row?: Row;
  active: boolean;
  onSelect: (d: string | null) => void;
}) {
  if (cx === undefined || cy === undefined || !row || row.anomaly === undefined) return <g />;
  const r = row.severity === "low" ? 6 : 9;
  const opacity = row.severity === "low" ? 0.55 : 1;
  const path = `M ${cx - r} ${cy - 1}
    C ${cx - r} ${cy - r - 1}, ${cx + r + 1} ${cy - r}, ${cx + r} ${cy + 1}
    C ${cx + r - 1} ${cy + r + 1}, ${cx - r} ${cy + r}, ${cx - r + 1} ${cy - 3}`;
  return (
    <g
      className={styles.ring}
      opacity={opacity}
      onClick={() => onSelect(active ? null : row.date)}
      role="presentation"
    >
      {active && <circle cx={cx} cy={cy} r={r + 7} fill="var(--red-wash)" />}
      <path d={path} fill="none" stroke="var(--red)" strokeWidth={active ? 2.6 : 1.8} strokeLinecap="round" />
    </g>
  );
}

function ChartTooltip({
  active,
  payload,
  frequency,
}: {
  active?: boolean;
  payload?: { payload: Row }[];
  frequency: "D" | "W" | "M";
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className={styles.tooltip}>
      <strong>{shortDate(row.date, frequency)}</strong>
      {row.value !== undefined && <span>Actual {full(row.value)}</span>}
      {row.forecast !== undefined && row.value === undefined && (
        <>
          <span className={styles.tooltipBlue}>Forecast {full(row.forecast)}</span>
          {row.band80 && (
            <span className={styles.tooltipMuted}>
              Likely between {compact(row.band80[0])} and {compact(row.band80[1])}
            </span>
          )}
        </>
      )}
      {row.anomaly !== undefined && <span className={styles.tooltipRed}>Flagged as unusual</span>}
    </div>
  );
}
