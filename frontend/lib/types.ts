export type Schema = {
  date_columns: string[];
  metric_columns: string[];
  category_columns: string[];
  suggested_date: string | null;
  suggested_metric: string | null;
};

export type ColumnProfile = {
  name: string;
  role: "date" | "metric" | "category" | "other";
  null_pct: number;
  unique: number;
};

export type Dataset = {
  dataset_id: string;
  name: string;
  schema: Schema;
  profile: { rows: number; columns: ColumnProfile[]; preview: Record<string, string>[] };
};

export type Frequency = "auto" | "D" | "W" | "M";

export type ForecastPoint = {
  date: string;
  forecast: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
};

export type BacktestRow = {
  model: string;
  mae?: number;
  mape?: number | null;
  smape?: number;
  skipped?: string;
};

export type AnomalyPoint = {
  date: string;
  actual: number;
  expected: number;
  deviation_pct: number | null;
  z_score: number;
  isolation_score: number;
  direction: "spike" | "drop";
  severity: "high" | "medium" | "low";
  detected_by: string;
};

export type Mover = {
  category: string;
  recent: number;
  prior: number;
  change: number;
  change_pct: number | null;
  window_periods: number;
  compared_to: string;
};

export type Analysis = {
  dataset: string;
  metric: string;
  frequency: "D" | "W" | "M";
  history: { date: string; value: number }[];
  kpis: {
    period: string;
    total: number;
    average: number;
    latest: number;
    latest_period: string;
    change_vs_previous_pct: number | null;
    change_vs_last_season_pct: number | null;
    recent_window_periods: number;
    recent_vs_prior_window_pct: number | null;
    peak: { value: number; date: string };
    trough: { value: number; date: string };
    volatility_cv: number | null;
    seasonality_strength: number | null;
    trend: {
      growth_pct: number | null;
      basis: string;
      direction: "up" | "down" | "flat";
      r2: number;
    };
  };
  forecast: {
    model: string;
    horizon: number;
    holdout_periods: number;
    backtest: BacktestRow[];
    improvement_vs_baseline_pct: number | null;
    expected_total: number;
    points: ForecastPoint[];
  } | null;
  anomalies: {
    method: string;
    count: number;
    by_severity?: Record<"high" | "medium" | "low", number>;
    points: AnomalyPoint[];
    expected?: { date: string; value: number }[];
  };
  top_movers: Mover[];
  insights: { headline: string; insights: string[]; actions: string[]; source: "llm" | "rules" };
  elapsed_ms: number;
};

export type AnalyzeParams = {
  dataset_id: string;
  date_column: string;
  metric_column: string;
  category_column: string | null;
  frequency: Frequency;
  aggregation: "sum" | "mean";
};
