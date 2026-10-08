"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { analyze, loadSample, uploadCsv } from "@/lib/api";
import type { Analysis, AnalyzeParams, Dataset, Frequency } from "@/lib/types";
import { DataProfile, Findings, ForecastQuality, Ledger, MarginNotes, Movers } from "@/components/Panels";
import styles from "@/components/dashboard.module.css";

const TrendChart = dynamic(() => import("@/components/TrendChart"), { ssr: false });

type Selection = Omit<AnalyzeParams, "dataset_id">;

export default function Home() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState<null | "upload" | "analyze">(null);
  const [error, setError] = useState<string | null>(null);
  const [activeAnomaly, setActiveAnomaly] = useState<string | null>(null);

  const startWith = useCallback(async (load: () => Promise<Dataset>) => {
    setBusy("upload");
    setError(null);
    try {
      const ds = await load();
      setDataset(ds);
      setAnalysis(null);
      setActiveAnomaly(null);
      setSelection({
        date_column: ds.schema.suggested_date!,
        metric_column: ds.schema.suggested_metric!,
        category_column: ds.schema.category_columns[0] ?? null,
        frequency: "auto",
        aggregation: "sum",
      });
    } catch (e) {
      setError((e as Error).message);
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    if (!dataset || !selection) return;
    let cancelled = false;
    setBusy("analyze");
    setError(null);
    analyze({ dataset_id: dataset.dataset_id, ...selection })
      .then((a) => {
        if (!cancelled) {
          setAnalysis(a);
          setActiveAnomaly(null);
        }
      })
      .catch((e) => !cancelled && setError((e as Error).message))
      .finally(() => !cancelled && setBusy(null));
    return () => {
      cancelled = true;
    };
  }, [dataset, selection]);

  const reset = () => {
    setDataset(null);
    setSelection(null);
    setAnalysis(null);
    setError(null);
  };

  return (
    <div className={styles.page}>
      <header className={styles.masthead}>
        <button type="button" className={styles.wordmark} onClick={reset}>
          RevealQ
        </button>
     {dataset && (
      <div className={styles.mastheadActions}>
                <button type="button" className={styles.newFileButton} onClick={reset}>
                   <svg viewBox="0 0 16 16" aria-hidden="true">
                       <path d="M8 10.5V2.5M4.5 6 8 2.5 11.5 6M2.5 10v2.5a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V10" />
                        </svg>
                           Analyse another file
                            </button>
                             </div>
                            )}
                       </header>

      <main>
        {!dataset && <UploadPad busy={busy === "upload"} error={error} onStart={startWith} />}

        {dataset && selection && (
          <>
            <Controls dataset={dataset} selection={selection} onChange={setSelection} busy={busy === "analyze"} />

            {error && (
              <p role="alert" className={styles.error}>
                {error}
              </p>
            )}

            {analysis && (
              <div className={styles.results} aria-busy={busy === "analyze"} data-stale={busy === "analyze"}>
                <h1 className={styles.headline}>{analysis.insights.headline}.</h1>
                <Ledger analysis={analysis} />

                <div className={styles.chartRow}>
                  <TrendChart
                    analysis={analysis}
                    activeAnomaly={activeAnomaly}
                    onSelectAnomaly={setActiveAnomaly}
                  />
                  <MarginNotes
                    points={analysis.anomalies.points}
                    frequency={analysis.frequency}
                    active={activeAnomaly}
                    onSelect={setActiveAnomaly}
                  />
                </div>

                <div className={styles.lowerGrid}>
                  <Findings analysis={analysis} />
                  <div className={styles.stack}>
                    <Movers analysis={analysis} />
                    <ForecastQuality analysis={analysis} />
                  </div>
                </div>

                <DataProfile dataset={dataset} />
                <p className={styles.footnote}>Analysed in {analysis.elapsed_ms} ms.</p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function UploadPad({
  busy,
  error,
  onStart,
}: {
  busy: boolean;
  error: string | null;
  onStart: (load: () => Promise<Dataset>) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const take = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onStart(() => uploadCsv(f));
  };

  return (
    <section className={styles.intro}>
      <h1 className={styles.introTitle}>See what changed in your numbers, what comes next, and what looks wrong.</h1>
      <p className={styles.introLede}>
        Upload a CSV with a date column and at least one number, such as orders, sales or bookings. Signalboard
        finds the trend, forecasts the next few periods and circles anything unusual.
      </p>

      <div
        className={styles.drop}
        data-dragging={dragging}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          take(e.dataTransfer.files);
        }}
      >
        <p className={styles.dropText}>{busy ? "Reading your file…" : "Drop a CSV here"}</p>
        <div className={styles.dropActions}>
          <button
            type="button"
            className={styles.primary}
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            Choose a file
          </button>
          <button type="button" className={styles.secondary} onClick={() => onStart(loadSample)} disabled={busy}>
            Use sample retail data
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.tsv,.txt,text/csv"
          className="visually-hidden"
          aria-label="Choose a CSV file"
          onChange={(e) => take(e.target.files)}
        />
        <p className={styles.muted}>Up to 15 MB. Files are held in memory for an hour and never stored.</p>
      </div>
      {error && (
        <p role="alert" className={styles.error}>
          {error}
        </p>
      )}
    </section>
  );
}

function Controls({
  dataset,
  selection,
  onChange,
  busy,
}: {
  dataset: Dataset;
  selection: Selection;
  onChange: (s: Selection) => void;
  busy: boolean;
}) {
  const { schema } = dataset;
  const set = <K extends keyof Selection>(key: K, value: Selection[K]) => onChange({ ...selection, [key]: value });

  return (
    <form className={styles.controls} onSubmit={(e) => e.preventDefault()} aria-label="Analysis settings">
      <label>
        <span>Measure</span>
        <select value={selection.metric_column} onChange={(e) => set("metric_column", e.target.value)}>
          {schema.metric_columns.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>
      <label>
        <span>Combine as</span>
        <select
          value={selection.aggregation}
          onChange={(e) => set("aggregation", e.target.value as Selection["aggregation"])}
        >
          <option value="sum">Total</option>
          <option value="mean">Average</option>
        </select>
      </label>
      <label>
        <span>Group by</span>
        <select value={selection.frequency} onChange={(e) => set("frequency", e.target.value as Frequency)}>
          <option value="auto">Best fit</option>
          <option value="D">Day</option>
          <option value="W">Week</option>
          <option value="M">Month</option>
        </select>
      </label>
      {schema.date_columns.length > 1 && (
        <label>
          <span>Date</span>
          <select value={selection.date_column} onChange={(e) => set("date_column", e.target.value)}>
            {schema.date_columns.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
      )}
      {schema.category_columns.length > 0 && (
        <label>
          <span>Break down by</span>
          <select
            value={selection.category_column ?? ""}
            onChange={(e) => set("category_column", e.target.value || null)}
          >
            <option value="">Nothing</option>
            {schema.category_columns.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
      )}
      <p className={styles.status} aria-live="polite">
        {busy ? "Updating…" : ""}
      </p>
    </form>
  );
}
