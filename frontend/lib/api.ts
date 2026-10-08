import type { Analysis, AnalyzeParams, Dataset } from "./types";

const BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error(`Can't reach the analysis server at ${BASE}. Check that the backend is running.`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`;
    throw new Error(detail);
  }
  return body as T;
}

export function uploadCsv(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  return request<Dataset>("/api/datasets", { method: "POST", body: form });
}

export function loadSample(): Promise<Dataset> {
  return request<Dataset>("/api/datasets/sample", { method: "POST" });
}

export function analyze(params: AnalyzeParams): Promise<Analysis> {
  return request<Analysis>("/api/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
}
