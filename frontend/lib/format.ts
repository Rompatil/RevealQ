export function compact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function full(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return new Intl.NumberFormat("en", { maximumFractionDigits: 2 }).format(n);
}

export function pct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  const sign = n > 0 ? "+" : n < 0 ? "−" : "";
  return `${sign}${Math.abs(n).toFixed(digits)}%`;
}

export function shortDate(iso: string, freq: "D" | "W" | "M" = "D"): string {
  const d = new Date(`${iso}T00:00:00`);
  if (freq === "M") return d.toLocaleDateString("en", { month: "short", year: "numeric" });
  return d.toLocaleDateString("en", { day: "numeric", month: "short", year: "numeric" });
}

export const PERIOD_NAME = { D: "day", W: "week", M: "month" } as const;
