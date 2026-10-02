import { TickMarkType, type Time } from "lightweight-charts";

const LOCALE = "th-TH-u-ca-gregory"; // ชื่อเดือนไทย ปี ค.ศ. (ตรงกับกราฟ)

export function fmtPrice(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  const d = v >= 1000 ? 2 : v >= 1 ? 3 : 5;
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "–";
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPct(v: number | null | undefined, digits = 1, sign = false): string {
  if (v == null || !Number.isFinite(v)) return "–";
  const s = v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return `${sign && v > 0 ? "+" : ""}${s}%`;
}

export function fmtR(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}R`;
}

export function fmtTime(sec: number, withTime = true): string {
  return new Date(sec * 1000).toLocaleString(LOCALE, {
    day: "numeric",
    month: "short",
    year: "2-digit",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  });
}

export function fmtDate(sec: number): string {
  return fmtTime(sec, false);
}

export const SYMBOL_LABEL: Record<string, string> = {
  BTCUSDT: "Bitcoin (BTC)",
  PAXGUSDT: "ทองคำ (PAXG)",
};

/** ป้ายแกนเวลาของกราฟ : เดือนไทย ปี ค.ศ. ตามเวลาเครื่องผู้ดู */
export function tickFormatter(time: Time, type: TickMarkType): string {
  const d = new Date((time as number) * 1000);
  if (type === TickMarkType.Year) return d.toLocaleDateString(LOCALE, { year: "numeric" });
  if (type === TickMarkType.Month) return d.toLocaleDateString(LOCALE, { month: "short" });
  if (type === TickMarkType.DayOfMonth) return d.toLocaleDateString(LOCALE, { day: "numeric", month: "short" });
  return d.toLocaleTimeString(LOCALE, { hour: "2-digit", minute: "2-digit" });
}

export const TF_SEC: Record<string, number> = { "15m": 900, "1h": 3600, "4h": 14_400, "1d": 86_400 };
export const TF_LABEL: Record<string, string> = { "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1D" };
