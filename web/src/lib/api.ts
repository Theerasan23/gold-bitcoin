// เรียก engine ผ่าน route handler /engine/* (proxy ไป FastAPI) — ฝั่ง browser ไม่ต้องรู้ที่อยู่ API
export async function getJSON<T>(path: string, signal?: AbortSignal, init?: RequestInit): Promise<T> {
  const r = await fetch(`/engine/${path}`, { signal, cache: "no-store", ...init });
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const body = await r.json();
      msg = body.detail ?? msg;
    } catch {}
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number };

export type Trade = {
  entry_time: number;
  exit_time: number | null;
  side: "long" | "short";
  entry_price: number;
  exit_price: number | null;
  r: number | null;
  exit_reason: string | null;
  open: boolean;
};

export type StrategyStatus = {
  time: number;
  close: number;
  regime: "up" | "down" | "neutral";
  in_position: boolean;
  units: number;                 // จำนวนไม้ที่ถือ (เข้าเพิ่มได้หลายไม้)
  entry_price: number | null;    // ไม้ล่าสุด
  avg_entry: number | null;
  stop: number | null;           // SL ที่ใกล้ราคาที่สุด
  positions: { entry_time: number; entry_price: number }[];
  breakout_level: number;        // ราคาที่จะเข้าไม้ถัดไปถ้าแตะ (เหนือไม้ล่าสุดด้วย)
  signal_on_last_bar: boolean;
  risk_per_unit: number;
  atr: number;
};

export type Strategy = {
  symbol: string;
  interval: string;
  params: Record<string, unknown>;
  status: StrategyStatus;
  summary: { trades: number; total_r: number; win_rate: number; since: number };
  trades: Trade[];
  profile: { from: number; poc: number; vah: number; val: number; bars: number } | null;
  sideways: { from: number; to: number; top: number; bottom: number; bars: number; active: boolean }[];
  series: { time: number[]; regime: number[]; breakout: (number | null)[]; stop: (number | null)[] };
};

export type PortfolioMetric = {
  portfolio: string;
  period: "all" | "in_sample" | "out_sample";
  trades: number;
  avg_r: number;
  sqn: number;
  return_pct: number;
  cagr_pct: number;
  max_dd_pct: number;
  mar: number | null;
  sharpe: number;
  avg_exposure_pct: number;
  bh_5050_pct: number;
  bh_5050_dd_pct: number;
};

export type Portfolio = {
  run: string;
  config: { risk_pct?: number; max_lev?: number; interval?: string; trend?: Record<string, unknown>; split?: string };
  series: Record<string, { time: number[]; value: number[] }>;
  metrics: PortfolioMetric[];
};

export type Research<T> = { run: string; rows: T[] };

// ---------------------------------------------------------------- บัญชีเดโม
export type PaperPosition = {
  symbol: string; side: number; qty: number; entry_price: number; ref_price: number | null; entry_time: number;
  entry_bar: number; risk: number; init_stop: number; stop: number; price: number | null; pnl: number | null;
  r_now: number | null;
};
export type PaperRunInfo = {
  run: string; interval: string; capital: number; risk_pct: number; created_ms: number; equity: number;
  return_pct: number; trades: number; wins: number; total_r: number; open_positions: number;
  heartbeat_ms: number | null; active: boolean;
  compare?: { paper_return_pct: number; bt_return_pct: number; trades_paper: number; trades_bt: number;
    matched: number; avg_entry_diff_pct: number | null } | null;
};
export type PaperStatus = {
  run: string;
  active: boolean;
  config: { capital: number; risk_pct: number; symbols: string[]; interval: string; created_ms: number };
  equity: number; cash: number; heartbeat_ms: number | null; started_bar: number | null;
  last_bar: Record<string, number>; prices: Record<string, number>; positions: PaperPosition[];
  watch: Record<string, { bar: number; trigger: number; risk: number }>;   // จุดแตะของแท่งที่กำลังวิ่ง
  closed: { trades: number; wins: number; total_r: number; pnl: number };
};
export type PaperTrade = {
  symbol: string; side: string; qty: number; entry_time: number; entry_bar: number; entry_price: number;
  ref_entry_price: number | null; exit_time: number; exit_price: number; ref_exit_price: number | null;
  reason: string; risk: number; pnl: number; r: number;
};
export type PaperEvent = { time: number; type: string; symbol: string | null; [k: string]: unknown };
export type PaperEquity = { time: number; equity: number; source: string };
export type CompareRow = {
  symbol: string; entry_bar: number; status: string; paper_entry: number | null; bt_entry: number | null;
  entry_diff_pct: number | null; paper_r: number | null; bt_r: number | null; paper_reason: string | null;
};
export type PaperCompare =
  | { ready: false; message: string }
  | {
      ready: true; from: number; to: number; capital: number;
      summary: { paper_return_pct: number; bt_return_pct: number; trades_paper: number; trades_bt: number;
        matched: number; avg_entry_diff_pct: number | null };
      trades: CompareRow[]; equity_paper: { time: number; equity: number }[]; equity_bt: { time: number; equity: number }[];
    };
