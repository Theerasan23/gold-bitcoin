"""ระบบเดียวกัน (portfolio.TREND) รันทุก timeframe บนช่วงเวลาปฏิทินเดียวกัน -> เทียบกันได้ตรง ๆ

ต่อ timeframe : ผลรายเหรียญ (R) + พอร์ตรวมเงินก้อนเดียว (เสี่ยง risk_pct % ต่อไม้) · แยกทั้งช่วง / ก่อน split / ตั้งแต่ split
ค่าพารามิเตอร์เป็น "จำนวนแท่ง" เท่ากันทุก TF (breakout 20 แท่ง, ATR 14 แท่ง) — คือการใช้ระบบเดิมบนกราฟ TF นั้นตรง ๆ
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import data as dt
from .config import DATA_DIR
from .core import sqn
from .portfolio import TREND, _period, build_sleeve, closes_on, simulate_portfolio
from .walkforward import _ms, _stats, data_start

INTERVALS = ["15m", "1h", "4h", "1d"]


def common_period(symbols: list[str], intervals: list[str], data_dir: Path = DATA_DIR) -> tuple[int, int]:
    """ช่วงที่ทุกเหรียญ x ทุก TF มีข้อมูล (และผ่านช่วงอุ่นเครื่องแล้ว)"""
    starts, ends = [], []
    for s in symbols:
        for i in intervals:
            t = dt.load(s, i, data_dir)["open_time"].dt.epoch("ms").to_numpy()
            starts.append(_ms(data_start(t)))
            ends.append(int(t[-1]) + 1)
    return max(starts), min(ends)


def run(symbols: list[str], intervals: list[str] | None = None, split: str = "2023-01-01",
        risk_pct: float = 1.0, max_lev: float = 1.0, data_dir: Path = DATA_DIR):
    intervals = intervals or INTERVALS
    t0, t1 = common_period(symbols, intervals, data_dir)
    s_ms = _ms(datetime.fromisoformat(split).replace(tzinfo=timezone.utc))
    periods = (("all", t0, t1), ("in_sample", t0, s_ms), ("out_sample", s_ms, t1))
    rows = []
    for interval in intervals:
        sleeves = {s: build_sleeve("trend", s, interval, t0, t1, data_dir) for s in symbols}
        groups = {**{s: [s] for s in symbols}, "portfolio": list(symbols)}
        for scope, members in groups.items():
            res = simulate_portfolio([sleeves[m] for m in members], t0, t1, risk_pct, max_lev)
            px = {s: closes_on(res["time"], sleeves[s]) for s in symbols}
            bench = {"bh_5050": sum(px[s] / px[s][0] for s in symbols) / len(symbols)}
            if scope in px:
                bench["bh_self"] = px[scope]
            for label, x0, x1 in periods:
                r = np.concatenate([sleeves[m].r_between(x0, x1) for m in members])
                st = _stats(r)
                days = max((x1 - x0) / 86_400_000, 1)
                rows.append({
                    "interval": interval, "scope": scope, "period": label,
                    "trades": st["trades"], "trades_per_month": st["trades"] / days * 30.4,
                    "win_rate": st["win_rate"], "avg_r": st["avg_r"], "total_r": st["total_r"], "sqn": sqn(r),
                    "max_dd_r": st["max_dd_r"], **_period(res, x0, x1, label, bench),
                })
    table = pl.DataFrame(rows)
    out = data_dir / "results" / f"timeframes-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "timeframes.csv")
    (out / "config.json").write_text(json.dumps({
        "symbols": symbols, "intervals": intervals, "split": split, "risk_pct": risk_pct, "max_lev": max_lev,
        "from_ms": t0, "to_ms": t1, "trend": TREND,
    }, indent=2, ensure_ascii=False))
    return table, out


def print_report(table: pl.DataFrame) -> None:
    cols = ["interval", "scope", "trades", "trades_per_month", "win_rate", "avg_r", "sqn", "return_pct", "cagr_pct",
            "max_dd_pct", "sharpe"]
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220, float_precision=2,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        for per in ("all", "in_sample", "out_sample"):
            print(f"\n=== {per} ===")
            print(table.filter(pl.col("period") == per).select([c for c in cols if c in table.columns]))
