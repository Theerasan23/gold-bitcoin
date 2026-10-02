"""ทดสอบตัดทีละส่วน (ablation) เพื่อดูว่าแต่ละส่วนของกลยุทธ์ช่วยจริงไหม"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from . import backtest as bt
from . import data as dt
from .config import DATA_DIR, Params
from .metrics import summarize

# ชื่อ -> พารามิเตอร์ที่เปลี่ยนจาก baseline (= ค่าเริ่มต้นใน Pine ตอนนี้)
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "no_elliott": {"ew_on": False},
    "no_ew_filter": {"ew_filter": False},
    "no_confluence": {"fib_on": False, "div_on": False, "vp_on": False},
    "no_sr_filter": {"use_sr_filter": False},
    "no_struct_exit": {"exit_on_rev": False},
    "no_struct_filter": {"rev_filter": False},
    "no_struct": {"exit_on_rev": False, "rev_filter": False},  # = Pine เวอร์ชันก่อนเพิ่มจุดกลับตัว
    "no_htf": {"use_htf": False},
    "no_breakeven": {"use_be": False},
    "tp_fix_3r": {"tp_mode": "fix"},
    "tp_mix": {"tp_mode": "mix"},
    "score_70": {"min_score": 70},
    "score_60": {"min_score": 60},
    "core_only": {"ew_on": False, "fib_on": False, "div_on": False, "vp_on": False,
                  "use_sr_filter": False, "rev_filter": False, "exit_on_rev": False},
}

SHOW = ["period", "trades", "win_rate", "profit_factor", "total_r", "avg_r", "max_dd_r",
        "net_pct", "max_dd_pct", "buy_hold_pct"]


def _run_dataset(args) -> pl.DataFrame:
    symbol, interval, variants, split, data_dir = args
    df = dt.load(symbol, interval, data_dir)
    base = Params().with_overrides(tick_size=dt.tick_size(symbol, data_dir))
    out = []
    for name in variants:
        res = bt.run(df, symbol, interval, base.with_overrides(**VARIANTS[name]))
        out.append(summarize(res, split).with_columns(
            pl.lit(symbol).alias("symbol"), pl.lit(interval).alias("interval"), pl.lit(name).alias("variant")))
    return pl.concat(out, how="diagonal_relaxed")


def ablate(symbols: list[str], intervals: list[str], variants: list[str] | None = None,
           split: str = "2023-01-01", data_dir: Path = DATA_DIR, workers: int | None = None) -> tuple[pl.DataFrame, Path]:
    variants = variants or list(VARIANTS)
    jobs = [(s, i, variants, split, data_dir) for s in symbols for i in intervals]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        frames = list(ex.map(_run_dataset, jobs))
    res = pl.concat(frames, how="diagonal_relaxed")
    out = data_dir / "results" / f"ablation-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out / "ablation.parquet")
    res.write_csv(out / "ablation.csv")
    return res, out


def print_table(res: pl.DataFrame) -> None:
    cfg = pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220, float_precision=2,
                    tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
    with cfg:
        for (symbol, interval), g in res.group_by(["symbol", "interval"], maintain_order=True):
            for period in ("in_sample", "out_sample"):
                part = g.filter(pl.col("period") == period)
                if part.is_empty() or "trades" not in part.columns:
                    continue
                print(f"\n=== {symbol} {interval} · {period} ===")
                print(part.select(["variant"] + [c for c in SHOW if c in part.columns and c != "period"]))
