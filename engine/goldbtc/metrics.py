"""ตัวชี้วัดผล backtest — แยกช่วง in-sample / out-of-sample ได้จากการรันครั้งเดียว"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from .backtest import Result

MS_DAY = 86_400_000


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _max_dd(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    peak = np.maximum.accumulate(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, x / peak - 1.0, 0.0)
    return float(-dd.min())


def _max_dd_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float((np.maximum.accumulate(np.r_[0.0, x]) - np.r_[0.0, x]).max())


def period_metrics(res: Result, start_ms: int, end_ms: int, label: str) -> dict:
    t = res.time_ms
    m = (t >= start_ms) & (t <= end_ms)
    out: dict = {"period": label}
    if m.sum() < 2:
        return out
    eq = res.equity[m]
    cl = res.close[m]
    days = (t[m][-1] - t[m][0]) / MS_DAY
    ret = eq[-1] / eq[0] - 1.0

    # Sharpe จาก equity รายวัน
    day_idx = (t[m] // MS_DAY)
    last_of_day = np.r_[np.nonzero(np.diff(day_idx))[0], day_idx.size - 1]
    d_eq = eq[last_of_day]
    d_ret = np.diff(d_eq) / d_eq[:-1] if d_eq.size > 2 else np.array([0.0])
    sharpe = float(d_ret.mean() / d_ret.std() * np.sqrt(365)) if d_ret.std() > 0 else 0.0

    tr = res.trades.filter(
        (pl.col("entry_time").dt.epoch("ms") >= start_ms) & (pl.col("entry_time").dt.epoch("ms") <= end_ms)
    )
    # ไม้หนึ่งไม้อาจปิดเป็นหลายส่วน (TP1) -> รวมต่อการเข้า 1 ครั้ง
    per_entry = tr.with_columns(
        (pl.col("exit_bar") - pl.col("entry_bar")).alias("bars"),
        (pl.col("r") * pl.col("frac")).alias("r_w"),
    ).group_by("entry_bar", maintain_order=True).agg(
        pl.col("pnl").sum(), pl.col("r_w").sum().alias("r"), pl.col("bars").max(),
    )
    pnl = per_entry["pnl"].to_numpy()
    r = per_entry["r"].to_numpy()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()

    out.update({
        "trades": int(per_entry.height),
        "win_rate": float((pnl > 0).mean() * 100) if pnl.size else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else None,  # ไม่มีไม้แพ้ = หาค่าไม่ได้
        "total_r": float(np.nansum(r)),
        "avg_r": float(np.nanmean(r)) if r.size else 0.0,
        "max_dd_r": _max_dd_abs(np.nancumsum(r)) if r.size else 0.0,
        "net_pct": float(ret * 100),
        "cagr_pct": float(((1 + ret) ** (365.25 / days) - 1) * 100) if days > 0 and ret > -1 else 0.0,
        "max_dd_pct": _max_dd(eq) * 100,
        "sharpe": sharpe,
        "exposure_pct": float((res.position[m] != 0).mean() * 100),
        "avg_bars": float(per_entry["bars"].mean()) if per_entry.height else 0.0,
        "buy_hold_pct": float((cl[-1] / cl[0] - 1) * 100),
    })
    return out


def summarize(res: Result, split: str | None = "2023-01-01") -> pl.DataFrame:
    """แถว 'all' + (ถ้ามี split) แถว 'in_sample' ก่อนวันแบ่ง / 'out_sample' ตั้งแต่วันแบ่ง"""
    start = _ms(res.params.start)
    end = _ms(res.params.end) if res.params.end else int(res.time_ms[-1])
    rows = [period_metrics(res, start, end, "all")]
    if split:
        s = _ms(split)
        rows.append(period_metrics(res, start, s - 1, "in_sample"))
        rows.append(period_metrics(res, s, end, "out_sample"))
    return pl.DataFrame(rows)
