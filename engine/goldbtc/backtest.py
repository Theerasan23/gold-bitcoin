"""จำลองการส่งคำสั่งแบบ broker emulator ของ TradingView

- สัญญาณตัดสินตอนแท่งปิด -> คำสั่ง market เข้า/ออกที่ราคาเปิดแท่งถัดไป (+ slippage)
- stop / limit ตรวจในแท่ง ตามกฎของ TradingView: ถ้า open ใกล้ high กว่า = วิ่งไป high ก่อนแล้วค่อยไป low
- ราคาเปิด gap ทะลุ stop / limit -> ได้ราคาเปิด
- ค่าคอมคิดทั้งขาเข้าและขาออก, ขนาดไม้ = % ของ equity ตอนแท่งสัญญาณปิด
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from . import data as dt
from . import signals as sg
from .config import DATA_DIR, TP_FIX, TP_MIX, Params

# เหตุผลที่ปิดไม้
EXIT_STOP, EXIT_TRAIL, EXIT_TP, EXIT_TP1, EXIT_HTF, EXIT_STRUCT, EXIT_REVERSE, EXIT_END, EXIT_TIME = range(9)
EXIT_NAMES = ["stop", "trail/BE", "tp", "tp1", "htf_reversal", "signal_exit", "reverse_signal", "end_of_data",
              "time_stop"]
TP_CODE = {"trail": 0, TP_MIX: 1, TP_FIX: 2}
N_COLS = 11  # entry_bar, exit_bar, side, entry_px, exit_px, qty, frac, pnl, reason, risk, r


@njit(cache=True)
def _record(T, k, entry_bar, exit_bar, side, entry_px, exit_px, q, q0, risk, reason, comm, entry_comm):
    gross = (exit_px - entry_px) * side * q
    exit_comm = exit_px * q * comm
    T[k, 0] = entry_bar
    T[k, 1] = exit_bar
    T[k, 2] = side
    T[k, 3] = entry_px
    T[k, 4] = exit_px
    T[k, 5] = q
    T[k, 6] = q / q0
    T[k, 7] = gross - exit_comm - entry_comm * q / q0
    T[k, 8] = reason
    T[k, 9] = risk
    T[k, 10] = (exit_px - entry_px) * side / risk if risk > 0 else np.nan
    return gross - exit_comm


@njit(cache=True)
def simulate(o, h, l, c, atr, long_sig, short_sig, choch_up, choch_dn, htf_up, htf_dn,
             use_htf, exit_on_rev, cooldown, tp_mode, atr_mult_sl, trail_mult, tp1_r, tp1_frac,
             rr, use_be, capital, qty_pct, comm, slip, protect_fill, size_mode, risk_pct, max_lev,
             risk_l, risk_s, tgt_l, tgt_s, trail_l, trail_s, exit_mode, max_bars):
    """size_mode 0 = qty_pct % ของ equity (แบบ Pine) · 1 = เสี่ยง risk_pct % ของ equity ต่อไม้ (จำกัด notional ≤ max_lev เท่า)
    tp_mode   0 trailing · 1 TP1 บางส่วน + trailing · 2 TP คงที่ rr เท่าของ R · 3 เป้าจากรูปแบบ (tgt_*) ถ้าไม่มีใช้ rr
              4 ไม่มี trailing ไม่มีเป้า (ออกด้วย SL เริ่มต้น / สัญญาณออก choch_* / time stop เท่านั้น)
    choch_up / choch_dn  สัญญาณปิด short / long ที่ราคาเปิดแท่งถัดไป (ใช้เมื่อ exit_on_rev)
    max_bars  ถือไม่เกินกี่แท่ง (0 = ไม่จำกัด)
    risk_*    ระยะ SL ต่อแท่งสัญญาณ (เช่น ใต้ก้นของรูปแบบ) — NaN = ใช้ atr x atr_mult_sl
    tgt_*     ระยะเป้าจากราคาปิดแท่งสัญญาณ — ใช้เมื่อ tp_mode 3
    exit_mode 0 Chandelier (จุดสุดหลังเข้า - trail_mult x ATR) · 1 ใช้ระดับ trailing จากภายนอก trail_* (เช่น Parabolic SAR)"""
    n = c.size
    T = np.full((2 * n + 2, N_COLS), np.nan)
    k = 0
    equity = np.empty(n)
    pos_arr = np.zeros(n, np.int64)

    side = 0
    qty = 0.0
    qty0 = 0.0
    avg = 0.0
    risk = 0.0
    stop = np.nan
    init_stop = np.nan
    lim = np.nan
    tp1_done = False
    ext = np.nan          # high สูงสุด (long) / low ต่ำสุด (short) หลังเข้าไม้
    trail = np.nan
    entry_bar = -1
    entry_comm = 0.0
    eq_real = capital

    pend_entry = 0
    pend_qty = 0.0
    pend_risk = 0.0
    pend_tgt = np.nan
    tgt = np.nan
    pend_close = False
    pend_reason = 0
    last_entry = -1

    for i in range(n):
        # ---------- 1) คำสั่ง market ที่ราคาเปิด ----------
        if pend_close and side != 0:
            px = o[i] - side * slip
            eq_real += _record(T, k, entry_bar, i, side, avg, px, qty, qty0, risk, pend_reason, comm, entry_comm)
            k += 1
            side = 0
        if pend_entry != 0:
            if side != 0 and side != pend_entry:
                px = o[i] - side * slip
                eq_real += _record(T, k, entry_bar, i, side, avg, px, qty, qty0, risk, EXIT_REVERSE, comm, entry_comm)
                k += 1
                side = 0
            if side == 0:
                side = pend_entry
                avg = o[i] + side * slip
                qty = pend_qty
                qty0 = pend_qty
                risk = pend_risk
                entry_comm = avg * qty * comm
                eq_real -= entry_comm
                entry_bar = i
                tp1_done = False
                trail = np.nan
                ext = np.nan
                init_stop = avg - side * risk
                stop = init_stop if protect_fill else np.nan
                if tp_mode == 2:
                    lim = avg + side * risk * rr
                elif tp_mode == 3:
                    lim = avg + side * (pend_tgt if not np.isnan(pend_tgt) and pend_tgt > 0 else risk * rr)
                elif tp_mode == 1:
                    lim = avg + side * risk * tp1_r
                else:
                    lim = np.nan
        pend_entry = 0
        pend_close = False

        # ---------- 2) stop / limit ระหว่างแท่ง ----------
        if side != 0:
            use_lim = (tp_mode == 2) or (tp_mode == 3) or (tp_mode == 1 and not tp1_done)
            stop_ok = not np.isnan(stop)
            lim_ok = use_lim and not np.isnan(lim)
            if side == 1:
                stop_gap = stop_ok and o[i] <= stop
                lim_gap = lim_ok and o[i] >= lim
                stop_hit = stop_ok and l[i] <= stop
                lim_hit = lim_ok and h[i] >= lim
            else:
                stop_gap = stop_ok and o[i] >= stop
                lim_gap = lim_ok and o[i] <= lim
                stop_hit = stop_ok and h[i] >= stop
                lim_hit = lim_ok and l[i] <= lim
            up_first = (h[i] - o[i]) <= (o[i] - l[i])
            fav_first = up_first if side == 1 else not up_first
            stop_reason = EXIT_STOP if stop == init_stop else EXIT_TRAIL

            do_lim_first = False
            do_stop = False
            stop_px = stop - side * slip
            if stop_gap:
                do_stop = True
                stop_px = o[i] - side * slip
            elif lim_gap or (lim_hit and fav_first):
                do_lim_first = True
                do_stop = stop_hit
            elif stop_hit:
                do_stop = True
            elif lim_hit:
                do_lim_first = True

            if do_lim_first:
                lim_px = o[i] if lim_gap else lim
                if tp_mode == 2 or tp_mode == 3:
                    eq_real += _record(T, k, entry_bar, i, side, avg, lim_px, qty, qty0, risk, EXIT_TP, comm, entry_comm)
                    k += 1
                    side = 0
                    do_stop = False
                else:
                    part = min(qty, qty0 * tp1_frac)
                    eq_real += _record(T, k, entry_bar, i, side, avg, lim_px, part, qty0, risk, EXIT_TP1, comm, entry_comm)
                    k += 1
                    qty -= part
                    tp1_done = True
                    if qty <= 1e-12:
                        side = 0
                        do_stop = False
            if do_stop and side != 0:
                eq_real += _record(T, k, entry_bar, i, side, avg, stop_px, qty, qty0, risk, stop_reason, comm, entry_comm)
                k += 1
                side = 0

        # ---------- 3) แท่งปิด : สคริปต์ตัดสินใจ ----------
        eq_close = eq_real + (side * qty * (c[i] - avg) if side != 0 else 0.0)
        cool_ok = last_entry < 0 or (i - last_entry) >= cooldown
        go_long = long_sig[i] and cool_ok and side <= 0 and not np.isnan(atr[i])
        go_short = short_sig[i] and cool_ok and side >= 0 and not np.isnan(atr[i])
        if go_long or go_short:
            pend_entry = 1 if go_long else -1
            rk = risk_l[i] if go_long else risk_s[i]
            pend_risk = rk if (not np.isnan(rk) and rk > 0) else atr[i] * atr_mult_sl
            pend_tgt = tgt_l[i] if go_long else tgt_s[i]
            if size_mode == 1 and pend_risk > 0:
                pend_qty = min(risk_pct / 100.0 * eq_close / pend_risk, max_lev * eq_close / c[i])
            else:
                pend_qty = qty_pct / 100.0 * eq_close / c[i]
            last_entry = i

        if side != 0:
            if entry_bar == i:
                stop = avg - side * risk
                init_stop = stop
                ext = h[i] if side == 1 else l[i]
                trail = np.nan
            trailing = tp_mode <= 1
            if side == 1:
                ext = max(ext, h[i])
                ch = (ext - atr[i] * trail_mult) if exit_mode == 0 else trail_l[i]
                if not np.isnan(ch):
                    trail = ch if np.isnan(trail) else max(trail, ch)
                eff = stop
                if use_be and c[i] >= avg + risk:
                    eff = max(eff, avg)
                if trailing and not np.isnan(trail):
                    eff = max(eff, trail)
                stop = eff
            else:
                ext = min(ext, l[i])
                ch = (ext + atr[i] * trail_mult) if exit_mode == 0 else trail_s[i]
                if not np.isnan(ch):
                    trail = ch if np.isnan(trail) else min(trail, ch)
                eff = stop
                if use_be and c[i] <= avg - risk:
                    eff = min(eff, avg)
                if trailing and not np.isnan(trail):
                    eff = min(eff, trail)
                stop = eff

            if exit_on_rev and ((side == 1 and choch_dn[i]) or (side == -1 and choch_up[i])):
                pend_close = True
                pend_reason = EXIT_STRUCT
            elif use_htf and ((side == 1 and htf_dn[i]) or (side == -1 and htf_up[i])):
                pend_close = True
                pend_reason = EXIT_HTF
            elif max_bars > 0 and i - entry_bar + 1 >= max_bars:
                pend_close = True
                pend_reason = EXIT_TIME

        equity[i] = eq_close
        pos_arr[i] = side

    if side != 0:
        eq_real += _record(T, k, entry_bar, n - 1, side, avg, c[n - 1], qty, qty0, risk, EXIT_END, comm, entry_comm)
        k += 1
        equity[n - 1] = eq_real
    return T[:k], equity, pos_arr


def nan_extras(n: int) -> tuple[np.ndarray, ...]:
    """risk_l, risk_s, tgt_l, tgt_s, trail_l, trail_s ว่างทั้งหมด (= ใช้ค่าจาก ATR)"""
    z = np.full(n, np.nan)
    return z, z, z, z, z, z


@dataclass
class Result:
    symbol: str
    interval: str
    params: Params
    time_ms: np.ndarray
    close: np.ndarray
    equity: np.ndarray
    position: np.ndarray
    trades: pl.DataFrame


def run(df: pl.DataFrame, symbol: str, interval: str, p: Params) -> Result:
    f = sg.compute(df, interval, p)
    tp_frac = p.tp1_pct / 100.0
    T, equity, pos = simulate(
        f["open"], f["high"], f["low"], f["close"], f["atr"],
        f["long_sig"], f["short_sig"], f["choch_up"], f["choch_dn"], f["htf_up"], f["htf_down"],
        p.use_htf, p.exit_on_rev, p.cooldown_bars, TP_CODE[p.tp_mode], p.atr_mult_sl, p.trail_mult,
        p.tp1_r, tp_frac, p.risk_reward, p.use_be, float(p.initial_capital), p.qty_pct,
        p.commission_pct / 100.0, p.slippage_ticks * p.tick_size, p.protect_fill_bar, 0, 0.0, 1.0,
        *nan_extras(f["close"].size), 0, 0,
    )
    return Result(symbol, interval, p, f["time_ms"], f["close"], equity, pos, trades_frame(T, f["time_ms"]))


def trades_frame(T: np.ndarray, t: np.ndarray) -> pl.DataFrame:
    eb = T[:, 0].astype(np.int64)
    xb = T[:, 1].astype(np.int64)
    qty0 = T[:, 5] / T[:, 6]
    return pl.DataFrame({
        "entry_time": pl.Series(t[eb]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "exit_time": pl.Series(t[xb]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "entry_bar": eb, "exit_bar": xb,
        "side": np.where(T[:, 2] > 0, "long", "short"),
        "entry_price": T[:, 3], "exit_price": T[:, 4],
        "qty": T[:, 5], "frac": T[:, 6], "pnl": T[:, 7],
        "exit_reason": pl.Series([EXIT_NAMES[int(r)] for r in T[:, 8]], dtype=pl.Utf8),
        "risk": T[:, 9], "r": T[:, 10],
        # R หลังหักค่าคอม/slippage (เทียบกับเงินที่เสี่ยงจริงของทั้งไม้)
        "net_r": T[:, 7] / (T[:, 9] * qty0),
    })


def load_and_run(symbol: str, interval: str, p: Params | None = None, data_dir: Path = DATA_DIR) -> Result:
    p = p or Params()
    p = p.with_overrides(tick_size=dt.tick_size(symbol, data_dir))
    return run(dt.load(symbol, interval, data_dir), symbol, interval, p)


def save(res: Result, metrics: pl.DataFrame, data_dir: Path = DATA_DIR) -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = data_dir / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    res.trades.write_parquet(out / "trades.parquet", compression="zstd")
    pl.DataFrame({
        "time": pl.Series(res.time_ms).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "equity": res.equity, "close": res.close, "position": res.position,
    }).write_parquet(out / "equity.parquet", compression="zstd")
    metrics.with_columns(
        pl.lit(run_id).alias("run_id"),
        pl.lit(res.symbol).alias("symbol"),
        pl.lit(res.interval).alias("interval"),
        pl.lit(json.dumps(res.params.to_dict())).alias("params"),
        pl.lit(time.strftime("%Y-%m-%dT%H:%M:%S")).alias("created"),
    ).write_parquet(out / "metrics.parquet")
    return run_id
