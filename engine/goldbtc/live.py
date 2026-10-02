"""สถานะปัจจุบันของระบบ (สำหรับหน้าเว็บ) — ใช้ระบบ trend ตัวเดียวกับพอร์ต (portfolio.TREND)"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import data as dt
from .config import INTERVAL_SEC
from .core import net_r
from .core import simulate as core_sim
from .indicators import highest_prev
from .structure import volume_profile
from .portfolio import TREND
from .walkforward import Dataset, _ms, data_start

_lock = threading.Lock()
_last_check: dict[tuple[str, str], float] = {}


def ensure_fresh(symbol: str, interval: str, min_gap_s: float = 60.0) -> int:
    """ดึงแท่งที่ปิดแล้วเพิ่มจาก Binance ถ้าข้อมูลเก่ากว่า 1 แท่ง (เช็คไม่เกินทุก min_gap_s วินาที)"""
    key = (symbol, interval)
    now = time.time()
    with _lock:
        if now - _last_check.get(key, 0) < min_gap_s:
            return 0
        _last_check[key] = now
        try:
            last = dt.load(symbol, interval)["open_time"].max().timestamp()
        except FileNotFoundError:
            last = 0
        if now - last < 2 * INTERVAL_SEC[interval]:
            return 0
        _, n = dt.update(symbol, interval)
        return n


def _stop_path(h: np.ndarray, atr: np.ndarray, entry: int, last: int, init_stop: float, trail: float,
               side: int) -> np.ndarray:
    """เส้น SL ของไม้หนึ่งไม้ ณ ราคาปิดแต่ละแท่ง (= ระดับที่ใช้กับแท่งถัดไป) ตามตรรกะ Chandelier ใน backtest"""
    if side == 1:
        cand = np.maximum.accumulate(h[entry:last + 1]) - atr[entry:last + 1] * trail
        return np.maximum.accumulate(np.maximum(np.nan_to_num(cand, nan=init_stop), init_stop))
    cand = np.minimum.accumulate(h[entry:last + 1]) + atr[entry:last + 1] * trail
    return np.minimum.accumulate(np.minimum(np.nan_to_num(cand, nan=init_stop), init_stop))


def volume_profile_now(b: dict, length: int = 250, bins: int = 24) -> dict | None:
    """POC / Value Area 70% ของ length แท่งล่าสุด (ตัวเดียวกับใน Pine และ backtest)"""
    n = b["close"].size
    if n < 30:
        return None
    s = max(0, n - length)
    h, l, c, v = (b[k][s:] for k in ("high", "low", "close", "volume"))
    m = c.size
    poc, vah, val = volume_profile(h, l, c, v, m, bins, m - 1)
    if np.isnan(poc[-1]):
        return None
    return {"from": int(b["time_ms"][s] // 1000), "poc": float(poc[-1]), "vah": float(vah[-1]), "val": float(val[-1]),
            "bars": int(m)}


def sideways_boxes(b: dict, start: int, adx_max: float = 20.0, min_bars: int = 12) -> list[dict]:
    """กรอบ sideway : ช่วงที่ ADX(14) < adx_max ต่อเนื่อง >= min_bars แท่ง (บทความ: ADX < 20 = ตลาดไม่มีเทรนด์)
    ขอบบน/ล่าง = High สูงสุด / Low ต่ำสุดของช่วงนั้น · กรอบที่ยังไม่จบ (แท่งล่าสุดยังไซด์เวย์) = active"""
    flat = np.nan_to_num(b["adx"], nan=99.0) < adx_max
    t, h, l = b["time_ms"], b["high"], b["low"]
    n = flat.size
    out = []
    i = start
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        if j - i + 1 >= min_bars:
            out.append({"from": int(t[i] // 1000), "to": int(t[j] // 1000), "top": float(h[i:j + 1].max()),
                        "bottom": float(l[i:j + 1].min()), "bars": int(j - i + 1), "active": bool(j == n - 1)})
        i = j + 1
    return out


def strategy_state(symbol: str, interval: str, bars: int = 500) -> dict:
    ds = Dataset(symbol, interval)
    p = ds.params(TREND)
    b, t = ds.b, ds.t
    n = t.size
    t0 = _ms(data_start(t))
    mask = t >= t0
    raw = ds.raw(p)
    T, eq, pos = core_sim(b, raw, p, mask, n)
    r = net_r(T)
    up, down = b["regimes"][p.regime]
    breakout = highest_prev(b["high"], p.breakout_len)
    start = max(0, n - bars)

    # เส้น SL ของทุกไม้ที่อยู่ในช่วงที่แสดง (ไม้ที่ยังถืออยู่ = ถึงแท่งล่าสุด)
    stop = np.full(n, np.nan)
    trades = []
    next_stop = None
    for k, row in enumerate(T):
        eb, xb, side = int(row[0]), int(row[1]), int(row[2])
        entry_px, exit_px, risk = float(row[3]), float(row[4]), float(row[9])
        is_open = bool(int(row[8]) == 7 and xb == n - 1 and pos[-1] != 0)   # end_of_data = ยังไม่ปิดจริง
        last = n - 1 if is_open else max(xb - 1, eb)
        init = entry_px - side * risk
        path = _stop_path(b["high"] if side == 1 else b["low"], b["atr"], eb, last, init, p.trail_atr, side)
        if last >= start:
            # SL ที่ "มีผล" ในแท่ง i คือค่าที่คำนวณตอนปิดแท่ง i-1 (แท่งที่เข้า = SL เริ่มต้น)
            stop[eb:last + 1] = np.r_[init, path[:-1]]
        next_stop = float(path[-1])
        trades.append({
            "entry_time": int(t[eb] // 1000), "exit_time": None if is_open else int(t[xb] // 1000),
            "side": "long" if side == 1 else "short", "entry_price": entry_px,
            "exit_price": None if is_open else exit_px, "r": None if is_open else float(r[k]) if k < r.size else None,
            "exit_reason": None if is_open else ["stop", "trailing", "tp", "tp1", "trend_flip", "signal_exit",
                                                   "reverse", "end", "time"][int(row[8])],
            "open": is_open,
        })

    last_i = n - 1
    in_pos = bool(pos[-1] != 0)
    risk_now = float(b["atr"][last_i] * p.sl_atr)
    status = {
        "time": int(t[last_i] // 1000),
        "close": float(b["close"][last_i]),
        "regime": "up" if up[last_i] else "down" if down[last_i] else "neutral",
        "in_position": in_pos,
        "entry_price": trades[-1]["entry_price"] if in_pos and trades else None,
        "stop": next_stop if in_pos else None,          # SL สำหรับแท่งถัดไป
        "breakout_level": float(np.nanmax(b["high"][last_i - p.breakout_len + 1:last_i + 1])),
        "signal_on_last_bar": bool(raw["long"][last_i] and up[last_i]),
        "risk_per_unit": risk_now,       # ระยะ SL ถ้าเข้าตอนนี้ (2 ATR) — ใช้คำนวณขนาดไม้
        "atr": float(b["atr"][last_i]),
    }
    closed = [x["r"] for x in trades if x["r"] is not None]
    sl = slice(start, n)
    # sideway ที่เริ่มก่อนช่วงที่แสดง : ย้อนไปหาจุดเริ่มจริง (ไม่งั้นกรอบถูกตัดครึ่ง)
    first = start
    while first > 0 and np.nan_to_num(b["adx"][first - 1], nan=99.0) < 20.0:
        first -= 1
    return {
        "symbol": symbol, "interval": interval, "params": p.to_dict(), "status": status,
        "summary": {"trades": len(closed), "total_r": float(np.sum(closed)) if closed else 0.0,
                    "win_rate": float(np.mean(np.array(closed) > 0) * 100) if closed else 0.0,
                    "since": int(t0 // 1000)},
        "trades": trades,
        "profile": volume_profile_now(b),
        "sideways": sideways_boxes(b, first),
        "series": {
            "time": (t[sl] // 1000).tolist(),
            "regime": np.where(up[sl], 1, np.where(down[sl], -1, 0)).tolist(),
            "breakout": [None if np.isnan(x) else float(x) for x in breakout[sl]],
            "stop": [None if np.isnan(x) else float(x) for x in stop[sl]],
        },
    }
