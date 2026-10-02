"""สถานะปัจจุบันของระบบ (สำหรับหน้าเว็บ) — ใช้ระบบ trend ตัวเดียวกับพอร์ต (portfolio.TREND)"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import data as dt
from .config import INTERVAL_SEC
from .core import net_r, simulate_touch
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
    if p.entry_on != "touch":
        raise ValueError("strategy_state ใช้กับ entry_on=touch")
    b, t = ds.b, ds.t
    n = t.size
    t0 = _ms(data_start(t))
    T, eq, pos, stop_line = simulate_touch(b, p, t >= t0, n, cap=False)
    r = net_r(T)
    up, down = b["regimes"][p.regime]
    breakout = highest_prev(b["high"], p.breakout_len)
    start = max(0, n - bars)

    # SL ที่ "มีผล" ในแท่ง i = ค่าที่คำนวณตอนปิดแท่ง i-1 (SL ของไม้ที่ใกล้ราคาที่สุด)
    stop = np.r_[np.nan, stop_line[:-1]]
    trades = []
    for k, row in enumerate(T):
        eb, xb, side = int(row[0]), int(row[1]), int(row[2])
        is_open = bool(int(row[8]) == 7 and xb == n - 1 and pos[-1] != 0)   # end_of_data = ยังไม่ปิดจริง
        if np.isnan(stop[eb]):   # ไม้แรกของชุด : แท่งที่เข้าใช้ SL เริ่มต้น
            stop[eb] = float(row[3]) - side * float(row[9])
        trades.append({
            "entry_time": int(t[eb] // 1000), "exit_time": None if is_open else int(t[xb] // 1000),
            "side": "long" if side == 1 else "short", "entry_price": float(row[3]),
            "exit_price": None if is_open else float(row[4]), "r": None if is_open else float(r[k]),
            "exit_reason": None if is_open else ["stop", "trailing", "tp", "tp1", "trend_flip", "signal_exit",
                                                   "reverse", "end", "time"][int(row[8])],
            "open": is_open, "risk": float(row[9]),
        })

    last_i = n - 1
    held = [x for x in trades if x["open"]]
    in_pos = bool(held)
    level = float(np.nanmax(b["high"][last_i - p.breakout_len + 1:last_i + 1]))   # High 20 แท่งล่าสุด (รวมแท่งนี้)
    last_px = held[-1]["entry_price"] if held else None   # ไม้ล่าสุด
    next_entry = max(level, last_px) if last_px is not None else level
    status = {
        "time": int(t[last_i] // 1000),
        "close": float(b["close"][last_i]),
        "regime": "up" if up[last_i] else "down" if down[last_i] else "neutral",
        "in_position": in_pos,
        "units": len(held),
        "entry_price": held[-1]["entry_price"] if held else None,    # ไม้ล่าสุด
        "avg_entry": float(np.mean([x["entry_price"] for x in held])) if held else None,
        "stop": float(stop_line[last_i]) if in_pos else None,         # SL ที่ใกล้ราคาที่สุด สำหรับแท่งถัดไป
        "positions": [{"entry_time": x["entry_time"], "entry_price": x["entry_price"]} for x in held],
        "breakout_level": next_entry,     # ราคาที่จะเข้าไม้ (ถัดไป) ถ้าแตะในแท่งถัดไป
        "signal_on_last_bar": bool(T.shape[0] > 0 and int(T[:, 0].max()) == last_i),   # เข้าไม้ในแท่งล่าสุด
        "risk_per_unit": float(b["atr"][last_i] * p.sl_atr),   # ระยะ SL ถ้าเข้าแท่งถัดไป (2 ATR) — ใช้คำนวณขนาดไม้
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
