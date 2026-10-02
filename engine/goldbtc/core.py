"""กลยุทธ์แกนหลัก + กฎจากบทความ masterthecrypto (ดู docs/masterthecrypto-notes.md)

โครงสร้าง : ทิศ (regime) -> จังหวะเข้า (entry) -> ตัวกรองเสริม (ADX / Volume) -> วิธีออก (exit) -> ขนาดไม้ตามความเสี่ยง

regime  htf_ema       EMA เร็ว/ช้า ของแท่ง TF ใหญ่ที่ปิดแล้ว (ที่ ablation ยืนยัน)
        golden_cross  SMA50 > SMA200 รายวัน (golden / death cross)
        ma200         ราคาปิดรายวัน > SMA200 รายวัน
        none          ไม่กรองทิศ
entry   breakout · pullback (RSI2) · macd · rsi14 · stoch · bb_break · bb_squeeze · obv · candle
        double_bottom · inv_hs · asc_triangle · flag   (รูปแบบกราฟ: stop="structure" ใช้ SL ตามโครงสร้าง)
exit    chandelier (trail_atr) · psar · target (target_r x R) · measured (เป้าจากรูปแบบ ไม่มีใช้ target_r)
entry_on  close  ตัดสินตอนแท่งปิด เข้าราคาเปิดแท่งถัดไป ถือได้ทีละไม้
          touch  (breakout เท่านั้น) เข้าทันทีที่ราคาแตะจุด breakout ระหว่างแท่ง + เข้าเพิ่มทุกครั้งที่ทำ high ใหม่
                 เหนือไม้ล่าสุด (ไม่ต้องปิดไม้เดิม) · ทิศกลับปิดทุกไม้ที่ราคาเปิดแท่งแรกที่รู้
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import polars as pl

from . import backtest as bt
from . import indicators as ta
from . import patterns as pt
from .config import htf_resolution

ENTRIES = ("breakout", "pullback", "macd", "rsi14", "stoch", "bb_break", "bb_squeeze", "obv", "candle",
           "double_bottom", "inv_hs", "asc_triangle", "flag")
PATTERNS = ("double_bottom", "inv_hs", "asc_triangle", "flag")
REGIMES = ("htf_ema", "golden_cross", "ma200", "none")
EXITS = ("chandelier", "psar", "target", "measured")
ENTRY_ON = ("close", "touch")


@dataclass(frozen=True)
class CoreParams:
    entry: str = "breakout"
    breakout_len: int = 20
    pb_level: float = 10.0          # RSI(2) ต่ำกว่านี้ = ย่อ
    rsi_level: float = 30.0         # RSI(14) ตัดกลับขึ้นเหนือระดับนี้ (บทความ: 30 ทั่วไป / 20 คริปโต)
    stoch_level: float = 20.0       # %K ตัด %D ขึ้นใต้ระดับนี้
    bb_len: int = 20
    bb_mult: float = 2.0
    obv_len: int = 20
    swing: int = 5                  # ความลึกจุดกลับตัวของรูปแบบกราฟ
    pattern_tol_atr: float = 1.0    # ก้น/ยอดที่ "สูงใกล้กัน" ต่างไม่เกินกี่ ATR
    stop: str = "atr"               # atr | structure
    regime: str = "htf_ema"
    htf: str = "1d"
    htf_fast: int = 21
    htf_slow: int = 55
    adx_min: float = 0.0            # 0 = ปิด · บทความ: >25 มีเทรนด์
    vol_confirm: bool = False       # Volume แท่งสัญญาณ > vol_mult x ค่าเฉลี่ย 20 แท่ง
    vol_mult: float = 1.5
    allow_long: bool = True
    allow_short: bool = False
    atr_len: int = 14
    sl_atr: float = 2.0
    exit: str = "chandelier"
    trail_atr: float = 3.0
    target_r: float = 3.0
    exit_on_regime: bool = True
    entry_on: str = "close"
    max_units: int = 0              # entry_on=touch : ไม้สูงสุดต่อเหรียญ (0 = ไม่จำกัด ใช้เพดานเงินคุม)
    # ต้นทุน: Binance spot taker 0.1% ต่อขา
    risk_pct: float = 1.0
    max_lev: float = 1.0
    commission_pct: float = 0.10
    slippage_ticks: int = 2
    tick_size: float = 0.01
    initial_capital: float = 10_000

    def with_overrides(self, **kw) -> "CoreParams":
        bad = set(kw) - {f.name for f in fields(self)}
        if bad:
            raise ValueError(f"unknown params: {sorted(bad)}")
        p = replace(self, **kw)
        for name, allowed in (("entry", ENTRIES), ("regime", REGIMES), ("exit", EXITS), ("stop", ("atr", "structure")),
                               ("entry_on", ENTRY_ON)):
            if getattr(p, name) not in allowed:
                raise ValueError(f"{name} ต้องเป็นหนึ่งใน {allowed}")
        if p.entry_on == "touch" and p.entry != "breakout":
            raise ValueError("entry_on=touch ใช้ได้กับ entry=breakout เท่านั้น (สัญญาณอื่นต้องรอแท่งปิด)")
        return p

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# ข้อมูลพื้นฐาน (คำนวณครั้งเดียวต่อชุดข้อมูล)
# ----------------------------------------------------------------------------
def _htf_prev(times: pl.Series, close: np.ndarray, every: str):
    """ราคาปิดของแท่ง TF ใหญ่ทุกแท่ง + ตำแหน่ง 'แท่ง TF ใหญ่ที่ปิดแล้วล่าสุด' ของแต่ละแท่งกราฟ"""
    keys = times.dt.truncate("1mo" if every == "1M" else every).to_numpy()
    uniq, first_idx = np.unique(keys, return_index=True)
    last_idx = np.r_[first_idx[1:] - 1, len(keys) - 1]
    return close[last_idx], np.searchsorted(uniq, keys) - 1


def _take(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return np.where(pos >= 0, arr[np.clip(pos, 0, None)], np.nan)


def base_features(df: pl.DataFrame, interval: str, p: CoreParams) -> dict:
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    v = df["volume"].to_numpy().astype(np.float64)
    times = df["open_time"]
    atr = ta.atr(h, l, c, p.atr_len)

    regimes = {}
    hc, pos = _htf_prev(times, c, htf_resolution(interval, p.htf))
    ef, es, hcl = _take(ta.ema(hc, p.htf_fast), pos), _take(ta.ema(hc, p.htf_slow), pos), _take(hc, pos)
    regimes["htf_ema"] = ((hcl > es) & (ef > es), (hcl < es) & (ef < es))
    dc, dpos = _htf_prev(times, c, "1d")
    s50, s200, dcl = _take(ta.sma(dc, 50), dpos), _take(ta.sma(dc, 200), dpos), _take(dc, dpos)
    regimes["golden_cross"] = (s50 > s200, s50 < s200)
    regimes["ma200"] = (dcl > s200, dcl < s200)
    ones = np.ones(c.size, bool)
    regimes["none"] = (ones, ones)

    _, _, adx = ta.dmi(h, l, c, 14, 14)
    sar, bull = ta.psar_next(h, l, 0.02, 0.02, 0.2)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = v / ta.sma(v, 20)
    return {
        "time_ms": times.dt.epoch("ms").to_numpy(), "times": times,
        "open": o, "high": h, "low": l, "close": c, "volume": v, "atr": atr,
        "regimes": regimes, "adx": adx,
        "vol_ratio": vol_ratio,
        "psar_l": np.where(bull, sar, np.nan), "psar_s": np.where(~bull, sar, np.nan),
    }


# ----------------------------------------------------------------------------
# จังหวะเข้า (ขึ้นกับพารามิเตอร์ของ entry เท่านั้น)
# ----------------------------------------------------------------------------
def entry_raw(b: dict, p: CoreParams) -> dict:
    """{long, short, risk_l, risk_s, tgt_l, tgt_s} — risk/tgt มีเฉพาะรูปแบบกราฟ (นอกนั้น NaN)"""
    o, h, l, c, v, atr = b["open"], b["high"], b["low"], b["close"], b["volume"], b["atr"]
    n = c.size
    nan = np.full(n, np.nan)
    risk_l = risk_s = tgt_l = tgt_s = nan
    e = p.entry
    if e == "breakout":
        go_l = c > ta.highest_prev(h, p.breakout_len)
        go_s = c < ta.lowest_prev(l, p.breakout_len)
    elif e == "pullback":
        r = ta.rsi(c, 2)
        go_l, go_s = r < p.pb_level, r > 100 - p.pb_level
    elif e == "macd":
        m, s, _ = ta.macd(c, 12, 26, 9)
        go_l, go_s = ta.crossover(m, s), ta.crossover(s, m)
    elif e == "rsi14":
        r = ta.rsi(c, 14)
        lv = np.full(n, p.rsi_level)
        go_l, go_s = ta.crossover(r, lv), ta.crossover(100 - lv, r)
    elif e == "stoch":
        k, d = ta.stochastic(h, l, c, 14, 3, 3)
        go_l = ta.crossover(k, d) & (k < p.stoch_level)
        go_s = ta.crossover(d, k) & (k > 100 - p.stoch_level)
    elif e in ("bb_break", "bb_squeeze"):
        mid = ta.sma(c, p.bb_len)
        sd = ta.stdev(c, p.bb_len)
        up, lo = mid + p.bb_mult * sd, mid - p.bb_mult * sd
        go_l, go_s = ta.crossover(c, up), ta.crossover(lo, c)
        if e == "bb_squeeze":   # บทความ: แบนด์แคบ = ผันผวนต่ำ -> มักตามด้วยการวิ่งแรง
            width = (up - lo) / mid
            prev_w = np.r_[np.nan, width[:-1]]
            narrow = prev_w <= 1.2 * ta.lowest_prev(width, 120)   # แบนด์แท่งก่อนหน้าแคบใกล้สุดในรอบ 120 แท่ง
            go_l, go_s = go_l & narrow, go_s & narrow
    elif e == "obv":            # บทความ: OBV มักทะลุก่อนราคา
        ob = ta.obv(c, v)
        go_l = (ob > ta.highest_prev(ob, p.obv_len)) & (c <= ta.highest_prev(h, p.obv_len))
        go_s = (ob < ta.lowest_prev(ob, p.obv_len)) & (c >= ta.lowest_prev(l, p.obv_len))
    elif e == "candle":
        cl = pt.bullish_candles(o, h, l, c, atr)
        cs = pt.bullish_candles(*pt.mirror(o, h, l, c), atr)
        go_l = cl["engulfing"] | cl["hammer"] | cl["morning_star"] | cl["piercing"]
        go_s = cs["engulfing"] | cs["hammer"] | cs["morning_star"] | cs["piercing"]
    else:   # รูปแบบกราฟ
        res = {}
        for side, (hh, ll, cc) in (("l", (h, l, c)), ("s", (-l, -h, -c))):
            if e == "flag":
                sig, rk, tg = pt.bull_flag(hh, ll, cc, atr, 4.0, 10, 3, 15)
            else:
                P, D0 = pt.swings(hh, ll, p.swing)
                fn = {"double_bottom": pt.double_bottom, "inv_hs": pt.inverse_head_shoulders,
                      "asc_triangle": pt.ascending_triangle}[e]
                tol = p.pattern_tol_atr * (1.5 if e == "inv_hs" else 0.75 if e == "asc_triangle" else 1.0)
                sig, rk, tg = fn(cc, atr, P, D0, tol, 2.0)
            res[side] = (sig, rk, tg)
        go_l, risk_l, tgt_l = res["l"]
        go_s, risk_s, tgt_s = res["s"]
        if p.stop == "atr":
            risk_l = risk_s = nan
    return {"long": go_l, "short": go_s, "risk_l": risk_l, "risk_s": risk_s, "tgt_l": tgt_l, "tgt_s": tgt_s}


def compose(b: dict, raw: dict, p: CoreParams) -> dict:
    """รวมจังหวะเข้า + ทิศ + ตัวกรองเสริม -> สัญญาณสุดท้าย"""
    up, down = b["regimes"][p.regime]
    ok = np.ones(b["close"].size, bool)
    if p.adx_min > 0:
        ok &= b["adx"] >= p.adx_min
    if p.vol_confirm:
        ok &= b["vol_ratio"] > p.vol_mult
    return {
        "long": raw["long"] & up & ok & p.allow_long,
        "short": raw["short"] & down & ok & p.allow_short,
        "up": up, "down": down,
    }


def touch_inputs(b: dict, p: CoreParams) -> dict:
    """entry_on=touch : ทุกค่าของแท่ง i รู้ได้ตั้งแต่ราคาเปิดแท่ง i
    (ทิศใช้แท่ง TF ใหญ่ที่ปิดแล้ว · ตัวกรอง ADX/Volume และ ATR ใช้แท่งก่อนหน้า · จุดแตะ = High/Low n แท่งก่อนหน้า)"""
    up, down = b["regimes"][p.regime]
    n = b["close"].size
    ok = np.ones(n, bool)
    if p.adx_min > 0:
        ok &= b["adx"] >= p.adx_min
    if p.vol_confirm:
        ok &= b["vol_ratio"] > p.vol_mult
    ok = np.r_[False, ok[:-1]]
    risk = np.r_[np.nan, b["atr"][:-1]] * p.sl_atr
    level_l = ta.highest_prev(b["high"], p.breakout_len)
    level_s = ta.lowest_prev(b["low"], p.breakout_len)
    flip = p.exit_on_regime and p.regime != "none"
    return {
        "level_l": level_l, "level_s": level_s, "risk": risk,
        "arm_l": up & ok & p.allow_long & ~np.isnan(level_l) & ~np.isnan(risk),
        "arm_s": down & ok & p.allow_short & ~np.isnan(level_s) & ~np.isnan(risk),
        "exit_l": down & flip, "exit_s": up & flip,
    }


def simulate_touch(b: dict, p: CoreParams, entry_mask: np.ndarray | None = None, end: int | None = None,
                   cap: bool = True):
    """entry_on=touch -> (T, equity, จำนวนไม้ x ทิศ, SL ที่ใกล้ราคาที่สุด ณ ปิดแท่ง)
    cap=False : บันทึกทุกไม้ตามสัญญาณ ไม่ตัดเพราะเงินไม่พอ (พอร์ตรวมจะตัดเองตามเงินทั้งพอร์ต)"""
    n = b["close"].size if end is None else end
    x = touch_inputs(b, p)
    arm_l, arm_s = x["arm_l"], x["arm_s"]
    if entry_mask is not None:
        arm_l = arm_l & entry_mask
        arm_s = arm_s & entry_mask
    return bt.simulate_touch(
        b["open"][:n], b["high"][:n], b["low"][:n], b["close"][:n], b["atr"][:n],
        x["level_l"][:n], x["level_s"][:n], arm_l[:n], arm_s[:n], x["exit_l"][:n], x["exit_s"][:n], x["risk"][:n],
        b["psar_l"][:n], b["psar_s"][:n], 1 if p.exit == "psar" else 0, p.exit in ("target", "measured"),
        p.target_r, p.trail_atr, float(p.initial_capital), p.commission_pct / 100.0, p.slippage_ticks * p.tick_size,
        p.risk_pct, p.max_lev, p.max_units, cap,
    )


def simulate(b: dict, raw: dict, p: CoreParams, entry_mask: np.ndarray | None = None, end: int | None = None,
             long_sig: np.ndarray | None = None, short_sig: np.ndarray | None = None):
    """รันถึงแท่ง end (ไม่รวม) ไม้ที่ค้างถูกปิดที่แท่งสุดท้าย · entry_mask = เข้าได้เฉพาะแท่งที่เป็น True"""
    if p.entry_on == "touch" and long_sig is None and short_sig is None:
        T, eq, pos, _ = simulate_touch(b, p, entry_mask, end)
        return T, eq, pos
    sg = compose(b, raw, p)
    n = b["close"].size if end is None else end
    ls = sg["long"] if long_sig is None else long_sig
    ss = sg["short"] if short_sig is None else short_sig
    if entry_mask is not None:
        ls = ls & entry_mask
        ss = ss & entry_mask
    z = np.zeros(n, bool)
    tp_mode = {"chandelier": 0, "psar": 0, "target": 2, "measured": 3}[p.exit]
    exit_mode = 1 if p.exit == "psar" else 0
    no_regime = p.regime == "none"
    return bt.simulate(
        b["open"][:n], b["high"][:n], b["low"][:n], b["close"][:n], b["atr"][:n],
        ls[:n], ss[:n], z, z, sg["up"][:n], sg["down"][:n],
        p.exit_on_regime and not no_regime, False, 0, tp_mode, p.sl_atr, p.trail_atr, 0.0, 0.0, p.target_r, False,
        float(p.initial_capital), 0.0, p.commission_pct / 100.0, p.slippage_ticks * p.tick_size, True,
        1, p.risk_pct, p.max_lev,
        raw["risk_l"][:n], raw["risk_s"][:n], raw["tgt_l"][:n], raw["tgt_s"][:n],
        b["psar_l"][:n], b["psar_s"][:n], exit_mode, 0,
    )


def raw_signals(df: pl.DataFrame, interval: str, p: CoreParams) -> dict:
    """สำหรับเรียกครั้งเดียว: ข้อมูลพื้นฐาน + จังหวะเข้า + สัญญาณสุดท้าย"""
    b = base_features(df, interval, p)
    raw = entry_raw(b, p)
    sg = compose(b, raw, p)
    return {**b, "raw": raw, "long": sg["long"], "short": sg["short"], "htf_up": sg["up"], "htf_down": sg["down"]}


def net_r(T: np.ndarray) -> np.ndarray:
    """R สุทธิต่อการเข้า 1 ครั้ง (รวมทุกส่วนที่ปิด) หลังหักต้นทุน"""
    if T.shape[0] == 0:
        return np.empty(0)
    qty0 = T[:, 5] / T[:, 6]
    r = T[:, 7] / (T[:, 9] * qty0)
    _, inv = np.unique(T[:, 0], return_inverse=True)
    return np.bincount(inv, weights=r)


def sqn(r: np.ndarray) -> float:
    """System Quality Number = mean / std x sqrt(n)  (= t-stat ของกำไรเฉลี่ยต่อไม้)"""
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(r.size))
