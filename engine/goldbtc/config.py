"""พารามิเตอร์ทั้งหมด — ค่าเริ่มต้นตรงกับ input ใน gold_bitcoin.pine"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

INTERVAL_SEC = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
}

TP_TRAIL, TP_MIX, TP_FIX = "trail", "mix", "fix"


@dataclass(frozen=True)
class Params:
    # 1) Timeframe ใหญ่
    use_htf: bool = True
    htf: str = "1d"
    htf_fast: int = 21
    htf_slow: int = 55
    # 3) โมเมนตัม
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    cross_lookback: int = 3
    rsi_len: int = 14
    rsi_long_min: float = 45
    rsi_long_max: float = 72
    rsi_short_max: float = 55
    rsi_short_min: float = 28
    # 4) ความแรงเทรนด์
    adx_len: int = 14
    adx_smooth: int = 14
    adx_min: float = 25
    vol_len: int = 20
    vol_mult: float = 1.0
    # 5) แนวรับแนวต้าน
    piv_left: int = 15
    piv_right: int = 15
    sr_max: int = 6
    sr_merge_atr: float = 0.7
    use_sr_filter: bool = True
    sr_min_atr: float = 1.5
    # 6) คะแนน
    min_score: int = 80
    cooldown_bars: int = 15
    # 7) ความเสี่ยง / ทำกำไร
    tp_mode: str = TP_TRAIL
    atr_len: int = 14
    atr_mult_sl: float = 2.0
    trail_mult: float = 3.0
    tp1_r: float = 2.0
    tp1_pct: int = 50
    risk_reward: float = 3.0
    use_be: bool = True
    # 8) ทิศทาง / ช่วงเวลา
    allow_long: bool = True
    allow_short: bool = True
    start: str = "2018-01-01"
    end: str | None = None
    # 5.5) คณิตศาสตร์
    holt_alpha: float = 0.10
    holt_beta: float = 0.05
    er_len: int = 20
    reg_len: int = 50
    z_len: int = 100
    slope_ref: float = 0.25
    z_max: float = 2.0
    conf_min: int = 50
    # 5.7) Elliott + หลักฐานร่วม
    ew_on: bool = True
    ew_depth: int = 12
    ew_min_atr: float = 1.5
    ew_min_conf: int = 60
    ew_filter: bool = True
    ew_corr_on: bool = True
    ew_corr_first: bool = False
    fib_on: bool = True          # ใช้ทดสอบตัดออกเท่านั้น (ใน Pine เปิดตลอด)
    div_on: bool = True
    div_min_rsi: float = 3.0
    vp_on: bool = True
    vp_len: int = 250
    vp_bins: int = 24
    vp_step: int = 5
    # 5.8) โครงสร้างราคา
    rev_len: int = 5
    rev_buf: float = 0.1
    exit_on_rev: bool = True
    rev_filter: bool = True
    # การจำลองคำสั่ง (strategy() ใน Pine)
    initial_capital: float = 10_000
    qty_pct: float = 10.0
    commission_pct: float = 0.05
    slippage_ticks: int = 2
    tick_size: float = 0.01
    protect_fill_bar: bool = True  # Pine ไม่มี SL บนแท่งที่เพิ่งเข้า (exit ถูกตั้งตอนแท่งปิด) — เปิดไว้ให้ตรงกับการเทรดจริง

    def with_overrides(self, **kw) -> "Params":
        known = {f.name: f.type for f in fields(self)}
        bad = set(kw) - set(known)
        if bad:
            raise ValueError(f"unknown params: {sorted(bad)}")
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_value(raw: str):
    """แปลงค่าจาก CLI 'key=value'"""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def htf_resolution(chart_interval: str, htf: str) -> str:
    """เหมือน htfRes ใน Pine: ถ้า TF ใหญ่ไม่ใหญ่กว่ากราฟ เลื่อนไปใช้ W / M"""
    chart = INTERVAL_SEC[chart_interval]
    if INTERVAL_SEC[htf] > chart:
        return htf
    return "1w" if chart < 604800 else "1M"
