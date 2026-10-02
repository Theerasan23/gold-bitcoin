"""ระบบสำหรับตลาดไซด์เวย์ (mean reversion) — ทำงานเฉพาะตอนที่ "ไม่มีเทรนด์"

ช่วงไซด์เวย์ (regime)
    adx20        ADX(14) < 20                 (บทความ: ต่ำกว่า 20 = ตลาดไม่มีเทรนด์)
    er_low       Efficiency Ratio(20) < 0.25  (ราคาเดินวกวน ไม่เป็นเส้นตรง)
    htf_neutral  เทรนด์ TF ใหญ่ไม่ใช่ทั้งขาขึ้นและขาลง
    none         ไม่กรอง (ใช้เทียบ)
จังหวะเข้า (สวนทาง)
    bb_fade      ปิดหลุดขอบล่าง Bollinger(20,2) -> ซื้อ · ปิดเหนือขอบบน -> ขาย
    rsi_fade     RSI(14) ลงต่ำกว่า 30 -> ซื้อ · ขึ้นเกิน 70 -> ขาย
    rsi2_fade    RSI(2) < 10 -> ซื้อ · > 90 -> ขาย
ออก   ราคาปิดกลับถึงเส้นกลาง (SMA20; rsi2 ใช้ SMA5) · SL เริ่มต้น sl_atr x ATR · ถือไม่เกิน max_bars แท่ง
ไม่มี trailing ไม่มีการปรับพารามิเตอร์ — ใช้ค่าตามตำรา
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from . import backtest as bt
from . import indicators as ta

MR_ENTRIES = ("bb_fade", "rsi_fade", "rsi2_fade")
MR_REGIMES = ("adx20", "er_low", "htf_neutral", "none")


@dataclass(frozen=True)
class MRParams:
    entry: str = "bb_fade"
    regime: str = "adx20"
    bb_len: int = 20
    bb_mult: float = 2.0
    rsi_low: float = 30.0
    rsi2_low: float = 10.0
    adx_max: float = 20.0
    er_max: float = 0.25
    sl_atr: float = 2.0
    max_bars: int = 20
    allow_long: bool = True
    allow_short: bool = True
    risk_pct: float = 1.0
    max_lev: float = 1.0
    commission_pct: float = 0.10
    slippage_ticks: int = 2
    tick_size: float = 0.01
    initial_capital: float = 10_000

    def with_overrides(self, **kw) -> "MRParams":
        bad = set(kw) - {f.name for f in fields(self)}
        if bad:
            raise ValueError(f"unknown params: {sorted(bad)}")
        p = replace(self, **kw)
        if p.entry not in MR_ENTRIES or p.regime not in MR_REGIMES:
            raise ValueError(f"entry ∈ {MR_ENTRIES}, regime ∈ {MR_REGIMES}")
        return p

    def to_dict(self) -> dict:
        return asdict(self)


def range_mask(b: dict, p: MRParams) -> np.ndarray:
    """แท่งที่ถือว่าเป็นตลาดไซด์เวย์ (b = core.base_features)"""
    if p.regime == "adx20":
        return b["adx"] < p.adx_max
    if p.regime == "er_low":
        return ta.efficiency_ratio(b["close"], 20) < p.er_max
    if p.regime == "htf_neutral":
        up, down = b["regimes"]["htf_ema"]
        return ~up & ~down
    return np.ones(b["close"].size, bool)


def signals(b: dict, p: MRParams) -> dict:
    c, h, l = b["close"], b["high"], b["low"]
    n = c.size
    if p.entry == "bb_fade":
        mid = ta.sma(c, p.bb_len)
        sd = ta.stdev(c, p.bb_len)
        up, lo = mid + p.bb_mult * sd, mid - p.bb_mult * sd
        go_l, go_s = ta.crossover(lo, c), ta.crossover(c, up)
        exit_mid = mid
    elif p.entry == "rsi_fade":
        r = ta.rsi(c, 14)
        go_l = ta.crossover(np.full(n, p.rsi_low), r)
        go_s = ta.crossover(r, np.full(n, 100 - p.rsi_low))
        exit_mid = ta.sma(c, p.bb_len)
    else:
        r = ta.rsi(c, 2)
        go_l, go_s = r < p.rsi2_low, r > 100 - p.rsi2_low
        exit_mid = ta.sma(c, 5)
    rng = range_mask(b, p)
    return {
        "long": go_l & rng & p.allow_long,
        "short": go_s & rng & p.allow_short,
        "range": rng,
        "exit_long": c >= exit_mid,    # กลับถึงเส้นกลาง = ถึงเป้า
        "exit_short": c <= exit_mid,
    }


def simulate(b: dict, sg: dict, p: MRParams, entry_mask: np.ndarray | None = None, end: int | None = None,
             long_sig: np.ndarray | None = None, short_sig: np.ndarray | None = None):
    n = b["close"].size if end is None else end
    ls = sg["long"] if long_sig is None else long_sig
    ss = sg["short"] if short_sig is None else short_sig
    if entry_mask is not None:
        ls = ls & entry_mask
        ss = ss & entry_mask
    z = np.zeros(n, bool)
    return bt.simulate(
        b["open"][:n], b["high"][:n], b["low"][:n], b["close"][:n], b["atr"][:n],
        ls[:n], ss[:n], sg["exit_short"][:n], sg["exit_long"][:n], z, z,
        False, True, 0, 4, p.sl_atr, 0.0, 0.0, 0.0, 0.0, False,
        float(p.initial_capital), 0.0, p.commission_pct / 100.0, p.slippage_ticks * p.tick_size, True,
        1, p.risk_pct, p.max_lev, *bt.nan_extras(n), 0, p.max_bars,
    )


# ----------------------------------------------------------------------------
# ทดสอบทีละแบบ เทียบการเข้าแบบสุ่ม (ในช่วงไซด์เวย์เดียวกัน ออกแบบเดียวกัน ความถี่เท่ากัน)
# ----------------------------------------------------------------------------
def _scan_job(args) -> list[dict]:
    from datetime import datetime, timezone

    from . import data as dt
    from .core import net_r
    from .walkforward import Dataset, _ms, _stats, _ym, data_start

    symbol, interval, seeds, split, data_dir = args
    ds = Dataset(symbol, interval, data_dir)
    base = MRParams().with_overrides(tick_size=dt.tick_size(symbol, data_dir))
    t = ds.t
    t0, t1 = _ms(data_start(t)), int(t[-1]) + 1
    s_ms = _ms(datetime.fromisoformat(split).replace(tzinfo=timezone.utc))
    rng = np.random.default_rng(11)

    def run(p, sg, x0, x1, ls=None, ss=None):
        mask = (t >= x0) & (t < x1)
        end = int(np.searchsorted(t, x1))
        T, eq, _ = simulate(ds.b, sg, p, mask, end, ls, ss)
        return net_r(T), eq[int(np.searchsorted(t, x0)):end], T

    out = []
    for entry in MR_ENTRIES:
        for regime in MR_REGIMES:
            p = base.with_overrides(entry=entry, regime=regime)
            sg = signals(ds.b, p)
            r, eq, T = run(p, sg, t0, t1)
            r_is, _, _ = run(p, sg, t0, s_ms)
            r_oos, _, _ = run(p, sg, s_ms, t1)
            mask = (t >= t0) & (t < t1)
            el_l = mask & sg["range"] & p.allow_long
            el_s = mask & sg["range"] & p.allow_short
            rate = (sg["long"][mask].sum() + sg["short"][mask].sum()) / max(int(el_l.sum() + el_s.sum()), 1)
            rm = np.zeros(seeds)
            for s in range(seeds):
                u_l = rng.random(t.size) < rate
                u_s = rng.random(t.size) < rate
                rr, _, _ = run(p, sg, t0, t1, el_l & u_l, el_s & u_s)
                rm[s] = rr.mean() if rr.size else 0.0
            reasons = [bt.EXIT_NAMES[int(x)] for x in T[:, 8]]
            out.append({
                "symbol": symbol, "interval": interval, "entry": entry, "regime": regime, "from": _ym(t0),
                **_stats(r), "return_pct": float((eq[-1] / eq[0] - 1) * 100) if eq.size else 0.0,
                "avg_r_is": float(r_is.mean()) if r_is.size else None,
                "avg_r_oos": float(r_oos.mean()) if r_oos.size else None,
                "random_avg_r": float(np.median(rm)),
                "beats_random_pct": float((rm < (r.mean() if r.size else 0.0)).mean() * 100),
                "pct_target_exit": 100 * reasons.count("signal_exit") / max(len(reasons), 1),
                "pct_stop_exit": 100 * reasons.count("stop") / max(len(reasons), 1),
            })
    return out


def run_scan(symbols, intervals, seeds=200, split="2023-01-01", data_dir=None, workers=None):
    import time
    from concurrent.futures import ProcessPoolExecutor

    import polars as pl

    from .config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    jobs = [(s, i, seeds, split, data_dir) for s in symbols for i in intervals]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = pl.DataFrame([row for rows in ex.map(_scan_job, jobs) for row in rows])
    out = data_dir / "results" / f"range-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "range.csv")
    return res, out


def print_scan(res) -> None:
    import polars as pl
    res = res.with_columns(
        (pl.col("symbol").str.replace("USDT", "") + " " + pl.col("interval")).alias("ds"),
        (pl.col("entry") + " · " + pl.col("regime")).alias("rule"),
    )
    cell = pl.when(pl.col("trades") < 10).then(pl.format("(น้อยไป) n{}", pl.col("trades"))).otherwise(
        pl.format("{} · {}% · n{}", pl.col("sqn").round(1), pl.col("beats_random_pct").round(0).cast(pl.Int64),
                  pl.col("trades")))
    stab = pl.format("{} → {}", pl.col("avg_r_is").round(2), pl.col("avg_r_oos").round(2))
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=240, fmt_str_lengths=60,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n=== ไซด์เวย์: SQN · ชนะการสุ่ม % · จำนวนไม้ ===")
        print(res.with_columns(cell.alias("v")).pivot(on="ds", index="rule", values="v"))
        print("\n=== R เฉลี่ยต่อไม้ : ก่อน 2023 → ตั้งแต่ 2023 ===")
        print(res.with_columns(stab.alias("v")).pivot(on="ds", index="rule", values="v"))
        print("\n=== ออกที่เส้นกลาง (ถึงเป้า) % / โดน SL % ===")
        print(res.with_columns(pl.format("{} / {}", pl.col("pct_target_exit").round(0),
                                         pl.col("pct_stop_exit").round(0)).alias("v"))
              .pivot(on="ds", index="rule", values="v"))
