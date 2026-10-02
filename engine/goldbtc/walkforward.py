"""Walk-forward + ทดสอบกฎทีละข้อ (rule scan)

walk-forward : เลือกชุดพารามิเตอร์ที่ดีที่สุดจากช่วง train (SQN) -> วัดผลบนช่วง test ที่ไม่เคยเห็น เลื่อนไปทีละช่วง
rule scan    : เปลี่ยนทีละอย่างจากชุดอ้างอิง แล้ววัดผลทั้งช่วงข้อมูล (กฎมาจากบทความ ไม่ได้ปรับจากข้อมูลเรา)

ทั้งสองแบบเทียบกับ "เข้าแบบสุ่ม" ในทิศเทรนด์เดียวกัน ออกแบบเดียวกัน ความถี่สัญญาณเท่ากัน
=> ถ้าชนะการสุ่มไม่ได้ แปลว่าจังหวะเข้า/ตัวกรองนั้นไม่ได้เพิ่มอะไร
"""

from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import data as dt
from .config import DATA_DIR
from .core import PATTERNS, CoreParams, base_features, compose, entry_raw, net_r, simulate, sqn

ENTRY_VARIANTS: list[dict] = [
    {"entry": "breakout", "breakout_len": 20},
    {"entry": "breakout", "breakout_len": 55},
    {"entry": "pullback", "pb_level": 10.0},
    {"entry": "macd"},
    {"entry": "rsi14", "rsi_level": 30.0},
    {"entry": "rsi14", "rsi_level": 40.0},
    {"entry": "stoch"},
    {"entry": "bb_break"},
    {"entry": "bb_squeeze"},
    {"entry": "obv"},
    {"entry": "candle"},
    *[{"entry": e, "stop": s} for e in PATTERNS for s in ("atr", "structure")],
]
FILTER_GRID = {"regime": ["htf_ema", "golden_cross", "ma200"], "adx_min": [0.0, 25.0], "vol_confirm": [False, True]}
EXIT_VARIANTS: list[dict] = [
    {"exit": "chandelier", "trail_atr": 3.0},
    {"exit": "chandelier", "trail_atr": 5.0},
    {"exit": "psar"},
    {"exit": "target", "target_r": 3.0},
    {"exit": "measured"},
]
SIDES = [False, True]
DAY_MS = 86_400_000
ENTRY_KEYS = ("entry", "breakout_len", "pb_level", "rsi_level", "stop")


# ----------------------------------------------------------------------------
# ช่วงเวลา
# ----------------------------------------------------------------------------
def _month_start(ms: int) -> datetime:
    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_months(d: datetime, m: int) -> datetime:
    y, mo = divmod(d.month - 1 + m, 12)
    return d.replace(year=d.year + y, month=mo + 1)


def _ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def _ym(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def make_windows(t: np.ndarray, train_months: int, test_months: int, warmup_days: int = 120,
                 anchored: bool = False) -> list[tuple[int, int, int, int]]:
    """[(train_start, train_end, test_start, test_end)] หน่วย ms · ช่วง test ไม่ทับกันและต่อกันพอดี"""
    first = data_start(t, warmup_days)
    test_start = _add_months(first, train_months)
    last = int(t[-1]) + 1
    out = []
    while _ms(test_start) < last - 28 * DAY_MS:
        test_end = min(_ms(_add_months(test_start, test_months)), last)
        train_start = first if anchored else _add_months(test_start, -train_months)
        out.append((_ms(train_start), _ms(test_start), _ms(test_start), test_end))
        test_start = _add_months(test_start, test_months)
    return out


def data_start(t: np.ndarray, warmup_days: int = 120) -> datetime:
    """ต้นเดือนแรกหลังช่วงอุ่นเครื่องอินดิเคเตอร์"""
    return _add_months(_month_start(int(t[0]) + warmup_days * DAY_MS), 1)


# ----------------------------------------------------------------------------
# ชุดพารามิเตอร์
# ----------------------------------------------------------------------------
def combos() -> list[dict]:
    out = []
    for ev in ENTRY_VARIANTS:
        for fv in itertools.product(*FILTER_GRID.values()):
            for xv in EXIT_VARIANTS:
                if xv["exit"] == "measured" and ev["entry"] not in PATTERNS:
                    continue    # ไม่มีเป้าจากรูปแบบ = ซ้ำกับ target
                for short in SIDES:
                    out.append({**ev, **dict(zip(FILTER_GRID, fv)), **xv, "allow_short": short})
    return out


def _ev_key(c: dict) -> str:
    return json.dumps({k: c[k] for k in ENTRY_KEYS if k in c}, sort_keys=True)


# ----------------------------------------------------------------------------
# สถิติ
# ----------------------------------------------------------------------------
def _stats(r: np.ndarray) -> dict:
    if r.size == 0:
        return {"trades": 0, "total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0, "pf": None, "sqn": 0.0, "max_dd_r": 0.0}
    cum = np.r_[0.0, np.cumsum(r)]
    loss = -r[r < 0].sum()
    return {
        "trades": int(r.size), "total_r": float(r.sum()), "avg_r": float(r.mean()),
        "win_rate": float((r > 0).mean() * 100), "pf": float(r[r > 0].sum() / loss) if loss > 0 else None,
        "sqn": sqn(r), "max_dd_r": float((np.maximum.accumulate(cum) - cum).max()),
    }


def _stitch(curves: list[np.ndarray]) -> tuple[float, float]:
    """ต่อ equity ของแต่ละช่วง test (เริ่มใหม่ที่ทุนเท่ากันทุกช่วง) -> ผลตอบแทนทบต้น %, max drawdown %"""
    factor = 1.0
    parts = []
    for eq in curves:
        if eq.size == 0:
            continue
        parts.append(eq / eq[0] * factor)
        factor = parts[-1][-1]
    if not parts:
        return 0.0, 0.0
    x = np.concatenate(parts)
    return float((factor - 1) * 100), float((1 - x / np.maximum.accumulate(x)).max() * 100)


class Dataset:
    """โหลดข้อมูล + คำนวณส่วนที่ใช้ร่วมกันครั้งเดียว แล้วรันชุดพารามิเตอร์ใดก็ได้บนช่วงเวลาใดก็ได้"""

    def __init__(self, symbol: str, interval: str, data_dir: Path = DATA_DIR):
        self.symbol, self.interval = symbol, interval
        df = dt.load(symbol, interval, data_dir)
        self.base_p = CoreParams().with_overrides(tick_size=dt.tick_size(symbol, data_dir))
        self.b = base_features(df, interval, self.base_p)
        self.t = self.b["time_ms"]
        self.close = self.b["close"]
        self._raw: dict[str, dict] = {}

    def params(self, c: dict) -> CoreParams:
        return self.base_p.with_overrides(**c)

    def raw(self, p: CoreParams) -> dict:
        key = _ev_key(p.to_dict())
        if key not in self._raw:
            self._raw[key] = entry_raw(self.b, p)
        return self._raw[key]

    def run(self, p: CoreParams, t0: int, t1: int, long_sig=None, short_sig=None):
        """เข้าได้เฉพาะ [t0, t1) และปิดไม้ค้างที่ t1 -> (R สุทธิต่อไม้, equity ช่วงนั้น)"""
        mask = (self.t >= t0) & (self.t < t1)
        end = int(np.searchsorted(self.t, t1))
        start = int(np.searchsorted(self.t, t0))
        T, eq, _ = simulate(self.b, self.raw(p), p, mask, end, long_sig=long_sig, short_sig=short_sig)
        return net_r(T), eq[start:end]

    def random_means(self, p: CoreParams, t0: int, t1: int, seeds: int, rng) -> np.ndarray:
        """ค่าเฉลี่ย R ต่อไม้ของการเข้าแบบสุ่ม (ทิศ regime เดียวกัน ความถี่สัญญาณเท่ากัน ออกแบบเดียวกัน)"""
        sg = compose(self.b, self.raw(p), p)
        mask = (self.t >= t0) & (self.t < t1)
        el_l = mask & sg["up"] & p.allow_long
        el_s = mask & sg["down"] & p.allow_short
        rate = (sg["long"][mask].sum() + sg["short"][mask].sum()) / max(int(el_l.sum() + el_s.sum()), 1)
        p_rand = p.with_overrides(stop="atr")   # การสุ่มไม่มีโครงสร้างรูปแบบ -> ใช้ SL ตาม ATR
        out = np.zeros(seeds)
        for s in range(seeds):
            u = rng.random(self.t.size) < rate
            r, _ = self.run(p_rand, t0, t1, long_sig=el_l & u, short_sig=el_s & u)
            out[s] = r.mean() if r.size else 0.0
        return out


# ----------------------------------------------------------------------------
# walk-forward
# ----------------------------------------------------------------------------
def run_dataset(symbol: str, interval: str, train_months: int = 36, test_months: int = 12,
                min_trades: int = 10, seeds: int = 200, anchored: bool = False,
                data_dir: Path = DATA_DIR) -> dict:
    ds = Dataset(symbol, interval, data_dir)
    wins = make_windows(ds.t, train_months, test_months, anchored=anchored)
    grid = [ds.params(c) for c in combos()]
    grid_dicts = combos()
    rng = np.random.default_rng(42)
    fixed = ds.base_p

    rows, sel_r, fix_r, curves_sel, curves_fix = [], [], [], [], []
    rand_r: list[list[float]] = [[] for _ in range(seeds)]
    for tr0, tr1, te0, te1 in wins:
        best_i, best_score, best_n = -1, -np.inf, 0
        for gi, p in enumerate(grid):
            r, _ = ds.run(p, tr0, tr1)
            if r.size >= min_trades:
                sc = sqn(r)
                if sc > best_score:
                    best_i, best_score, best_n = gi, sc, r.size
        p = grid[best_i] if best_i >= 0 else fixed
        r, eq = ds.run(p, te0, te1)
        rf, eqf = ds.run(fixed, te0, te1)
        sel_r.append(r)
        fix_r.append(rf)
        curves_sel.append(eq)
        curves_fix.append(eqf)
        # random : เก็บ R ทุกไม้ของทุก seed ไว้รวมข้ามช่วง
        sg = compose(ds.b, ds.raw(p), p)
        mask = (ds.t >= te0) & (ds.t < te1)
        el_l = mask & sg["up"] & p.allow_long
        el_s = mask & sg["down"] & p.allow_short
        rate = (sg["long"][mask].sum() + sg["short"][mask].sum()) / max(int(el_l.sum() + el_s.sum()), 1)
        p_rand = p.with_overrides(stop="atr")
        for s in range(seeds):
            u = rng.random(ds.t.size) < rate
            rr, _ = ds.run(p_rand, te0, te1, long_sig=el_l & u, short_sig=el_s & u)
            rand_r[s].extend(rr.tolist())

        start, end = int(np.searchsorted(ds.t, te0)), int(np.searchsorted(ds.t, te1))
        rows.append({
            "symbol": symbol, "interval": interval, "test_from": _ym(te0), "test_to": _ym(te1 - 1),
            "chosen": "fallback" if best_i < 0 else json.dumps(grid_dicts[best_i], sort_keys=True),
            "train_sqn": None if best_i < 0 else best_score, "train_trades": best_n,
            "test_trades": int(r.size), "test_r": float(r.sum()),
            "test_ret_pct": float((eq[-1] / eq[0] - 1) * 100) if eq.size else 0.0,
            "fixed_r": float(rf.sum()),
            "buy_hold_pct": float((ds.close[end - 1] / ds.close[start] - 1) * 100) if end > start else 0.0,
        })

    sel = np.concatenate(sel_r) if sel_r else np.empty(0)
    fix = np.concatenate(fix_r) if fix_r else np.empty(0)
    rand_means = np.array([np.mean(x) if x else 0.0 for x in rand_r])
    ret_sel, dd_sel = _stitch(curves_sel)
    ret_fix, dd_fix = _stitch(curves_fix)
    first = int(np.searchsorted(ds.t, wins[0][2])) if wins else 0
    summary = {
        "symbol": symbol, "interval": interval, "windows": len(wins), "combos": len(grid),
        "oos_from": rows[0]["test_from"] if rows else None, "oos_to": rows[-1]["test_to"] if rows else None,
        **{f"wf_{k}": v for k, v in _stats(sel).items()},
        "wf_return_pct": ret_sel, "wf_max_dd_pct": dd_sel,
        **{f"fixed_{k}": v for k, v in _stats(fix).items()},
        "fixed_return_pct": ret_fix, "fixed_max_dd_pct": dd_fix,
        "random_avg_r_median": float(np.median(rand_means)),
        "wf_beats_random_pct": float((rand_means < (sel.mean() if sel.size else 0.0)).mean() * 100),
        "buy_hold_pct": float((ds.close[-1] / ds.close[first] - 1) * 100) if wins else 0.0,
    }
    return {"windows": rows, "summary": summary}


def _job(args):
    return run_dataset(*args)


def _out_dir(data_dir: Path, name: str) -> Path:
    out = data_dir / "results" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_all(symbols: list[str], intervals: list[str], train_months: int = 36, test_months: int = 12,
            min_trades: int = 10, seeds: int = 200, anchored: bool = False,
            data_dir: Path = DATA_DIR, workers: int | None = None):
    jobs = [(s, i, train_months, test_months, min_trades, seeds, anchored, data_dir) for s in symbols for i in intervals]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_job, jobs))
    windows = pl.DataFrame([w for r in res for w in r["windows"]])
    summary = pl.DataFrame([r["summary"] for r in res])
    out = _out_dir(data_dir, "walkforward")
    windows.write_csv(out / "windows.csv")
    summary.write_csv(out / "summary.csv")
    (out / "config.json").write_text(json.dumps({
        "symbols": symbols, "intervals": intervals, "train_months": train_months, "test_months": test_months,
        "min_trades": min_trades, "seeds": seeds, "anchored": anchored, "entries": ENTRY_VARIANTS,
        "filters": FILTER_GRID, "exits": EXIT_VARIANTS, "base": CoreParams().to_dict(),
    }, indent=2))
    return windows, summary, out


# ----------------------------------------------------------------------------
# rule scan : เปลี่ยนทีละอย่างจากชุดอ้างอิง
# ----------------------------------------------------------------------------
def rule_variants() -> dict[str, dict]:
    v: dict[str, dict] = {"ref (breakout20·htf_ema·chandelier3)": {}}
    for ev in ENTRY_VARIANTS:
        name = "entry " + ev["entry"] + "".join(f" {k}={ev[k]}" for k in ev if k != "entry")
        v[name] = ev
    for e in PATTERNS:   # แบบที่บทความสอน: SL ใต้โครงสร้าง + เป้า measured move
        v[f"entry {e} ตามบทความ (SL โครงสร้าง + เป้า)"] = {"entry": e, "stop": "structure", "exit": "measured"}
    for rg in ("golden_cross", "ma200", "none"):
        v[f"regime {rg}"] = {"regime": rg}
    v["filter ADX>=25"] = {"adx_min": 25.0}
    v["filter volume>1.5x"] = {"vol_confirm": True}
    v["exit chandelier 5 ATR"] = {"exit": "chandelier", "trail_atr": 5.0}
    v["exit parabolic SAR"] = {"exit": "psar"}
    v["exit TP 3R คงที่"] = {"exit": "target", "target_r": 3.0}
    v["short ด้วย"] = {"allow_short": True}
    return v


def _rules_job(args) -> list[dict]:
    symbol, interval, seeds, split, data_dir = args
    ds = Dataset(symbol, interval, data_dir)
    t0 = _ms(data_start(ds.t))
    t1 = int(ds.t[-1]) + 1
    s_ms = _ms(datetime.fromisoformat(split).replace(tzinfo=timezone.utc))
    rng = np.random.default_rng(7)
    out = []
    for name, over in rule_variants().items():
        p = ds.params(over)
        r, eq = ds.run(p, t0, t1)
        r_is, _ = ds.run(p, t0, s_ms)
        r_oos, _ = ds.run(p, s_ms, t1)
        rm = ds.random_means(p, t0, t1, seeds, rng)
        st = _stats(r)
        out.append({
            "symbol": symbol, "interval": interval, "rule": name, "from": _ym(t0),
            **st, "return_pct": float((eq[-1] / eq[0] - 1) * 100) if eq.size else 0.0,
            "avg_r_is": float(r_is.mean()) if r_is.size else None,
            "avg_r_oos": float(r_oos.mean()) if r_oos.size else None,
            "random_avg_r": float(np.median(rm)),
            "beats_random_pct": float((rm < (r.mean() if r.size else 0.0)).mean() * 100),
        })
    return out


def run_rules(symbols: list[str], intervals: list[str], seeds: int = 200, split: str = "2023-01-01",
              data_dir: Path = DATA_DIR, workers: int | None = None):
    jobs = [(s, i, seeds, split, data_dir) for s in symbols for i in intervals]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = pl.DataFrame([row for rows in ex.map(_rules_job, jobs) for row in rows])
    out = _out_dir(data_dir, "rules")
    res.write_csv(out / "rules.csv")
    return res, out


# ----------------------------------------------------------------------------
# พิมพ์ผล
# ----------------------------------------------------------------------------
def _cfg():
    return pl.Config(tbl_rows=300, tbl_cols=30, tbl_width_chars=260, float_precision=2, fmt_str_lengths=110,
                     tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)


def print_report(windows: pl.DataFrame, summary: pl.DataFrame) -> None:
    with _cfg():
        print("\n=== แต่ละช่วง test ===")
        print(windows.select("symbol", "interval", "test_from", "test_to", "chosen", "train_sqn",
                             "test_trades", "test_r", "fixed_r", "test_ret_pct", "buy_hold_pct"))
        print("\n=== สรุปช่วง out-of-sample ต่อกันทั้งหมด ===")
        print(summary.select("symbol", "interval", "oos_from", "oos_to", "combos", "wf_trades", "wf_total_r",
                             "wf_avg_r", "wf_win_rate", "wf_pf", "wf_sqn", "wf_max_dd_r", "wf_return_pct",
                             "wf_max_dd_pct"))
        print(summary.select("symbol", "interval", "fixed_trades", "fixed_total_r", "fixed_avg_r", "fixed_sqn",
                             "fixed_return_pct", "fixed_max_dd_pct", "random_avg_r_median", "wf_beats_random_pct",
                             "buy_hold_pct"))


def print_rules(res: pl.DataFrame) -> None:
    res = res.with_columns((pl.col("symbol").str.replace("USDT", "") + " " + pl.col("interval")).alias("ds"))
    # SQN จากไม้ไม่ถึง 10 ไม้ไม่มีความหมาย -> แสดงแค่จำนวนไม้
    cell = pl.when(pl.col("trades") < 10).then(pl.format("(น้อยไป) n{}", pl.col("trades"))).otherwise(
        pl.format("{} · {}% · n{}", pl.col("sqn").round(1), pl.col("beats_random_pct").round(0).cast(pl.Int64),
                  pl.col("trades")))
    with _cfg():
        print("\n=== SQN · ชนะการสุ่ม % · จำนวนไม้  (ทั้งช่วงข้อมูล) ===")
        print(res.with_columns(cell.alias("v")).pivot(on="ds", index="rule", values="v"))
        print("\n=== R เฉลี่ยต่อไม้ : ก่อน 2023 → ตั้งแต่ 2023 ===")
        stab = pl.format("{} → {}", pl.col("avg_r_is").round(2), pl.col("avg_r_oos").round(2))
        print(res.with_columns(stab.alias("v")).pivot(on="ds", index="rule", values="v"))
