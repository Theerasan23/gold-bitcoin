"""พอร์ตรวม : หลายระบบ (sleeve) หลายเหรียญ ใช้เงินก้อนเดียว

แต่ละ sleeve = (เหรียญ, ระบบ) รันแยกเพื่อเอา "รายการไม้" (เวลา/ราคาเข้า-ออก/ระยะ SL)
จากนั้นจำลองพอร์ตใหม่บนเส้นเวลาเดียวกัน:
  - ขนาดไม้ = เสี่ยง risk_pct % ของ equity ทั้งพอร์ต ณ ตอนเข้า
  - มูลค่าที่ถือรวมทุกไม้ ≤ max_lev เท่าของ equity (spot = 1 เท่า) ถ้าเกินจะลดขนาดไม้ใหม่ลง
  - mark-to-market ทุกแท่งด้วยราคาปิด -> drawdown จริงระหว่างถือ
จังหวะเข้า/ออกไม่ขึ้นกับขนาดไม้ จึงใช้รายการไม้จากการรันเดี่ยวได้ตรง ๆ
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import data as dt
from .config import DATA_DIR
from .core import CoreParams, net_r, sqn
from .core import simulate as core_sim
from .meanrev import MRParams
from .meanrev import signals as mr_signals
from .meanrev import simulate as mr_sim
from .walkforward import Dataset, _ms, data_start

MS_DAY = 86_400_000

# ระบบที่ใช้ — ตั้งค่าตายตัวไว้ก่อน ไม่ได้ปรับจากผลพอร์ต
TREND = {"entry": "breakout", "breakout_len": 20, "regime": "htf_ema", "exit": "chandelier", "trail_atr": 5.0}
RANGE = {"entry": "bb_fade", "regime": "adx20"}


@dataclass
class Sleeve:
    name: str
    symbol: str
    t: np.ndarray          # เวลาแท่งของเหรียญนี้ (ms)
    close: np.ndarray
    trades: list[dict]     # {entry_ms, side, entry_px, risk, exits: [(exit_ms, exit_px, frac)]}
    r: np.ndarray          # R สุทธิต่อไม้ (จากการรันเดี่ยว) เรียงตามเวลาเข้าเหมือน trades

    def r_between(self, x0: int, x1: int) -> np.ndarray:
        e = np.array([tr["entry_ms"] for tr in self.trades], dtype=np.int64)
        return self.r[(e >= x0) & (e < x1)] if e.size else self.r


def _trades_from_T(T: np.ndarray, t: np.ndarray) -> list[dict]:
    by_entry: dict[int, dict] = {}
    for row in T:
        eb = int(row[0])
        tr = by_entry.setdefault(eb, {"entry_ms": int(t[eb]), "side": int(row[2]), "entry_px": float(row[3]),
                                      "risk": float(row[9]), "exits": []})
        tr["exits"].append((int(t[int(row[1])]), float(row[4]), float(row[6])))
    return sorted(by_entry.values(), key=lambda x: x["entry_ms"])


def build_sleeve(kind: str, symbol: str, interval: str, t0: int, t1: int, data_dir: Path = DATA_DIR,
                 overrides: dict | None = None) -> Sleeve:
    ds = Dataset(symbol, interval, data_dir)
    mask = (ds.t >= t0) & (ds.t < t1)
    end = int(np.searchsorted(ds.t, t1))
    if kind == "trend":
        p = ds.params({**TREND, **(overrides or {})})
        T, _, _ = core_sim(ds.b, ds.raw(p), p, mask, end)
    else:
        p = MRParams().with_overrides(tick_size=dt.tick_size(symbol, data_dir), **{**RANGE, **(overrides or {})})
        T, _, _ = mr_sim(ds.b, mr_signals(ds.b, p), p, mask, end)
    short = symbol.replace("USDT", "")
    return Sleeve(f"{short}-{kind}", symbol, ds.t, ds.close, _trades_from_T(T, ds.t), net_r(T))


def closes_on(timeline: np.ndarray, s: Sleeve) -> np.ndarray:
    """ราคาปิดล่าสุดของเหรียญ ณ ทุกจุดบนเส้นเวลา (ก่อนมีข้อมูล = NaN)"""
    idx = np.searchsorted(s.t, timeline, side="right") - 1
    return np.where(idx >= 0, s.close[np.clip(idx, 0, None)], np.nan)


def simulate_portfolio(sleeves: list[Sleeve], t0: int, t1: int, risk_pct: float = 1.0, max_lev: float = 1.0,
                       comm_pct: float = 0.10, capital: float = 10_000) -> dict:
    timeline = np.unique(np.concatenate([s.t[(s.t >= t0) & (s.t < t1)] for s in sleeves]))
    # ราคาปิดล่าสุดของแต่ละเหรียญ ณ ทุกจุดของเส้นเวลา (เหรียญที่ยังไม่มีข้อมูล = NaN)
    closes = {}
    for s in sleeves:
        if s.symbol not in closes:
            closes[s.symbol] = closes_on(timeline, s)
    comm = comm_pct / 100.0

    entries: dict[int, list] = {}
    for si, s in enumerate(sleeves):
        for tr in s.trades:
            entries.setdefault(tr["entry_ms"], []).append((si, tr))
    pos_k = {int(x): k for k, x in enumerate(timeline)}

    cash = capital
    open_pos: list[dict] = []
    exits_at: dict[int, list] = {}
    equity = np.empty(timeline.size)
    exposure = np.zeros(timeline.size)
    skipped = 0
    eq_prev = capital
    for k, tk in enumerate(timeline):
        tk = int(tk)
        # 1) เข้าไม้ที่ราคาเปิดแท่งนี้ — ขนาดจาก equity ล่าสุด
        for si, tr in entries.get(tk, []):
            sym = sleeves[si].symbol
            qty = risk_pct / 100.0 * eq_prev / tr["risk"]
            held = sum(p["qty"] * closes[p["sym"]][k - 1 if k else 0] for p in open_pos)
            room = max_lev * eq_prev - held
            if qty * tr["entry_px"] > room:
                qty = max(room, 0.0) / tr["entry_px"]
            if qty <= 0:
                skipped += 1
                continue
            pos = {"sym": sym, "side": tr["side"], "entry_px": tr["entry_px"], "qty": qty, "qty0": qty, "sleeve": si}
            cash -= tr["entry_px"] * qty * comm
            open_pos.append(pos)
            for ex_ms, ex_px, frac in tr["exits"]:
                exits_at.setdefault(ex_ms, []).append((pos, ex_px, frac))
        # 2) ออกไม้ระหว่างแท่ง (SL / เป้า / สัญญาณ)
        for pos, ex_px, frac in exits_at.pop(tk, []):
            q = min(pos["qty0"] * frac, pos["qty"])
            cash += q * (ex_px - pos["entry_px"]) * pos["side"] - ex_px * q * comm
            pos["qty"] -= q
        open_pos = [p for p in open_pos if p["qty"] > 1e-12]
        # 3) mark-to-market ที่ราคาปิด
        unreal = sum(p["qty"] * (closes[p["sym"]][k] - p["entry_px"]) * p["side"] for p in open_pos)
        equity[k] = cash + unreal
        exposure[k] = sum(p["qty"] * closes[p["sym"]][k] for p in open_pos) / equity[k]
        eq_prev = equity[k]
    return {"time": timeline, "equity": equity, "exposure": exposure, "skipped": skipped, "closes": closes}


def _period(res: dict, x0: int, x1: int, label: str, bench: dict[str, np.ndarray]) -> dict:
    t, eq = res["time"], res["equity"]
    m = (t >= x0) & (t < x1)
    if m.sum() < 2:
        return {"period": label}
    e = eq[m]
    tt = t[m]
    days = (tt[-1] - tt[0]) / MS_DAY
    ret = e[-1] / e[0] - 1
    last_of_day = np.r_[np.nonzero(np.diff(tt // MS_DAY))[0], tt.size - 1]
    d = e[last_of_day]
    dr = np.diff(d) / d[:-1]
    sharpe = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.size > 2 and dr.std() > 0 else 0.0
    dd = float((1 - e / np.maximum.accumulate(e)).max() * 100)
    cagr = ((1 + ret) ** (365.25 / days) - 1) * 100 if days > 0 and ret > -1 else 0.0
    out = {
        "period": label, "return_pct": ret * 100, "cagr_pct": cagr, "max_dd_pct": dd,
        "mar": cagr / dd if dd > 0 else None, "sharpe": sharpe,
        "avg_exposure_pct": float(res["exposure"][m].mean() * 100),
    }
    for name, b in bench.items():
        bb = b[m]
        out[f"{name}_pct"] = float((bb[-1] / bb[0] - 1) * 100)
        out[f"{name}_dd_pct"] = float((1 - bb / np.maximum.accumulate(bb)).max() * 100)
    return out


def daily_returns(res: dict) -> pl.DataFrame:
    t, eq = res["time"], res["equity"]
    last = np.r_[np.nonzero(np.diff(t // MS_DAY))[0], t.size - 1]
    return pl.DataFrame({"day": t[last] // MS_DAY, "eq": eq[last]}).with_columns(pl.col("eq").pct_change().alias("ret"))


def run(symbols: list[str], interval: str = "4h", split: str = "2023-01-01", risk_pct: float = 1.0,
        max_lev: float = 1.0, data_dir: Path = DATA_DIR):
    # ช่วงร่วม : เริ่มเมื่อทุกเหรียญมีข้อมูลและผ่านช่วงอุ่นเครื่องแล้ว
    starts = []
    ends = []
    for s in symbols:
        t = dt.load(s, interval, data_dir)["open_time"].dt.epoch("ms").to_numpy()
        starts.append(_ms(data_start(t)))
        ends.append(int(t[-1]) + 1)
    t0, t1 = max(starts), min(ends)
    s_ms = _ms(datetime.fromisoformat(split).replace(tzinfo=timezone.utc))

    sleeves = {f"{s.replace('USDT', '')}-{k}": build_sleeve(k, s, interval, t0, t1, data_dir)
               for s in symbols for k in ("trend", "range")}
    names = list(sleeves)
    trend = [n for n in names if n.endswith("-trend")]
    rng = [n for n in names if n.endswith("-range")]
    combos = {**{n: [n] for n in names}, "รวม trend": trend, "รวม range": rng, "รวมทั้งหมด": names}
    for s in symbols:   # trend + range ของเหรียญเดียว
        k = s.replace("USDT", "")
        combos[f"{k} trend+range"] = [f"{k}-trend", f"{k}-range"]

    rows, curves, singles = [], {}, {}
    for cname, members in combos.items():
        res = simulate_portfolio([sleeves[m] for m in members], t0, t1, risk_pct, max_lev)
        timeline = res["time"]
        px = {s: closes_on(timeline, sleeves[f"{s.replace('USDT', '')}-trend"]) for s in symbols}
        bench = {f"bh_{s.replace('USDT', '').lower()}": px[s] for s in symbols}
        bench["bh_5050"] = sum(px[s] / px[s][0] for s in symbols) / len(symbols)   # ซื้อครึ่ง-ครึ่งแล้วถือ
        for label, x0, x1 in (("all", t0, t1), ("in_sample", t0, s_ms), ("out_sample", s_ms, t1)):
            r = np.concatenate([sleeves[m].r_between(x0, x1) for m in members])
            rows.append({"portfolio": cname, "sleeves": len(members), "trades": int(r.size), "sqn": sqn(r),
                         "avg_r": float(r.mean()) if r.size else 0.0,
                         "skipped": res["skipped"], **_period(res, x0, x1, label, bench)})
        curves[cname] = pl.DataFrame({"time": timeline, "equity": res["equity"]})
        if cname == "รวม trend":   # เส้นเทียบ : ซื้อแล้วถือ ช่วงเดียวกัน
            for bname, bb in bench.items():
                curves[bname] = pl.DataFrame({"time": timeline, "equity": bb / bb[0] * 10_000})
        if len(members) == 1:
            singles[cname] = daily_returns(res).select("day", pl.col("ret").alias(cname))

    corr_df = None
    if singles:
        joined = None
        for df in singles.values():
            joined = df if joined is None else joined.join(df, on="day", how="full", coalesce=True)
        corr_df = joined.drop("day").fill_null(0.0).drop_nans().corr().with_columns(pl.Series("sleeve", list(singles)))

    out = data_dir / "results" / f"portfolio-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows)
    table.write_csv(out / "portfolio.csv")
    pl.concat([c.with_columns(pl.lit(n).alias("portfolio")) for n, c in curves.items()]).write_parquet(
        out / "equity.parquet")
    if corr_df is not None:
        corr_df.write_csv(out / "correlation.csv")
    (out / "config.json").write_text(json.dumps({
        "symbols": symbols, "interval": interval, "risk_pct": risk_pct, "max_lev": max_lev, "split": split,
        "trend": TREND, "range": RANGE, "from_ms": t0, "to_ms": t1,
    }, indent=2, ensure_ascii=False))
    return table, corr_df, out


def print_report(table: pl.DataFrame, corr: pl.DataFrame | None) -> None:
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250, float_precision=2,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        for per in ("all", "in_sample", "out_sample"):
            part = table.filter(pl.col("period") == per)
            cols = [c for c in ("portfolio", "trades", "avg_r", "sqn", "return_pct", "cagr_pct", "max_dd_pct", "mar", "sharpe",
                                "avg_exposure_pct", "bh_5050_pct", "bh_5050_dd_pct") if c in part.columns]
            print(f"\n=== พอร์ต · {per} ===")
            print(part.select(cols))
        if corr is not None:
            print("\n=== correlation ของผลตอบแทนรายวันระหว่าง sleeve ===")
            print(corr.select(["sleeve"] + [c for c in corr.columns if c != "sleeve"]))
