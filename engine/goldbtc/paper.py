"""บัญชีเดโม (paper trading) — เงินจำลอง คำสั่งจำลอง ราคาจริงจาก Binance (ตลาดสาธารณะ อ่านอย่างเดียว ไม่มีบัญชี)

เป้าหมาย : เก็บข้อมูลการทำงานจริงไว้เทียบกับ backtest ช่วงเวลาเดียวกัน
  - กฎเหมือน backtest ทุกข้อ (Engine ตัวเดียวกันใช้ทั้ง replay และ live) : ระบบ trend ของพอร์ต (portfolio.TREND)
      เข้าเมื่อราคาแตะจุด breakout (คำสั่ง stop รอไว้) เข้าเพิ่มได้ทุกครั้งที่ทำ high ใหม่เหนือไม้ล่าสุด
      แต่ละไม้มี SL ของตัวเอง · ทิศกลับปิดทุกไม้
  - ต่างกันแค่ "ราคาที่ได้" :
      เข้า/SL : backtest มองทั้งแท่งเป็นช่วงเดียว (open->high->low->close) · เดโมเดินทีละแท่ง 1 นาทีที่ปิดแล้ว
      ทิศกลับ : backtest ปิดที่ราคาเปิดแท่ง · เดโมใช้ราคาจริงตอนที่ worker เห็นว่าแท่งก่อนหน้าปิด (+ slippage)
  - เก็บทั้งราคาที่ได้ และราคาที่ backtest สมมติ (ref_price) เพื่อวัดความต่าง

ไฟล์ (data/paper/runs/<run_id>/) : config.json · state.json · events.jsonl · trades.jsonl · equity.jsonl
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from . import data as dt
from .config import DATA_DIR, INTERVAL_SEC
from .core import CoreParams, base_features, touch_inputs
from .portfolio import TREND

MIN_MS = 60_000


def paper_dir(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "paper"


@dataclass
class PaperConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "PAXGUSDT"])
    interval: str = "4h"
    capital: float = 10_000
    risk_pct: float = 1.0
    max_lev: float = 1.0
    commission_pct: float = 0.10
    slippage_ticks: int = 2
    strategy: dict = field(default_factory=lambda: dict(TREND))


@dataclass
class Unit:
    """ไม้ 1 ไม้ (เหรียญเดียวถือได้หลายไม้) · qty 0 = มีสัญญาณแต่เงินไม่พอ (นับเป็นสัญญาณ ไม่ได้ถือจริง)"""
    symbol: str
    side: int
    qty: float
    entry_price: float           # ราคาที่ได้จริง (รวม slippage)
    ref_price: float | None      # ราคาที่ backtest สมมติ (มองทั้งแท่งเป็นช่วงเดียว)
    entry_time: int              # ms เวลาที่เข้า (นาทีที่ราคาแตะ)
    entry_bar: int               # ms เวลาเปิดของแท่งที่เข้า
    risk: float                  # ระยะ SL เริ่มต้น (ราคา)
    init_stop: float
    stop: float


@dataclass
class Bar:
    time: int
    open: float
    high: float
    low: float
    close: float
    atr: float
    # รู้ได้ตั้งแต่ราคาเปิดแท่ง (core.touch_inputs)
    level_l: float   # จุดแตะ long = High สูงสุด n แท่งก่อนหน้า
    level_s: float
    arm_l: bool      # เข้า long ได้ (ทิศ + ตัวกรอง)
    arm_s: bool
    exit_l: bool     # ทิศกลับ -> ปิดทุกไม้ long ที่ราคาเปิด
    exit_s: bool
    risk: float      # ระยะ SL ของไม้ที่จะเข้าในแท่งนี้


def _num(x: float) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


class Engine:
    """สถานะบัญชี + กฎของระบบ — ไม่รู้จักเวลาจริงหรือเครือข่าย (ทดสอบ/replay ได้ตรง ๆ)

    กฎ (เหมือน backtest.simulate_touch ทุกข้อ) :
      - เปิดแท่ง : ทิศกลับ -> ปิดทุกไม้ · ตั้งจุดแตะของแท่งนี้ = max(High 20 แท่งก่อน, ราคาเข้าไม้ล่าสุด)
      - ระหว่างแท่ง : ราคาแตะจุด -> เข้าทันที (แท่งละ 1 ไม้ ไม่ปิดไม้เดิม) · ราคาแตะ SL ของไม้ไหน ปิดไม้นั้น
      - ปิดแท่ง : เลื่อน trailing (Chandelier จากจุดสูงสุดตั้งแต่ไม้แรก) ให้ทุกไม้
    on_price รับช่วงราคา (open, high, low, close) จะเป็นแท่ง 1 นาที (live) หรือทั้งแท่ง (replay) ก็ได้"""

    def __init__(self, cfg: PaperConfig, state: dict | None = None, sink=None, tick: dict | None = None):
        self.cfg = cfg
        self.sp = CoreParams().with_overrides(**{**TREND, **cfg.strategy})
        if self.sp.entry_on != "touch" or self.sp.exit != "chandelier":
            raise ValueError("บัญชีเดโมรองรับเฉพาะ entry_on=touch + exit=chandelier")
        st = state or {}
        self.cash: float = st.get("cash", cfg.capital)
        self.units: list[Unit] = []
        self.camp: dict[str, dict] = st.get("campaigns", {})   # เหรียญ -> {side, last_px, ext, trail, start}
        pos = st.get("positions", [])
        if isinstance(pos, dict):   # state รุ่นก่อน : ถือได้เหรียญละไม้
            for sym, v in pos.items():
                self.units.append(Unit(**{k: v[k] for k in Unit.__dataclass_fields__}))
                self.camp[sym] = {"side": v["side"], "last_px": v["entry_price"], "ext": v.get("ext"),
                                  "trail": v.get("trail"), "start": v["entry_bar"]}
        else:
            self.units = [Unit(**u) for u in pos + st.get("ghosts", [])]
        self.arm: dict[str, dict] = st.get("arm", {})            # เหรียญ -> จุดแตะของแท่งที่กำลังวิ่ง
        self.last_close: dict[str, float] = st.get("last_close", {})
        self.eq_prev: float = st.get("eq_prev", cfg.capital)
        self.bar_t: int | None = st.get("bar_t")
        self.freed: float = st.get("freed", 0.0)
        self.tick = tick or st.get("tick", {})
        self.sink = sink or (lambda kind, rec: None)

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        return {"cash": self.cash,
                "positions": [asdict(u) for u in self.units if u.qty > 0],
                "ghosts": [asdict(u) for u in self.units if u.qty <= 0],
                "campaigns": self.camp, "arm": self.arm, "last_close": self.last_close, "eq_prev": self.eq_prev,
                "bar_t": self.bar_t, "freed": self.freed, "tick": self.tick}

    def _slip(self, sym: str) -> float:
        return self.cfg.slippage_ticks * self.tick.get(sym, 0.01)

    def _comm(self) -> float:
        return self.cfg.commission_pct / 100.0

    def _event(self, t_ms: int, kind: str, sym: str | None = None, **kw):
        self.sink("events", {"time": t_ms, "type": kind, "symbol": sym, **kw})

    def of(self, sym: str) -> list[Unit]:
        return [u for u in self.units if u.symbol == sym]

    # ---------------------------------------------------------------- actions
    def begin_bar(self, t: int) -> None:
        """แท่งใหม่ (ทุกเหรียญพร้อมกัน) : ที่ว่างสำหรับไม้ใหม่คิดจากไม้ที่ถือตอนต้นแท่ง เหมือน backtest พอร์ต"""
        if self.bar_t != t:
            self.bar_t = t
            self.freed = 0.0

    def on_bar_open(self, sym: str, b: Bar, t_ms: int, price: float, ref: float | None, enter: bool = True) -> None:
        """ราคาเปิดแท่ง : ทิศกลับ -> ปิดทุกไม้ที่ราคา price · ตั้งจุดแตะของแท่งนี้"""
        camp = self.camp.get(sym)
        if camp and ((camp["side"] == 1 and b.exit_l) or (camp["side"] == -1 and b.exit_s)):
            self._event(t_ms, "signal_exit", sym, reason="trend_flip")
            for u in self.of(sym):
                self._close(u, t_ms, price - u.side * self._slip(sym), "trend_flip", ref)
        camp = self.camp.get(sym)
        side = camp["side"] if camp else 0
        ok = enter and b.risk > 0 and not math.isnan(b.risk) and \
            (self.sp.max_units == 0 or len(self.of(sym)) < self.sp.max_units)
        trig_l = trig_s = None
        if ok and b.arm_l and side >= 0 and not math.isnan(b.level_l):
            trig_l = b.level_l if side == 0 else max(b.level_l, camp["last_px"])
        if ok and b.arm_s and side <= 0 and not math.isnan(b.level_s):
            trig_s = b.level_s if side == 0 else min(b.level_s, camp["last_px"])
        self.arm[sym] = {"bar": b.time, "open": ref, "risk": _num(b.risk), "trig_l": trig_l, "trig_s": trig_s,
                         "used": trig_l is None and trig_s is None}

    def on_price(self, sym: str, t_ms: int, o: float, h: float, l: float, c: float) -> None:
        """ราคาเดินในช่วงหนึ่ง : open -> high -> low -> close ถ้า open ใกล้ high กว่า ไม่งั้น open -> low -> high -> close"""
        arm = self.arm.get(sym)
        pts = (o, h, l, c) if (h - o) <= (o - l) else (o, l, h, c)
        a = o
        for k, b in enumerate(pts):
            gap = k == 0
            up, dn = gap or b > a, gap or b < a
            for u in self.of(sym):
                if u.side == 1 and dn and b <= u.stop:
                    px = (b if gap else u.stop) - self._slip(sym)
                elif u.side == -1 and up and b >= u.stop:
                    px = (b if gap else u.stop) + self._slip(sym)
                else:
                    continue
                self._close(u, t_ms, px, "stop" if u.stop == u.init_stop else "trailing", u.stop)
            if arm and not arm["used"]:
                if arm["trig_l"] is not None and up and b > arm["trig_l"]:
                    self._enter(sym, 1, b if gap else arm["trig_l"], arm["trig_l"], t_ms, arm)
                elif arm["trig_s"] is not None and dn and b < arm["trig_s"]:
                    self._enter(sym, -1, b if gap else arm["trig_s"], arm["trig_s"], t_ms, arm)
            a = b

    def on_bar_close(self, sym: str, b: Bar) -> None:
        """ปิดแท่ง : เลื่อน trailing stop ร่วมของทุกไม้ในเหรียญนี้ (เลื่อนเข้าหาราคาอย่างเดียว)"""
        camp = self.camp.get(sym)
        mine = self.of(sym)
        if camp and mine:
            s = camp["side"]
            first = camp["start"] == b.time or camp["ext"] is None
            ext = (b.high if s == 1 else b.low) if first else (max if s == 1 else min)(camp["ext"], b.high if s == 1 else b.low)
            camp["ext"] = ext
            ch = ext - s * b.atr * self.sp.trail_atr
            if not math.isnan(ch):
                camp["trail"] = ch if camp["trail"] is None else (max if s == 1 else min)(camp["trail"], ch)
            real = [u for u in mine if u.qty > 0]
            old = self._nearest(real, s)
            if camp["trail"] is not None:
                for u in mine:
                    u.stop = (max if s == 1 else min)(u.stop, camp["trail"])
            new = self._nearest(real, s)
            if real and new != old:
                self._event(b.time, "stop_moved", sym, old=old, new=new, units=len(real))
        self.last_close[sym] = b.close

    @staticmethod
    def _nearest(units: list[Unit], side: int) -> float | None:
        return (max if side == 1 else min)(u.stop for u in units) if units else None

    def mark(self, prices: dict[str, float]) -> float:
        unreal = sum(u.qty * (prices.get(u.symbol, u.entry_price) - u.entry_price) * u.side for u in self.units)
        return self.cash + unreal

    def _enter(self, sym: str, side: int, raw: float, trig: float, t_ms: int, arm: dict) -> None:
        arm["used"] = True
        fill = raw + side * self._slip(sym)
        rk = arm["risk"]
        E = self.eq_prev
        qty = self.cfg.risk_pct / 100.0 * E / rk
        held = sum(u.qty * self.last_close.get(u.symbol, u.entry_price) for u in self.units) + self.freed
        room = self.cfg.max_lev * E - held
        capped = qty * fill > room
        if capped:
            qty = max(room, 0.0) / fill
        qty = max(qty, 0.0)
        camp = self.camp.get(sym)
        if camp is None:
            self.camp[sym] = {"side": side, "last_px": raw, "ext": None, "trail": None, "start": arm["bar"]}
        else:
            camp["last_px"] = raw
        bo = arm["open"]
        ref = None if bo is None else (max(bo, trig) if side == 1 else min(bo, trig))
        stop = fill - side * rk
        self.units.append(Unit(sym, side, qty, fill, ref, t_ms, arm["bar"], rk, stop, stop))
        n = len([u for u in self.of(sym) if u.qty > 0])
        self._event(t_ms, "signal", sym, trigger=trig, unit=len(self.of(sym)))
        if qty <= 0:
            self._event(t_ms, "skip", sym, reason="เงินไม่พอ (ถือเต็มเพดานแล้ว)")
            return
        self.cash -= fill * qty * self._comm()
        self._event(t_ms, "entry", sym, price=fill, ref_price=ref, qty=qty, stop=stop, risk_money=qty * rk,
                    capped=capped, units=n, signal_bar=arm["bar"])

    def _close(self, u: Unit, t_ms: int, px: float, reason: str, ref: float | None = None) -> None:
        self.units.remove(u)
        sym = u.symbol
        if not self.of(sym):
            self.camp.pop(sym, None)
        if u.qty <= 0:
            return
        self.freed += u.qty * self.last_close.get(sym, u.entry_price)
        comm = self._comm()
        gross = (px - u.entry_price) * u.side * u.qty
        exit_comm = px * u.qty * comm
        pnl = gross - exit_comm - u.entry_price * u.qty * comm
        self.cash += gross - exit_comm
        rec = {
            "symbol": sym, "side": "long" if u.side == 1 else "short", "qty": u.qty,
            "entry_time": u.entry_time, "entry_bar": u.entry_bar, "entry_price": u.entry_price,
            "ref_entry_price": u.ref_price, "exit_time": t_ms, "exit_price": px, "ref_exit_price": ref,
            "reason": reason, "risk": u.risk, "pnl": pnl, "r": pnl / (u.risk * u.qty),
        }
        self.sink("trades", rec)
        self._event(t_ms, "exit", sym, price=px, reason=reason, pnl=pnl, r=rec["r"])


# ----------------------------------------------------------------------------
# สัญญาณจากข้อมูลแท่งเทียน (ใช้ตัวเดียวกับ backtest)
# ----------------------------------------------------------------------------
def signal_frame(df: pl.DataFrame, interval: str, cfg: PaperConfig, tick: float) -> dict:
    """ทุกแท่งที่ปิดแล้ว + แถวสุดท้าย = แท่งที่กำลังวิ่ง (รู้แค่ค่าที่รู้ได้ตอนเปิดแท่ง : จุดแตะ ทิศ ระยะ SL)"""
    p = CoreParams().with_overrides(tick_size=tick, **{**TREND, **cfg.strategy})
    n = df.height
    last = df.tail(1)
    nxt = last.with_columns(
        (pl.col("open_time") + pl.duration(seconds=INTERVAL_SEC[interval])).alias("open_time"),
        *[pl.col("close").alias(k) for k in ("open", "high", "low")],
        pl.lit(0.0).cast(df.schema["volume"]).alias("volume"),
    )
    b = base_features(pl.concat([df, nxt.select(df.columns)]), interval, p)
    x = touch_inputs(b, p)
    return {"n": n, "time": b["time_ms"], "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "atr": b["atr"], **{k: x[k] for k in ("level_l", "level_s", "arm_l", "arm_s", "exit_l", "exit_s", "risk")}}


def bar_at(f: dict, i: int) -> Bar:
    return Bar(int(f["time"][i]), float(f["open"][i]), float(f["high"][i]), float(f["low"][i]),
               float(f["close"][i]), float(f["atr"][i]), float(f["level_l"][i]), float(f["level_s"][i]),
               bool(f["arm_l"][i]), bool(f["arm_s"][i]), bool(f["exit_l"][i]), bool(f["exit_s"][i]),
               float(f["risk"][i]))


def replay(engine: Engine, frames: dict[str, dict], t0: int, t1: int, on_mark=None) -> None:
    """เล่นย้อนหลังด้วยแท่งที่ปิดแล้ว ตามลำดับเดียวกับ backtest พอร์ต :
    ทุกเหรียญ -> (1) เปิดแท่ง (2) ราคาเดินทั้งแท่ง (3) ปิดแท่ง -> mark ราคาปิด"""
    idx = {s: {int(f["time"][i]): i for i in range(f["n"]) if t0 <= f["time"][i] < t1} for s, f in frames.items()}
    timeline = sorted({t for m in idx.values() for t in m})
    for t in timeline:
        here = {s: m[t] for s, m in idx.items() if t in m}
        engine.begin_bar(t)
        for s, i in here.items():
            o = float(frames[s]["open"][i])
            engine.on_bar_open(s, bar_at(frames[s], i), t, o, o)
        for s, i in here.items():
            f = frames[s]
            engine.on_price(s, t, float(f["open"][i]), float(f["high"][i]), float(f["low"][i]), float(f["close"][i]))
        for s, i in here.items():
            engine.on_bar_close(s, bar_at(frames[s], i))
        engine.eq_prev = engine.mark(engine.last_close)
        if on_mark:
            on_mark(t, engine)


# ----------------------------------------------------------------------------
# การเก็บไฟล์
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, rec: dict) -> None:
        with open(self.dir / f"{kind}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read(self, kind: str, limit: int | None = None) -> list[dict]:
        p = self.dir / f"{kind}.jsonl"
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        if limit:
            lines = lines[-limit:]
        return [json.loads(x) for x in lines if x.strip()]

    def save_json(self, name: str, obj: dict) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
        os.replace(tmp, self.dir / name)

    def load_json(self, name: str) -> dict | None:
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None


def current_run(data_dir: Path = DATA_DIR) -> Path | None:
    ptr = paper_dir(data_dir) / "current"
    if not ptr.exists():
        return None
    d = paper_dir(data_dir) / "runs" / ptr.read_text().strip()
    return d if d.exists() else None


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def active_runs(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """บัญชีที่กำลังเดิน : timeframe -> โฟลเดอร์ (เดินพร้อมกันได้ timeframe ละ 1 บัญชี)"""
    root = paper_dir(data_dir)
    f = root / "active.json"
    if f.exists():
        ids = json.loads(f.read_text())
    else:   # รุ่นก่อนมีแค่ current -> ถือเป็นบัญชีที่เดินอยู่
        cur = current_run(data_dir)
        ids = {load_config(cur).interval: cur.name} if cur else {}
    return {tf: root / "runs" / rid for tf, rid in ids.items() if (root / "runs" / rid / "config.json").exists()}


def new_run(cfg: PaperConfig, data_dir: Path = DATA_DIR) -> Path:
    """เริ่มบัญชีเดโมใหม่ — แทนบัญชีเดิมของ timeframe เดียวกัน (รอบเก่าเก็บไว้ ไม่ลบ) TF อื่นเดินต่อ"""
    if cfg.interval not in INTERVAL_SEC:
        raise ValueError(f"interval ไม่รองรับ: {cfg.interval}")
    active = {tf: d.name for tf, d in active_runs(data_dir).items()}
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{cfg.interval}"
    d = paper_dir(data_dir) / "runs" / run_id
    st = Store(d)
    st.save_json("config.json", {**asdict(cfg), "run_id": run_id, "created_ms": int(time.time() * 1000)})
    active[cfg.interval] = run_id
    _write_atomic(paper_dir(data_dir) / "active.json", json.dumps(active))
    _write_atomic(paper_dir(data_dir) / "current", run_id)
    st.append("events", {"time": int(time.time() * 1000), "type": "new_run", "symbol": None,
                         "capital": cfg.capital, "risk_pct": cfg.risk_pct, "interval": cfg.interval})
    return d


def load_config(run_dir: Path) -> PaperConfig:
    c = json.loads((run_dir / "config.json").read_text())
    return PaperConfig(**{k: c[k] for k in PaperConfig.__dataclass_fields__ if k in c})


# ----------------------------------------------------------------------------
# โหมด live : worker วนทุก poll วินาที
# ----------------------------------------------------------------------------
class Binance:
    """ข้อมูลตลาดสาธารณะของ Binance เท่านั้น (ไม่มี API key ไม่มีคำสั่งซื้อขาย)"""

    def __init__(self):
        self.c = httpx.Client(base_url=dt.BINANCE, timeout=20)

    def prices(self, symbols: list[str]) -> dict[str, float]:
        r = self.c.get("/api/v3/ticker/price", params={"symbols": json.dumps(symbols, separators=(",", ":"))})
        r.raise_for_status()
        return {x["symbol"]: float(x["price"]) for x in r.json()}

    def minutes(self, symbol: str, start_ms: int, end_ms: int | None = None) -> list[tuple[int, float, float, float, float]]:
        """แท่ง 1 นาที (open_time, open, high, low, close) ตั้งแต่ start_ms ถึง end_ms (เวลาเปิด) — รวมแท่งที่กำลังวิ่ง"""
        out: list[tuple[int, float, float, float, float]] = []
        while True:
            params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": 1000}
            if end_ms:
                params["endTime"] = end_ms
            r = self.c.get("/api/v3/klines", params=params)
            r.raise_for_status()
            batch = r.json()
            out += [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in batch]
            if len(batch) < 1000:
                return out
            start_ms = int(batch[-1][0]) + 1


class LiveRunner:
    """เดินบัญชีเดโม 1 บัญชี (1 timeframe)"""

    def __init__(self, data_dir: Path = DATA_DIR, poll_s: float = 15.0, market: Binance | None = None,
                 run_dir: Path | None = None):
        self.data_dir = data_dir
        self.poll_s = poll_s
        self.mkt = market or Binance()
        self._cache: dict[str, tuple[int, dict]] = {}
        self.run_dir = run_dir or current_run(data_dir) or new_run(PaperConfig(), data_dir)
        self._load()

    def _load(self) -> None:
        d = self.run_dir
        self.store = Store(d)
        self.cfg = load_config(d)
        st = self.store.load_json("state.json") or {}
        tick = {s: dt.tick_size(s, self.data_dir) for s in self.cfg.symbols}
        self._buf: list[tuple[str, dict]] = []   # บันทึกที่ยังไม่ได้เขียน : เขียนพร้อม state ตอน commit
        self.engine = Engine(self.cfg, st.get("engine"), self._emit, tick)
        self.last_bar: dict[str, int] = st.get("last_bar", {})
        self.checked: dict[str, int] = st.get("checked", st.get("stop_checked", {}))
        self.last_snapshot: int = st.get("last_snapshot", 0)
        self.prices: dict[str, float] = st.get("prices", {})
        self.started_bar: int | None = st.get("started_bar")

    def _emit(self, kind: str, rec: dict) -> None:
        self._buf.append((kind, rec))

    def _save(self, now: int) -> None:
        """commit : เขียนบันทึกที่ค้าง แล้วตามด้วย state (ถ้า step ล้มกลางทาง จะย้อนกลับไปที่ commit ล่าสุด)"""
        for kind, rec in self._buf:
            self.store.append(kind, rec)
        self._buf.clear()
        self.store.save_json("state.json", {
            "engine": self.engine.state(), "last_bar": self.last_bar, "checked": self.checked,
            "last_snapshot": self.last_snapshot, "prices": self.prices, "heartbeat_ms": now,
            "started_bar": self.started_bar, "equity": self.engine.mark(self.prices or self.engine.last_close),
        })

    def _frames(self, now: int) -> dict[str, dict]:
        """สัญญาณของทุกแท่งที่ปิดแล้ว + แท่งที่กำลังวิ่ง — คำนวณใหม่เฉพาะเมื่อมีแท่งใหม่ (cache ตามเวลาแท่งล่าสุด)"""
        H = INTERVAL_SEC[self.cfg.interval] * 1000
        frames = {}
        for s in self.cfg.symbols:
            path = dt.candle_path(s, self.cfg.interval, self.data_dir)
            last = int(pl.read_parquet(path, columns=["open_time"])["open_time"].max().timestamp() * 1000) \
                if path.exists() else 0
            if now >= last + 2 * H:            # มีแท่งใหม่ที่ปิดแล้วรอโหลด
                dt.update(s, self.cfg.interval, self.data_dir)
                last = int(pl.read_parquet(path, columns=["open_time"])["open_time"].max().timestamp() * 1000)
            cached = self._cache.get(s)
            if cached is None or cached[0] != last:
                df = dt.load(s, self.cfg.interval, self.data_dir)
                cached = (last, signal_frame(df, self.cfg.interval, self.cfg, dt.tick_size(s, self.data_dir)))
                self._cache[s] = cached
            frames[s] = cached[1]
        return frames

    def _walk(self, sym: str, f: dict, i: int, upto: int) -> None:
        """ราคาเดินในแท่ง i ด้วยแท่ง 1 นาทีที่ปิดแล้ว ช่วงที่ยังไม่ได้ดู ถึง upto (ไม่รวม) : เข้าเมื่อแตะจุด / ออกเมื่อแตะ SL"""
        t = int(f["time"][i])
        start = max(self.checked.get(sym, t), t)
        if start >= upto:
            return
        mins = [m for m in self.mkt.minutes(sym, start, upto - 1) if start <= m[0] < upto]
        for mt, o, h, l, c in mins:
            self.engine.on_price(sym, mt, o, h, l, c)
        if not mins and start == t and upto >= t + INTERVAL_SEC[self.cfg.interval] * 1000:
            # ไม่มีข้อมูลรายนาทีของทั้งแท่ง -> ใช้ทั้งแท่งแบบ backtest
            self.engine.on_price(sym, t, float(f["open"][i]), float(f["high"][i]), float(f["low"][i]), float(f["close"][i]))
        self.checked[sym] = upto

    def step(self) -> None:
        now = int(time.time() * 1000)
        H = INTERVAL_SEC[self.cfg.interval] * 1000
        eng = self.engine
        self.prices = self.mkt.prices(self.cfg.symbols)
        frames = self._frames(now)

        # เริ่มรอบใหม่ : เริ่มนับจากแท่งที่ปิดล่าสุด (ไม่ย้อนเทรดอดีต) · แท่งที่วิ่งอยู่ตอนเริ่มไม่เข้าไม้ใหม่ (เห็นไม่ครบแท่ง)
        for s, f in frames.items():
            if s not in self.last_bar:
                n = f["n"]
                self.last_bar[s] = int(f["time"][n - 1])
                eng.last_close[s] = float(f["close"][n - 1])
                self.started_bar = self.started_bar or int(f["time"][n - 1])
                eng.begin_bar(int(f["time"][n]))
                eng.on_bar_open(s, bar_at(f, n), now, self.prices[s], None, enter=False)
                self.checked[s] = (now // MIN_MS + 1) * MIN_MS

        # แท่งที่ปิดใหม่ตั้งแต่รอบก่อน (ปกติ 1 แท่ง / มากกว่าถ้า worker เคยหยุด)
        todo = {s: {int(f["time"][i]): i for i in range(f["n"]) if f["time"][i] > self.last_bar[s]}
                for s, f in frames.items()}
        for t in sorted({t for m in todo.values() for t in m}):
            here = {s: m[t] for s, m in todo.items() if t in m}
            eng.begin_bar(t)
            for s, i in here.items():       # worker ไม่ได้เห็นตอนแท่งนี้เปิด -> ใช้ราคาเปิดแท่ง
                if eng.arm.get(s, {}).get("bar") != t:
                    o = float(frames[s]["open"][i])
                    eng.on_bar_open(s, bar_at(frames[s], i), t, o, o)
            for s, i in here.items():       # ราคาที่เหลือของแท่งนี้ ด้วยข้อมูล 1 นาที
                self._walk(s, frames[s], i, t + H)
            for s, i in here.items():
                eng.on_bar_close(s, bar_at(frames[s], i))
                self.last_bar[s] = t
            eng.eq_prev = eng.mark(eng.last_close)
            self._emit("equity", {"time": t + H, "equity": eng.eq_prev, "cash": eng.cash,
                                  "positions": sum(u.qty > 0 for u in eng.units), "source": "bar_close"})
            # เปิดแท่งถัดไป : แท่งที่ปิดแล้ว (ตามทัน) ใช้ราคาเปิด · แท่งที่กำลังวิ่ง ใช้ราคาจริงตอนนี้
            eng.begin_bar(t + H)
            for s, i in here.items():
                f = frames[s]
                if i + 1 < f["n"]:
                    o = float(f["open"][i + 1])
                    eng.on_bar_open(s, bar_at(f, i + 1), t + H, o, o)
                else:
                    mins = self.mkt.minutes(s, t + H, t + H + MIN_MS - 1)
                    eng.on_bar_open(s, bar_at(f, i + 1), now, self.prices[s], mins[0][1] if mins else None)
                self.checked[s] = t + H
            self._save(now)                 # ตามทันทีละแท่ง : เน็ตหลุดกลางทางไม่ต้องเริ่มใหม่ทั้งหมด

        # แท่งที่กำลังวิ่ง : นาทีที่ปิดแล้ว
        upto = (now // MIN_MS) * MIN_MS
        for s, f in frames.items():
            self._walk(s, f, f["n"], upto)

        if now - self.last_snapshot >= 15 * MIN_MS:
            self._emit("equity", {"time": now, "equity": eng.mark(self.prices), "cash": eng.cash,
                                  "positions": sum(u.qty > 0 for u in eng.units), "source": "live"})
            self.last_snapshot = now
        self._save(now)

    def safe_step(self) -> None:
        try:
            self.step()
        except Exception as e:  # เน็ตหลุด/Binance ล่ม -> ย้อนกลับไปที่ commit ล่าสุด บันทึก แล้วลองใหม่รอบหน้า
            self._load()
            self.store.append("events", {"time": int(time.time() * 1000), "type": "error", "symbol": None,
                                         "message": f"{type(e).__name__}: {e}"})
            traceback.print_exc()


class MultiRunner:
    """worker ตัวเดียวเดินทุกบัญชีที่ active (timeframe ละ 1 บัญชี) — เริ่ม/เปลี่ยนบัญชีจากเว็บได้โดยไม่ต้องรีสตาร์ท"""

    def __init__(self, data_dir: Path = DATA_DIR, poll_s: float = 15.0, market: Binance | None = None):
        self.data_dir = data_dir
        self.poll_s = poll_s
        self.mkt = market or Binance()
        self.runners: dict[str, LiveRunner] = {}

    def sync(self) -> None:
        active = active_runs(self.data_dir)
        if not active:
            new_run(PaperConfig(), self.data_dir)
            active = active_runs(self.data_dir)
        want = {d.name: d for d in active.values()}
        for rid in list(self.runners):
            if rid not in want:
                del self.runners[rid]
        for rid, d in want.items():
            if rid not in self.runners:
                self.runners[rid] = LiveRunner(self.data_dir, self.poll_s, self.mkt, run_dir=d)
                print(f"paper worker: เดินบัญชี {rid}", flush=True)

    def run_forever(self) -> None:
        print(f"paper worker: เช็คทุก {self.poll_s:.0f} วินาที", flush=True)
        while True:
            try:
                self.sync()
            except Exception:
                traceback.print_exc()
            for r in list(self.runners.values()):
                r.safe_step()
            time.sleep(self.poll_s)


# ----------------------------------------------------------------------------
# เทียบเดโมกับ backtest ช่วงเดียวกัน
# ----------------------------------------------------------------------------
def compare(run_dir: Path, data_dir: Path = DATA_DIR) -> dict:
    from .portfolio import build_sleeve, simulate_portfolio

    store = Store(run_dir)
    cfg = load_config(run_dir)
    st = store.load_json("state.json") or {}
    H = INTERVAL_SEC[cfg.interval] * 1000
    started = st.get("started_bar")
    if not started or not st.get("last_bar"):
        return {"ready": False, "message": "worker ยังไม่ได้ปิดแท่งแรก — รอให้ครบ 1 แท่งก่อน"}
    t0 = started + 2 * H     # แท่งแรกที่เดโมเห็นครบทั้งแท่ง
    t1 = max(st["last_bar"].values()) + H
    if t1 <= t0:
        return {"ready": False, "message": "ยังไม่มีแท่งที่ปิดหลังเริ่มเดโม"}

    sleeves = [build_sleeve("trend", s, cfg.interval, t0, t1, data_dir, overrides=cfg.strategy) for s in cfg.symbols]
    res = simulate_portfolio(sleeves, t0, t1, cfg.risk_pct, cfg.max_lev, cfg.commission_pct, cfg.capital)

    bt = []
    for sl in sleeves:
        last_ms = int(sl.t[np.searchsorted(sl.t, t1) - 1])   # แท่งสุดท้ายของช่วง : backtest บังคับปิดที่นี่
        for tr, r in zip(sl.trades, sl.r):
            if (sl.symbol, tr["entry_ms"]) in res["skipped_keys"]:   # พอร์ตเงินไม่พอ ไม่ได้เข้า
                continue
            ex_ms, ex_px, _ = tr["exits"][-1]
            still_open = ex_ms == last_ms
            bt.append({"symbol": sl.symbol, "entry_bar": tr["entry_ms"], "entry_price": tr["entry_px"],
                       "exit_time": None if still_open else ex_ms, "exit_price": None if still_open else ex_px,
                       "r": None if still_open else float(r)})
    # ไม้ที่เข้าในแท่งที่ยังไม่ปิด backtest ยังไม่เห็น -> ยังไม่เทียบ
    paper = [x for x in store.read("trades") if x["entry_bar"] < t1]
    open_pos = [p for p in open_positions(st) if p["entry_bar"] < t1]
    paper_rows = paper + [{"symbol": p["symbol"], "entry_bar": p["entry_bar"], "entry_price": p["entry_price"],
                           "ref_entry_price": p["ref_price"], "exit_time": None, "exit_price": None, "r": None,
                           "reason": "ยังถืออยู่"} for p in open_pos]

    key = lambda x: (x["symbol"], int(x["entry_bar"]))  # noqa: E731
    bt_by = {key(x): x for x in bt}
    pp_by = {key(x): x for x in paper_rows}
    rows = []
    for k in sorted(set(bt_by) | set(pp_by), key=lambda k: k[1]):
        a, b = pp_by.get(k), bt_by.get(k)
        rows.append({
            "symbol": k[0], "entry_bar": k[1],
            "status": "ตรงกัน" if a and b else "มีแต่ในเดโม" if a else "มีแต่ใน backtest",
            "paper_entry": a and a["entry_price"], "bt_entry": b and b["entry_price"],
            "entry_diff_pct": (a["entry_price"] / b["entry_price"] - 1) * 100 if a and b else None,
            "paper_exit": a and a.get("exit_price"), "bt_exit": b and b.get("exit_price"),
            "paper_exit_time": a and a.get("exit_time"), "bt_exit_time": b and b.get("exit_time"),
            "paper_r": a and a.get("r"), "bt_r": b and b.get("r"), "paper_reason": a and a.get("reason"),
        })

    eq = store.read("equity")
    equity_paper = [{"time": e["time"], "equity": e["equity"]} for e in eq]
    bt_eq = [{"time": int(t) + H, "equity": float(v)} for t, v in zip(res["time"], res["equity"])]
    paper_end = st.get("equity", cfg.capital)
    matched = [r for r in rows if r["status"] == "ตรงกัน"]
    diffs = [r["entry_diff_pct"] for r in matched if r["entry_diff_pct"] is not None]
    return {
        "ready": True, "from": t0, "to": t1, "capital": cfg.capital,
        "summary": {
            "paper_return_pct": (paper_end / cfg.capital - 1) * 100,
            "bt_return_pct": (float(res["equity"][-1]) / cfg.capital - 1) * 100 if len(res["equity"]) else 0.0,
            "trades_paper": len(paper_rows), "trades_bt": len(bt), "matched": len(matched),
            "avg_entry_diff_pct": float(np.mean(diffs)) if diffs else None,
        },
        "trades": rows, "equity_paper": equity_paper, "equity_bt": bt_eq,
    }


def open_positions(st: dict) -> list[dict]:
    """ไม้ที่ถืออยู่จาก state.json (รองรับรุ่นก่อนที่เก็บเป็น dict เหรียญละไม้)"""
    pos = (st.get("engine") or {}).get("positions", [])
    return list(pos.values()) if isinstance(pos, dict) else pos


def iso(ms: int | None) -> str:
    return "-" if not ms else datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
