"""บัญชีเดโม (paper trading) — เงินจำลอง คำสั่งจำลอง ราคาจริงจาก Binance (ตลาดสาธารณะ อ่านอย่างเดียว ไม่มีบัญชี)

เป้าหมาย : เก็บข้อมูลการทำงานจริงไว้เทียบกับ backtest ช่วงเวลาเดียวกัน
  - กฎเหมือน backtest ทุกข้อ (Engine ตัวเดียวกันใช้ทั้ง replay และ live) : ระบบ trend ของพอร์ต (portfolio.TREND)
  - ต่างกันแค่ "ราคาที่ได้" :
      เข้า  : backtest สมมติราคาเปิดแท่งถัดไป · เดโมใช้ราคาจริง ณ ตอนที่ worker เห็นว่าแท่งปิด (+ slippage)
      SL   : backtest ดู high/low ของแท่ง 4h · เดโมดูทีละนาที (1m) ระหว่างแท่งกำลังวิ่ง
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
from .core import CoreParams, base_features, compose, entry_raw
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
class Pos:
    symbol: str
    side: int
    qty: float
    entry_price: float           # ราคาที่ได้จริง (รวม slippage)
    ref_price: float | None      # ราคาที่ backtest สมมติ (เปิดแท่งถัดไป)
    entry_time: int              # ms เวลาที่เข้า
    entry_bar: int               # ms เวลาเปิดของแท่งที่เข้า (= แท่งถัดจากแท่งสัญญาณ)
    risk: float                  # ระยะ SL เริ่มต้น (ราคา)
    init_stop: float
    stop: float
    ext: float | None = None     # high สูงสุดหลังเข้า
    trail: float | None = None


@dataclass
class Bar:
    time: int
    open: float
    high: float
    low: float
    close: float
    atr: float
    long: bool       # สัญญาณเข้า (รวมทิศ + ตัวกรองแล้ว)
    up: bool
    down: bool


class Engine:
    """สถานะบัญชี + กฎของระบบ — ไม่รู้จักเวลาจริงหรือเครือข่าย (ทดสอบ/replay ได้ตรง ๆ)"""

    def __init__(self, cfg: PaperConfig, state: dict | None = None, sink=None, tick: dict | None = None):
        self.cfg = cfg
        self.sp = CoreParams().with_overrides(**cfg.strategy)
        st = state or {}
        self.cash: float = st.get("cash", cfg.capital)
        self.pos: dict[str, Pos] = {k: Pos(**v) for k, v in st.get("positions", {}).items()}
        self.pending: dict[str, dict] = st.get("pending", {})
        self.last_close: dict[str, float] = st.get("last_close", {})
        self.eq_prev: float = st.get("eq_prev", cfg.capital)
        self.tick = tick or st.get("tick", {})
        self.sink = sink or (lambda kind, rec: None)

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        return {"cash": self.cash, "positions": {k: asdict(v) for k, v in self.pos.items()},
                "pending": self.pending, "last_close": self.last_close, "eq_prev": self.eq_prev, "tick": self.tick}

    def _slip(self, sym: str) -> float:
        return self.cfg.slippage_ticks * self.tick.get(sym, 0.01)

    def _comm(self) -> float:
        return self.cfg.commission_pct / 100.0

    def _event(self, t_ms: int, kind: str, sym: str | None = None, **kw):
        self.sink("events", {"time": t_ms, "type": kind, "symbol": sym, **kw})

    # ---------------------------------------------------------------- actions
    def execute_pending(self, sym: str, t_ms: int, price: float, ref_price: float | None, bar_ms: int) -> None:
        """คำสั่ง market ที่ค้างจากแท่งปิด -> ได้ราคา price (backtest = ราคาเปิดแท่งถัดไป)"""
        p = self.pending.pop(sym, None)
        if not p:
            return
        if p["action"] == "exit" and sym in self.pos:
            self._close(sym, t_ms, price - self.pos[sym].side * self._slip(sym), p["reason"], ref_price)
        elif p["action"] == "enter" and sym not in self.pos:
            fill = price + self._slip(sym)
            E = self.eq_prev
            qty = self.cfg.risk_pct / 100.0 * E / p["risk"]
            held = sum(q.qty * self.last_close.get(q.symbol, q.entry_price) for q in self.pos.values())
            room = self.cfg.max_lev * E - held
            capped = qty * fill > room
            if capped:
                qty = max(room, 0.0) / fill
            if qty <= 0:
                self._event(t_ms, "skip", sym, reason="เงินไม่พอ (ถือเต็มเพดานแล้ว)")
                return
            self.cash -= fill * qty * self._comm()
            stop = fill - p["risk"]
            self.pos[sym] = Pos(sym, 1, qty, fill, ref_price, t_ms, bar_ms, p["risk"], stop, stop)
            self._event(t_ms, "entry", sym, price=fill, ref_price=ref_price, qty=qty, stop=stop,
                        risk_money=qty * p["risk"], capped=capped, signal_bar=p.get("signal_bar"))

    def check_stop(self, sym: str, t_ms: int, o: float, h: float, l: float, min_time: int | None = None) -> bool:
        """ราคาแตะ SL ระหว่างแท่ง (ข้อมูล 1 นาทีในโหมด live / แท่ง 4h ในโหมด replay)"""
        p = self.pos.get(sym)
        if p is None or t_ms < (p.entry_bar if min_time is None else min_time):
            return False
        if o <= p.stop:
            px = o - self._slip(sym)          # เปิด gap ทะลุ SL
        elif l <= p.stop:
            px = p.stop - self._slip(sym)
        else:
            return False
        self._close(sym, t_ms, px, "stop" if p.stop == p.init_stop else "trailing", p.stop)
        return True

    def on_bar_close(self, sym: str, b: Bar) -> None:
        """ตัดสินใจตอนแท่งปิด : สัญญาณเข้า / เลื่อน trailing stop / ปิดเมื่อทิศกลับ"""
        if sym not in self.pos and b.long and not math.isnan(b.atr):
            self.pending[sym] = {"action": "enter", "risk": b.atr * self.sp.sl_atr, "signal_bar": b.time}
            self._event(b.time, "signal", sym, close=b.close, breakout=True, risk=b.atr * self.sp.sl_atr)
        p = self.pos.get(sym)
        if p is not None:
            if p.entry_bar == b.time:          # แท่งแรกที่ถือ
                p.stop = p.entry_price - p.risk
                p.init_stop = p.stop
                p.ext = b.high
                p.trail = None
            p.ext = max(p.ext if p.ext is not None else b.high, b.high)
            ch = p.ext - b.atr * self.sp.trail_atr
            if not math.isnan(ch):
                p.trail = ch if p.trail is None else max(p.trail, ch)
            new_stop = max(p.stop, p.trail) if p.trail is not None else p.stop
            if new_stop > p.stop:
                self._event(b.time, "stop_moved", sym, old=p.stop, new=new_stop)
            p.stop = new_stop
            if self.sp.exit_on_regime and b.down:
                self.pending[sym] = {"action": "exit", "reason": "trend_flip"}
                self._event(b.time, "signal_exit", sym, reason="trend_flip")
        self.last_close[sym] = b.close

    def mark(self, prices: dict[str, float]) -> float:
        unreal = sum(p.qty * (prices.get(s, p.entry_price) - p.entry_price) * p.side for s, p in self.pos.items())
        return self.cash + unreal

    def _close(self, sym: str, t_ms: int, px: float, reason: str, ref: float | None = None) -> None:
        p = self.pos.pop(sym)
        self.pending.pop(sym, None)
        comm = self._comm()
        gross = (px - p.entry_price) * p.side * p.qty
        exit_comm = px * p.qty * comm
        pnl = gross - exit_comm - p.entry_price * p.qty * comm
        self.cash += gross - exit_comm
        rec = {
            "symbol": sym, "side": "long" if p.side == 1 else "short", "qty": p.qty,
            "entry_time": p.entry_time, "entry_bar": p.entry_bar, "entry_price": p.entry_price,
            "ref_entry_price": p.ref_price, "exit_time": t_ms, "exit_price": px, "ref_exit_price": ref,
            "reason": reason, "risk": p.risk, "pnl": pnl, "r": pnl / (p.risk * p.qty),
        }
        self.sink("trades", rec)
        self._event(t_ms, "exit", sym, price=px, reason=reason, pnl=pnl, r=rec["r"])


# ----------------------------------------------------------------------------
# สัญญาณจากข้อมูลแท่งเทียน (ใช้ตัวเดียวกับ backtest)
# ----------------------------------------------------------------------------
def signal_frame(df: pl.DataFrame, interval: str, cfg: PaperConfig, tick: float) -> dict[str, np.ndarray]:
    p = CoreParams().with_overrides(tick_size=tick, **cfg.strategy)
    b = base_features(df, interval, p)
    sg = compose(b, entry_raw(b, p), p)
    return {"time": b["time_ms"], "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "atr": b["atr"], "long": sg["long"], "up": sg["up"], "down": sg["down"]}


def bar_at(f: dict, i: int) -> Bar:
    return Bar(int(f["time"][i]), float(f["open"][i]), float(f["high"][i]), float(f["low"][i]),
               float(f["close"][i]), float(f["atr"][i]), bool(f["long"][i]), bool(f["up"][i]), bool(f["down"][i]))


def replay(engine: Engine, frames: dict[str, dict], t0: int, t1: int, on_mark=None) -> None:
    """เล่นย้อนหลังด้วยแท่ง 4h ตามลำดับเดียวกับ backtest พอร์ต :
    ทุกเหรียญ -> (1) คำสั่งค้างได้ราคาเปิด (2) เช็ค SL ด้วย high/low (3) ตัดสินใจตอนปิด -> mark ราคาปิด"""
    idx = {s: {int(t): i for i, t in enumerate(f["time"]) if t0 <= t < t1} for s, f in frames.items()}
    timeline = sorted({t for m in idx.values() for t in m})
    for t in timeline:
        here = {s: m[t] for s, m in idx.items() if t in m}
        for s, i in here.items():
            o = float(frames[s]["open"][i])
            engine.execute_pending(s, t, o, o, t)
        for s, i in here.items():
            f = frames[s]
            engine.check_stop(s, t, float(f["open"][i]), float(f["high"][i]), float(f["low"][i]))
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

    def minutes(self, symbol: str, start_ms: int, end_ms: int | None = None) -> list[tuple[int, float, float, float]]:
        """แท่ง 1 นาที (open_time, open, high, low) ตั้งแต่ start_ms — รวมแท่งที่กำลังวิ่ง"""
        out: list[tuple[int, float, float, float]] = []
        while True:
            params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": 1000}
            if end_ms:
                params["endTime"] = end_ms
            r = self.c.get("/api/v3/klines", params=params)
            r.raise_for_status()
            batch = r.json()
            out += [(int(k[0]), float(k[1]), float(k[2]), float(k[3])) for k in batch]
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
        self.engine = Engine(self.cfg, st.get("engine"), self.store.append, tick)
        self.last_bar: dict[str, int] = st.get("last_bar", {})
        self.stop_checked: dict[str, int] = st.get("stop_checked", {})
        self.last_snapshot: int = st.get("last_snapshot", 0)
        self.prices: dict[str, float] = st.get("prices", {})
        self.started_bar: int | None = st.get("started_bar")

    def _save(self, now: int) -> None:
        self.store.save_json("state.json", {
            "engine": self.engine.state(), "last_bar": self.last_bar, "stop_checked": self.stop_checked,
            "last_snapshot": self.last_snapshot, "prices": self.prices, "heartbeat_ms": now,
            "started_bar": self.started_bar, "equity": self.engine.mark(self.prices or self.engine.last_close),
        })

    def _frames(self, now: int) -> dict[str, dict]:
        """สัญญาณของทุกแท่งที่ปิดแล้ว — คำนวณใหม่เฉพาะเมื่อมีแท่งใหม่ (cache ตามเวลาแท่งล่าสุด)"""
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

    def _check_stops(self, sym: str, upto_ms: int) -> None:
        """เช็ค SL ด้วยแท่ง 1 นาที ช่วงที่ยังไม่ได้เช็ค (ข้ามนาทีที่เพิ่งเข้าไม้)"""
        p = self.engine.pos.get(sym)
        if p is None:
            return
        min_time = (p.entry_time // MIN_MS + 1) * MIN_MS
        start = max(self.stop_checked.get(sym, min_time), min_time)
        if start >= upto_ms:
            return
        for t, o, h, l in self.mkt.minutes(sym, start, upto_ms - 1):
            if self.engine.check_stop(sym, t, o, h, l, min_time=min_time):
                break
        self.stop_checked[sym] = max(start, (upto_ms // MIN_MS) * MIN_MS)

    def step(self) -> None:
        now = int(time.time() * 1000)
        H = INTERVAL_SEC[self.cfg.interval] * 1000
        self.prices = self.mkt.prices(self.cfg.symbols)
        frames = self._frames(now)

        # เริ่มรอบใหม่ : เริ่มนับจากแท่งที่ปิดล่าสุด (ไม่ย้อนเทรดอดีต)
        for s, f in frames.items():
            if s not in self.last_bar:
                self.last_bar[s] = int(f["time"][-1])
                self.engine.last_close[s] = float(f["close"][-1])
                self.started_bar = self.started_bar or int(f["time"][-1])

        # แท่งที่ปิดใหม่ตั้งแต่รอบก่อน (ปกติ 1 แท่ง / มากกว่าถ้า worker เคยหยุด)
        todo = {s: {int(f["time"][i]): int(i) for i in np.nonzero(f["time"] > self.last_bar[s])[0]}
                for s, f in frames.items()}
        for t in sorted({t for m in todo.values() for t in m}):
            here = {s: m[t] for s, m in todo.items() if t in m}
            for s, i in here.items():       # คำสั่งค้าง (กรณี worker หยุดไปนาน) ได้ราคาเปิดแท่งนี้
                if s in self.engine.pending:
                    o = float(frames[s]["open"][i])
                    self.engine.execute_pending(s, t, o, o, t)
            for s in here:                  # SL ระหว่างแท่งนี้ ด้วยข้อมูล 1 นาที
                self._check_stops(s, t + H)
            for s, i in here.items():
                self.engine.on_bar_close(s, bar_at(frames[s], i))
                self.last_bar[s] = t
            self.engine.eq_prev = self.engine.mark(self.engine.last_close)
            self.store.append("equity", {"time": t + H, "equity": self.engine.eq_prev, "cash": self.engine.cash,
                                         "positions": len(self.engine.pos), "source": "bar_close"})

        # คำสั่งจากแท่งที่เพิ่งปิด -> เข้าตอนนี้ที่ราคาจริง (ref = ราคาเปิดแท่งที่กำลังวิ่ง ตามที่ backtest สมมติ)
        for s in list(self.engine.pending):
            bar_ms = self.last_bar[s] + H
            mins = self.mkt.minutes(s, bar_ms, bar_ms + MIN_MS - 1)
            ref = mins[0][1] if mins else None
            self.engine.execute_pending(s, now, self.prices[s], ref, bar_ms)
            if s in self.engine.pos:
                self.stop_checked[s] = (now // MIN_MS + 1) * MIN_MS

        # SL ของแท่งที่กำลังวิ่ง
        for s in self.cfg.symbols:
            self._check_stops(s, now)

        if now - self.last_snapshot >= 15 * MIN_MS:
            self.store.append("equity", {"time": now, "equity": self.engine.mark(self.prices), "cash": self.engine.cash,
                                         "positions": len(self.engine.pos), "source": "live"})
            self.last_snapshot = now
        self._save(now)

    def safe_step(self) -> None:
        try:
            self.step()
        except Exception as e:  # เน็ตหลุด/Binance ล่ม -> บันทึกแล้วลองใหม่รอบหน้า
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
    t0 = started + H
    t1 = max(st["last_bar"].values()) + H
    if t1 <= t0:
        return {"ready": False, "message": "ยังไม่มีแท่งที่ปิดหลังเริ่มเดโม"}

    sleeves = [build_sleeve("trend", s, cfg.interval, t0, t1, data_dir, overrides=cfg.strategy) for s in cfg.symbols]
    res = simulate_portfolio(sleeves, t0, t1, cfg.risk_pct, cfg.max_lev, cfg.commission_pct, cfg.capital)

    bt = []
    for sl in sleeves:
        last_ms = int(sl.t[np.searchsorted(sl.t, t1) - 1])   # แท่งสุดท้ายของช่วง : backtest บังคับปิดที่นี่
        for tr, r in zip(sl.trades, sl.r):
            ex_ms, ex_px, _ = tr["exits"][-1]
            still_open = ex_ms == last_ms
            bt.append({"symbol": sl.symbol, "entry_bar": tr["entry_ms"], "entry_price": tr["entry_px"],
                       "exit_time": None if still_open else ex_ms, "exit_price": None if still_open else ex_px,
                       "r": None if still_open else float(r)})
    paper = store.read("trades")
    open_pos = (st.get("engine") or {}).get("positions", {})
    paper_rows = paper + [{"symbol": p["symbol"], "entry_bar": p["entry_bar"], "entry_price": p["entry_price"],
                           "ref_entry_price": p["ref_price"], "exit_time": None, "exit_price": None, "r": None,
                           "reason": "ยังถืออยู่"} for p in open_pos.values()]

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


def iso(ms: int | None) -> str:
    return "-" if not ms else datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
