"""REST API สำหรับหน้าเว็บ (Next.js) — อ่านไฟล์ Parquet ตรง ๆ ด้วย Polars ไม่มีฐานข้อมูล"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import backtest as bt
from . import data as dt
from . import signals as sg
from .config import DATA_DIR, INTERVAL_SEC, Params
from .metrics import summarize

app = FastAPI(title="gold-bitcoin engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
RESULTS = DATA_DIR / "results"


def _check(symbol: str, interval: str) -> str:
    if interval not in INTERVAL_SEC:
        raise HTTPException(400, f"interval ไม่รองรับ: {interval}")
    if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol.upper()):
        raise HTTPException(400, "symbol ไม่ถูกต้อง")
    return symbol.upper()


def _load(symbol: str, interval: str) -> pl.DataFrame:
    try:
        return dt.load(symbol, interval)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


def _clean(obj):
    """JSON ไม่รองรับ NaN / inf -> แปลงเป็น null · ค่า numpy -> ค่า Python"""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/symbols")
def symbols():
    out = []
    for d in sorted((DATA_DIR / "candles").glob("*")):
        if d.is_dir():
            out.append({"symbol": d.name, "intervals": sorted(f.stem.replace("1mo", "1M") for f in d.glob("*.parquet"))})
    return out


@app.get("/api/strategy/{symbol}/{interval}")
def strategy(symbol: str, interval: str, bars: int = Query(500, ge=50, le=5000), refresh: bool = True):
    """ระบบ trend ที่ใช้ในพอร์ต: สถานะตอนนี้ + รายการไม้ + เส้น SL / breakout / ทิศ ย้อนหลัง bars แท่ง"""
    from .live import ensure_fresh, strategy_state
    symbol = _check(symbol, interval)
    if refresh:
        try:
            ensure_fresh(symbol, interval)
        except Exception:   # Binance ล่ม/เน็ตหลุด -> ใช้ข้อมูลเดิม
            pass
    try:
        return _clean(strategy_state(symbol, interval, bars))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


# ---------------------------------------------------------------- บัญชีเดโม
RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}(-[0-9a-zA-Z]+)?$")


def _paper_run(run: str | None = None):
    """run = ชื่อบัญชี (ถ้าไม่ระบุ = บัญชีล่าสุดที่สร้าง)"""
    from . import paper as pp
    if run:
        if not RUN_RE.fullmatch(run):
            raise HTTPException(400, "run ไม่ถูกต้อง")
        d = pp.paper_dir() / "runs" / run
        if not (d / "config.json").exists():
            raise HTTPException(404, "ไม่พบบัญชีเดโมนี้")
        return pp, d
    d = pp.current_run()
    if d is None:
        raise HTTPException(404, "ยังไม่มีบัญชีเดโม — เปิด worker: docker compose up -d paper")
    return pp, d


def _run_summary(pp, d) -> dict:
    st = pp.Store(d).load_json("state.json") or {}
    cfg = json.loads((d / "config.json").read_text())
    trades = pp.Store(d).read("trades")
    eq = st.get("equity", cfg["capital"])
    return {
        "run": d.name, "interval": cfg.get("interval", "4h"), "capital": cfg["capital"], "risk_pct": cfg["risk_pct"],
        "created_ms": cfg.get("created_ms"), "equity": eq, "return_pct": (eq / cfg["capital"] - 1) * 100,
        "trades": len(trades), "wins": sum(1 for t in trades if t["pnl"] > 0),
        "total_r": float(sum(t["r"] for t in trades)),
        "open_positions": len((st.get("engine") or {}).get("positions", {})),
        "heartbeat_ms": st.get("heartbeat_ms"),
    }


@app.get("/api/paper/status")
def paper_status(run: str | None = None):
    pp, d = _paper_run(run)
    st = pp.Store(d).load_json("state.json") or {}
    cfg = pp.load_config(d)
    eng = st.get("engine") or {}
    prices = st.get("prices", {})
    positions = []
    for p in eng.get("positions", {}).values():
        px = prices.get(p["symbol"])
        positions.append({**p, "price": px,
                          "pnl": (px - p["entry_price"]) * p["qty"] * p["side"] if px else None,
                          "r_now": (px - p["entry_price"]) * p["side"] / p["risk"] if px else None})
    trades = pp.Store(d).read("trades")
    wins = [t for t in trades if t["pnl"] > 0]
    active = {x.name for x in pp.active_runs().values()}
    return _clean({
        "run": d.name, "active": d.name in active, "config": json.loads((d / "config.json").read_text()),
        "equity": st.get("equity", cfg.capital), "cash": eng.get("cash", cfg.capital),
        "heartbeat_ms": st.get("heartbeat_ms"), "started_bar": st.get("started_bar"),
        "last_bar": st.get("last_bar", {}), "prices": prices, "positions": positions,
        "pending": eng.get("pending", {}),
        "closed": {"trades": len(trades), "wins": len(wins), "total_r": float(sum(t["r"] for t in trades)),
                   "pnl": float(sum(t["pnl"] for t in trades))},
    })


@app.get("/api/paper/trades")
def paper_trades(run: str | None = None):
    pp, d = _paper_run(run)
    return _clean(pp.Store(d).read("trades"))


@app.get("/api/paper/events")
def paper_events(run: str | None = None, limit: int = Query(200, ge=1, le=5000)):
    pp, d = _paper_run(run)
    return _clean(list(reversed(pp.Store(d).read("events", limit))))


@app.get("/api/paper/equity")
def paper_equity(run: str | None = None):
    pp, d = _paper_run(run)
    return _clean(pp.Store(d).read("equity"))


@app.get("/api/paper/compare")
def paper_compare(run: str | None = None):
    pp, d = _paper_run(run)
    return _clean(pp.compare(d))


class NewRunReq(BaseModel):
    capital: float = Field(10_000, gt=0, le=1e9)
    risk_pct: float = Field(1.0, gt=0, le=5)
    interval: str = Field("4h", pattern="^(15m|1h|4h|1d)$")


@app.post("/api/paper/new")
def paper_new(req: NewRunReq):
    """เริ่มบัญชีเดโมใหม่ของ timeframe นั้น — บัญชีเดิมของ TF เดียวกันเก็บไว้ (ไม่ลบ) TF อื่นเดินต่อ"""
    from . import paper as pp
    d = pp.new_run(pp.PaperConfig(capital=req.capital, risk_pct=req.risk_pct, interval=req.interval))
    return {"run": d.name}


@app.get("/api/paper/runs")
def paper_runs():
    from . import paper as pp
    root = pp.paper_dir() / "runs"
    active = {d.name for d in pp.active_runs().values()}
    out = []
    for d in sorted(root.glob("*"), reverse=True) if root.exists() else []:
        if (d / "config.json").exists():
            out.append({**_run_summary(pp, d), "active": d.name in active})
    return _clean(out)


@app.get("/api/paper/summary")
def paper_summary():
    """สรุปบัญชีเดโมที่เดินอยู่ แยกตาม timeframe + backtest ช่วงเวลาเดียวกัน"""
    from . import paper as pp
    out = []
    for tf, d in sorted(pp.active_runs().items(), key=lambda x: INTERVAL_SEC[x[0]]):
        row = _run_summary(pp, d)
        try:
            c = pp.compare(d)
            row["compare"] = c["summary"] if c.get("ready") else None
        except Exception as e:   # ข้อมูลยังไม่พอ ฯลฯ
            row["compare"] = None
            row["compare_error"] = str(e)
        out.append(row)
    return _clean(out)


def _latest(kind: str):
    dirs = sorted(d for d in RESULTS.glob(f"{kind}-*") if d.is_dir())
    return dirs[-1] if dirs else None


@app.get("/api/research/{kind}")
def research(kind: str):
    """ผลทดสอบล่าสุดแต่ละแบบ (ไฟล์ CSV ใน data/results)"""
    files = {"rules": "rules.csv", "walkforward": "summary.csv", "walkforward_windows": "windows.csv",
             "range": "range.csv", "portfolio": "portfolio.csv", "ablation": "ablation.csv",
             "timeframes": "timeframes.csv"}
    if kind not in files:
        raise HTTPException(404, f"kind ต้องเป็นหนึ่งใน {sorted(files)}")
    d = _latest(kind.replace("_windows", ""))
    if d is None or not (d / files[kind]).exists():
        raise HTTPException(404, "ยังไม่มีผล — รันคำสั่งทดสอบก่อน")
    return {"run": d.name, "rows": _clean(pl.read_csv(d / files[kind], infer_schema_length=None).to_dicts())}


@app.get("/api/portfolio/latest")
def portfolio_latest(every_days: int = Query(1, ge=1, le=30)):
    """equity ของพอร์ตล่าสุด (ตั้งต้น = 100) สุ่มเก็บเป็นรายวัน + ตาราง metric"""
    d = _latest("portfolio")
    if d is None:
        raise HTTPException(404, "ยังไม่มีผลพอร์ต — รัน: portfolio")
    eq = pl.read_parquet(d / "equity.parquet")
    day = pl.col("time") // (86_400_000 * every_days)
    curves = (eq.with_columns(day.alias("d")).group_by("portfolio", "d", maintain_order=True)
                .agg(pl.col("time").last(), pl.col("equity").last())
                .with_columns((pl.col("equity") / pl.col("equity").first().over("portfolio") * 100).alias("index")))
    series = {name: {"time": (g["time"] // 1000).to_list(), "value": g["index"].round(3).to_list()}
              for (name,), g in curves.group_by("portfolio", maintain_order=True)}
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    metrics = pl.read_csv(d / "portfolio.csv").to_dicts()
    return _clean({"run": d.name, "config": config, "series": series, "metrics": metrics})


@app.get("/api/candles/{symbol}/{interval}")
def candles(symbol: str, interval: str, limit: int = Query(1500, ge=1, le=50_000)):
    """รูปแบบเดียวกับ Lightweight Charts: time เป็นวินาที UTC"""
    symbol = _check(symbol, interval)
    df = _load(symbol, interval).tail(limit)
    return df.select(
        (pl.col("open_time").dt.epoch("s")).alias("time"), "open", "high", "low", "close", "volume"
    ).to_dicts()


@app.get("/api/analysis/{symbol}/{interval}")
def analysis(symbol: str, interval: str, bars: int = Query(300, ge=1, le=5000)):
    """สถานะล่าสุดของทุกตัววิเคราะห์ + ซีรีส์คะแนนย้อนหลัง"""
    symbol = _check(symbol, interval)
    df = _load(symbol, interval)
    f = sg.compute(df, interval, Params())
    last = {k: _jsonable(v[-1]) for k, v in f.items()}
    t = (f["time_ms"][-bars:] // 1000).tolist()
    series = {k: [_jsonable(x) for x in f[k][-bars:]] for k in ("long_score", "short_score", "struct_dir", "ew_bias")}
    return {"symbol": symbol, "interval": interval, "last": last, "time": t, "series": series}


class BacktestReq(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "4h"
    params: dict = Field(default_factory=dict)
    split: str | None = "2023-01-01"
    save: bool = True


@app.post("/api/backtest")
def run_backtest(req: BacktestReq):
    symbol = _check(req.symbol, req.interval)
    try:
        p = Params().with_overrides(**req.params)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e)) from e
    try:
        res = bt.load_and_run(symbol, req.interval, p)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    m = summarize(res, req.split)
    run_id = bt.save(res, m) if req.save else None
    return _clean({"run_id": run_id, "metrics": m.to_dicts()})


@app.get("/api/runs")
def runs():
    files = sorted(RESULTS.glob("*/metrics.parquet"))
    if not files:
        return []
    cols = ["run_id", "created", "symbol", "interval", "period", "trades", "win_rate", "profit_factor",
            "total_r", "net_pct", "max_dd_pct", "buy_hold_pct"]
    df = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")
    return _clean(df.filter(pl.col("period") == "all").select(cols)
                  .sort("created", descending=True).collect().to_dicts())


def _run_dir(run_id: str):
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(400, "run_id ไม่ถูกต้อง")
    d = RESULTS / run_id
    if not d.exists():
        raise HTTPException(404, "ไม่พบ run")
    return d


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    m = pl.read_parquet(_run_dir(run_id) / "metrics.parquet")
    params = json.loads(m["params"][0])
    return _clean({"run_id": run_id, "params": params, "metrics": m.drop("params").to_dicts()})


@app.get("/api/runs/{run_id}/trades")
def run_trades(run_id: str):
    tr = pl.read_parquet(_run_dir(run_id) / "trades.parquet")
    return _clean(tr.with_columns(
        pl.col("entry_time").dt.epoch("s"), pl.col("exit_time").dt.epoch("s")
    ).to_dicts())


@app.get("/api/runs/{run_id}/equity")
def run_equity(run_id: str, every: int = Query(1, ge=1, le=1000)):
    eq = pl.read_parquet(_run_dir(run_id) / "equity.parquet").gather_every(every)
    return _clean(eq.with_columns(pl.col("time").dt.epoch("s")).to_dicts())
