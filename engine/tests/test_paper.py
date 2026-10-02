"""บัญชีเดโมต้องให้ผลเหมือน backtest เมื่อป้อนข้อมูลชุดเดียวกัน (ต่างกันได้แค่ราคาที่ได้จริงในโหมด live)"""

import json

import numpy as np
import polars as pl
import pytest

from goldbtc import data as dt
from goldbtc import paper as pp
from goldbtc.portfolio import build_sleeve, simulate_portfolio

H = 4 * 3_600_000
T0 = 1_577_836_800_000  # 2020-01-01
N = 6000


def _write(data_dir, symbol, seed, n=N, vol=0.012):
    """random walk ที่มีช่วงเทรนด์ขึ้น/ลงสลับกัน -> มีสัญญาณ breakout และไม้จริง"""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([0.004, -0.003, 0.0], size=n // 250 + 1), 250)[:n]
    r = drift + rng.normal(0, vol, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, n))
    df = pl.DataFrame({
        "open_time": pl.Series(T0 + np.arange(n, dtype=np.int64) * H).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "open": o, "high": np.maximum(o, c) * (1 + wick), "low": np.minimum(o, c) * (1 - wick), "close": c,
        "volume": np.ones(n),
    })
    path = dt.candle_path(symbol, "4h", data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


@pytest.fixture
def two_symbols(tmp_path):
    _write(tmp_path, "AAAUSDT", 1)
    _write(tmp_path, "BBBUSDT", 2, vol=0.008)
    return tmp_path


def test_replay_matches_portfolio_backtest(two_symbols):
    d = two_symbols
    cfg = pp.PaperConfig(symbols=["AAAUSDT", "BBBUSDT"], max_lev=5.0)
    t0, t1 = T0 + 600 * H, T0 + N * H
    # backtest พอร์ต
    sleeves = [build_sleeve("trend", s, "4h", t0, t1, d) for s in cfg.symbols]
    bt = simulate_portfolio(sleeves, t0, t1, cfg.risk_pct, cfg.max_lev, cfg.commission_pct, cfg.capital)
    # เดโมแบบ replay
    trades, eq = [], {}
    eng = pp.Engine(cfg, sink=lambda kind, rec: trades.append(rec) if kind == "trades" else None,
                    tick={s: 0.01 for s in cfg.symbols})
    frames = {s: pp.signal_frame(dt.load(s, "4h", d), "4h", cfg, 0.01) for s in cfg.symbols}
    pp.replay(eng, frames, t0, t1, on_mark=lambda t, e: eq.__setitem__(t, e.eq_prev))

    n_bt = sum(len(s.trades) for s in sleeves)
    assert n_bt > 20, "ข้อมูลทดสอบต้องมีไม้มากพอ"
    # ไม้ที่ปิดแล้วต้องตรงกันทุกไม้ (backtest บังคับปิดไม้สุดท้ายที่แท่งสุดท้าย -> ไม่นับ)
    last_bar = T0 + (N - 1) * H
    bt_closed = {(s.symbol, tr["entry_ms"], tr["exits"][-1][0], round(tr["exits"][-1][1], 6))
                 for s in sleeves for tr in s.trades if tr["exits"][-1][0] != last_bar}
    pp_closed = {(x["symbol"], x["entry_bar"], x["exit_time"], round(x["exit_price"], 6)) for x in trades}
    assert pp_closed == bt_closed
    # equity ทุกแท่ง (ยกเว้นแท่งสุดท้าย) ต้องเท่ากัน
    k = len(bt["time"]) - 1
    got = np.array([eq[int(t)] for t in bt["time"][:k]])
    np.testing.assert_allclose(got, bt["equity"][:k], rtol=1e-9)


class FakeMarket:
    """ตลาดปลอม : ราคา = ราคาปิดแท่งล่าสุด, แท่ง 1 นาทีไม่มีการแตะ SL"""

    def __init__(self, price):
        self.price = price

    def prices(self, symbols):
        return {s: self.price for s in symbols}

    def minutes(self, symbol, start_ms, end_ms=None):
        return [(start_ms, self.price, self.price, self.price)]


def test_live_runner_cycle(two_symbols, monkeypatch):
    d = two_symbols
    df = dt.load("AAAUSDT", "4h", d)
    last = int(df["open_time"].max().timestamp() * 1000)
    now = {"t": (last + H + 5 * 60_000) / 1000}
    monkeypatch.setattr(pp.time, "time", lambda: now["t"])
    monkeypatch.setattr(pp.dt, "update", lambda *a, **k: (None, 0))
    pp.new_run(pp.PaperConfig(symbols=["AAAUSDT", "BBBUSDT"]), d)

    run = pp.LiveRunner(d, market=FakeMarket(float(df["close"][-1])))
    run.step()   # รอบแรก : เริ่มนับจากแท่งที่ปิดล่าสุด
    st = json.loads((run.run_dir / "state.json").read_text())
    assert st["started_bar"] == last and st["equity"] == pytest.approx(10_000)

    # แท่งใหม่ปิด -> worker ต้องประมวลผลแท่งนั้น 1 ครั้ง
    for sym, seed in (("AAAUSDT", 1), ("BBBUSDT", 2)):
        full = _write(d, sym, seed, n=N + 1)
        assert full.height == N + 1
    now["t"] = (last + 2 * H + 5 * 60_000) / 1000
    run.step()
    st = json.loads((run.run_dir / "state.json").read_text())
    assert st["last_bar"]["AAAUSDT"] == last + H
    eq_rows = [json.loads(x) for x in (run.run_dir / "equity.jsonl").read_text().splitlines()]
    assert any(r["source"] == "bar_close" and r["time"] == last + 2 * H for r in eq_rows)


class ReplayMarket:
    """ตลาดจำลองจากข้อมูลเต็ม : ราคาตอนนี้ = ราคาเปิดแท่งที่กำลังวิ่ง, แท่ง 1 นาที = OHLC ของแท่ง 4h นั้น"""

    def __init__(self, dfs):
        self.d = {s: (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()) for s, df in dfs.items()}
        self.k = 0

    def prices(self, symbols):
        return {s: float(self.d[s][0][self.k]) for s in symbols}

    def minutes(self, symbol, start_ms, end_ms=None):
        i = (start_ms - T0) // H
        if i > self.k:
            return []
        o, h, l = self.d[symbol]
        return [(start_ms, float(o[i]), float(h[i]), float(l[i]))]


def test_live_worker_matches_backtest_bar_by_bar(tmp_path, monkeypatch):
    """worker เดินจริงทีละแท่ง (ราคาที่ได้ = ราคาเปิดแท่ง) -> ผลเทียบต้อง 'ตรงกัน' ทุกไม้ R เท่ากัน"""
    syms = ["AAAUSDT", "BBBUSDT"]
    full = {"AAAUSDT": _write(tmp_path, "AAAUSDT", 1), "BBBUSDT": _write(tmp_path, "BBBUSDT", 2, vol=0.008)}
    clock = {"t": 0.0}
    monkeypatch.setattr(pp.time, "time", lambda: clock["t"])
    monkeypatch.setattr(pp.dt, "update", lambda *a, **k: (None, 0))
    pp.new_run(pp.PaperConfig(symbols=syms, max_lev=5.0), tmp_path)
    mkt = ReplayMarket(full)

    start, steps = 3000, 2000
    run = None
    for k in range(start, start + steps):
        for s in syms:   # ข้อมูลที่ "รู้" ณ เวลานี้ = แท่งที่ปิดแล้ว [0, k)
            full[s].head(k).write_parquet(dt.candle_path(s, "4h", tmp_path))
        mkt.k = k
        clock["t"] = (T0 + k * H + 5 * 60_000) / 1000
        run = run or pp.LiveRunner(tmp_path, market=mkt)
        run.step()

    res = pp.compare(run.run_dir, tmp_path)
    assert res["ready"]
    rows = res["trades"]
    assert len(rows) >= 5, "ช่วงทดสอบต้องมีไม้"
    assert all(r["status"] == "ตรงกัน" for r in rows), [r for r in rows if r["status"] != "ตรงกัน"]
    for r in rows:
        assert r["entry_diff_pct"] == pytest.approx(0.0, abs=1e-9)
        if r["paper_r"] is not None and r["bt_r"] is not None:
            assert r["paper_r"] == pytest.approx(r["bt_r"], abs=1e-9)


def test_one_active_account_per_timeframe(tmp_path, monkeypatch):
    clock = {"t": 1_790_000_000.0}
    real_gmtime = pp.time.gmtime
    monkeypatch.setattr(pp.time, "time", lambda: clock["t"])
    monkeypatch.setattr(pp.time, "gmtime", lambda *a: real_gmtime(clock["t"]))
    a = pp.new_run(pp.PaperConfig(interval="4h"), tmp_path)
    clock["t"] += 1
    b = pp.new_run(pp.PaperConfig(interval="15m"), tmp_path)
    clock["t"] += 1
    c = pp.new_run(pp.PaperConfig(interval="4h", capital=100), tmp_path)   # แทน 4h เดิม
    active = pp.active_runs(tmp_path)
    assert active == {"4h": c, "15m": b}
    assert a.exists()                                  # รอบเก่าเก็บไว้ ไม่ลบ
    assert pp.load_config(active["4h"]).capital == 100
