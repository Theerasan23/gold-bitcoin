"""คำสั่ง: python -m goldbtc <fetch|backtest|ablate|core|rules|walkforward|range|portfolio|timeframes|paper|serve>"""

from __future__ import annotations

import argparse
import sys

from .config import DATA_DIR, Params, parse_value

DEFAULT_SYMBOLS = ["BTCUSDT", "PAXGUSDT"]   # PAXG = โทเคนทองคำ ใช้แทน XAUUSD (ข้อมูลฟรีจาก Binance)
DEFAULT_INTERVALS = ["1h", "4h", "1d"]


def cmd_fetch(a) -> None:
    from . import data as dt
    for s in a.symbols:
        for i in a.intervals:
            path, n = dt.update(s, i)
            print(f"{s:10s} {i:4s} +{n:6d} แท่ง -> {path}")


def cmd_backtest(a) -> None:
    import polars as pl

    from . import backtest as bt
    from .metrics import summarize
    over = dict(kv.split("=", 1) for kv in a.set)
    p = Params().with_overrides(**{k: parse_value(v) for k, v in over.items()})
    res = bt.load_and_run(a.symbol, a.interval, p)
    m = summarize(res, a.split)
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=2, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print(m)
        if a.trades:
            print(res.trades.tail(a.trades))
    if a.save:
        print("saved run:", bt.save(res, m))


def cmd_ablate(a) -> None:
    from .experiments import ablate, print_table
    res, out = ablate(a.symbols, a.intervals, a.variants, a.split, workers=a.workers)
    print_table(res)
    print(f"\nผลทั้งหมด: {out}/ablation.csv")


def cmd_core(a) -> None:
    from datetime import datetime, timezone

    from .walkforward import Dataset, _stats, data_start

    def ms(day: str) -> int:
        return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)

    ds = Dataset(a.symbol, a.interval)
    p = ds.params({k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)})
    t0 = max(ms(a.start), int(data_start(ds.t).timestamp() * 1000))
    t1, split = int(ds.t[-1]) + 1, ms(a.split)
    for label, x0, x1 in (("all", t0, t1), ("in_sample", t0, split), ("out_sample", split, t1)):
        r, eq = ds.run(p, x0, x1)
        st = _stats(r)
        ret = (eq[-1] / eq[0] - 1) * 100 if eq.size else 0.0
        print(f"{label:11s} trades {st['trades']:4d} · R {st['total_r']:7.2f} · avg {st['avg_r']:5.2f} · "
              f"win {st['win_rate']:5.1f}% · SQN {st['sqn']:5.2f} · maxDD {st['max_dd_r']:5.1f}R · return {ret:7.1f}%")


def cmd_rules(a) -> None:
    from .walkforward import print_rules, run_rules
    res, out = run_rules(a.symbols, a.intervals, a.seeds, a.split, workers=a.workers)
    print_rules(res)
    print(f"\nผลทั้งหมด: {out}/rules.csv")


def cmd_walkforward(a) -> None:
    from .walkforward import print_report, run_all
    windows, summary, out = run_all(a.symbols, a.intervals, a.train_months, a.test_months, a.min_trades,
                                    a.seeds, a.anchored, workers=a.workers)
    print_report(windows, summary)
    print(f"\nผลทั้งหมด: {out}")


def cmd_range(a) -> None:
    from .meanrev import print_scan, run_scan
    res, out = run_scan(a.symbols, a.intervals, a.seeds, a.split, workers=a.workers)
    print_scan(res)
    print(f"\nผลทั้งหมด: {out}/range.csv")


def cmd_portfolio(a) -> None:
    from .portfolio import print_report, run
    table, corr, out = run(a.symbols, a.interval, a.split, a.risk_pct, a.max_lev)
    print_report(table, corr)
    print(f"\nผลทั้งหมด: {out}")


def cmd_paper(a) -> None:
    from . import paper as pp
    if a.action == "run":
        pp.MultiRunner(poll_s=a.poll).run_forever()
    elif a.action == "new":
        d = pp.new_run(pp.PaperConfig(capital=a.capital, risk_pct=a.risk_pct, interval=a.interval))
        print(f"เริ่มบัญชีเดโมใหม่ {d.name} · {a.interval} · ทุน {a.capital:,.0f} · เสี่ยง {a.risk_pct}%/ไม้ "
              f"(worker เริ่มเดินเองรอบถัดไป)")
    elif a.action == "status":
        active = pp.active_runs()
        if not active:
            print("ยังไม่มีบัญชีเดโม — รัน: paper new")
            return
        for tf, d in active.items():
            st = pp.Store(d).load_json("state.json") or {}
            cfg = pp.load_config(d)
            print(f"[{tf}] run {d.name} · ทุน {cfg.capital:,.0f} · equity {st.get('equity', cfg.capital):,.2f} · "
                  f"heartbeat {pp.iso(st.get('heartbeat_ms'))} UTC")
            for p in (st.get("engine") or {}).get("positions", {}).values():
                print(f"  ถือ {p['symbol']} {p['qty']:.6f} @ {p['entry_price']:,.2f} · SL {p['stop']:,.2f}")
            for t in pp.Store(d).read("trades")[-10:]:
                print(f"  {t['symbol']} {pp.iso(t['entry_time'])} -> {pp.iso(t['exit_time'])} {t['r']:+.2f}R ({t['reason']})")
    elif a.action == "compare":
        d = pp.active_runs().get(a.interval) or pp.current_run()
        res = pp.compare(d) if d else {"ready": False, "message": "ยังไม่มีบัญชีเดโม"}
        if not res["ready"]:
            print(res["message"])
            return
        sm = res["summary"]
        print(f"{pp.iso(res['from'])} -> {pp.iso(res['to'])} UTC · เดโม {sm['paper_return_pct']:+.2f}% · "
              f"backtest {sm['bt_return_pct']:+.2f}% · ไม้ตรงกัน {sm['matched']}/{max(sm['trades_paper'], sm['trades_bt'])}")
        for r in res["trades"]:
            print(f"  {r['symbol']} {pp.iso(r['entry_bar'])} {r['status']} · เข้า เดโม {r['paper_entry']} / bt {r['bt_entry']}"
                  f" · R เดโม {r['paper_r']} / bt {r['bt_r']}")


def cmd_timeframes(a) -> None:
    from .timeframes import print_report, run
    table, out = run(a.symbols, a.intervals, a.split, a.risk_pct)
    print_report(table)
    print(f"\nผลทั้งหมด: {out}/timeframes.csv")


def cmd_serve(a) -> None:
    import uvicorn
    uvicorn.run("goldbtc.api:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="goldbtc", description=f"BTC/Gold backtest engine (DATA_DIR={DATA_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="ดึง/อัปเดตแท่งเทียนจาก Binance ลง Parquet")
    f.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    f.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    f.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("backtest", help="รัน backtest หนึ่งชุด")
    b.add_argument("--symbol", default="BTCUSDT")
    b.add_argument("--interval", default="4h")
    b.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="เช่น min_score=70 tp_mode=fix")
    b.add_argument("--split", default="2023-01-01", help="วันแบ่ง in-sample / out-of-sample")
    b.add_argument("--trades", type=int, default=0, help="แสดงไม้ล่าสุด N ไม้")
    b.add_argument("--save", action="store_true", help="บันทึกผลลง data/results/")
    b.set_defaults(fn=cmd_backtest)

    x = sub.add_parser("ablate", help="ทดสอบตัดทีละส่วน")
    x.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    x.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    x.add_argument("--variants", nargs="*", default=None)
    x.add_argument("--split", default="2023-01-01")
    x.add_argument("--workers", type=int, default=None)
    x.set_defaults(fn=cmd_ablate)

    c = sub.add_parser("core", help="backtest กลยุทธ์แกนหลัก (core.py)")
    c.add_argument("--symbol", default="BTCUSDT")
    c.add_argument("--interval", default="4h")
    c.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="เช่น entry=asc_triangle stop=structure exit=measured regime=ma200")
    c.add_argument("--start", default="2018-01-01")
    c.add_argument("--split", default="2023-01-01")
    c.set_defaults(fn=cmd_core)

    w = sub.add_parser("walkforward", help="walk-forward ของกลยุทธ์แกนหลัก")
    w.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    w.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    w.add_argument("--train-months", type=int, default=36)
    w.add_argument("--test-months", type=int, default=12)
    w.add_argument("--min-trades", type=int, default=10, help="ช่วง train ต้องมีไม้อย่างน้อยเท่านี้จึงเลือกได้")
    w.add_argument("--seeds", type=int, default=200, help="จำนวนรอบของการเข้าแบบสุ่มที่ใช้เทียบ")
    w.add_argument("--anchored", action="store_true", help="ช่วง train เริ่มจากต้นข้อมูลเสมอ (แทนการเลื่อน)")
    w.add_argument("--workers", type=int, default=None)
    w.set_defaults(fn=cmd_walkforward)

    r = sub.add_parser("rules", help="ทดสอบกฎทีละข้อ (จากบทความ) เทียบการเข้าแบบสุ่ม")
    r.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    r.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    r.add_argument("--seeds", type=int, default=200)
    r.add_argument("--split", default="2023-01-01")
    r.add_argument("--workers", type=int, default=None)
    r.set_defaults(fn=cmd_rules)

    g = sub.add_parser("range", help="ทดสอบระบบตลาดไซด์เวย์ (mean reversion) เทียบการเข้าแบบสุ่ม")
    g.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    g.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    g.add_argument("--seeds", type=int, default=200)
    g.add_argument("--split", default="2023-01-01")
    g.add_argument("--workers", type=int, default=None)
    g.set_defaults(fn=cmd_range)

    q = sub.add_parser("portfolio", help="พอร์ตรวมหลายเหรียญ/หลายระบบ ใช้เงินก้อนเดียว")
    q.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    q.add_argument("--interval", default="4h")
    q.add_argument("--split", default="2023-01-01")
    q.add_argument("--risk-pct", type=float, default=1.0, help="เสี่ยงกี่ %% ของพอร์ตต่อไม้")
    q.add_argument("--max-lev", type=float, default=1.0, help="มูลค่าที่ถือรวม ≤ กี่เท่าของพอร์ต (spot = 1)")
    q.set_defaults(fn=cmd_portfolio)

    f2 = sub.add_parser("timeframes", help="ระบบเดียวกันบนทุก timeframe (ช่วงเวลาเดียวกัน) เทียบกัน")
    f2.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    f2.add_argument("--intervals", nargs="+", default=["15m", "1h", "4h", "1d"])
    f2.add_argument("--split", default="2023-01-01")
    f2.add_argument("--risk-pct", type=float, default=1.0)
    f2.set_defaults(fn=cmd_timeframes)

    k = sub.add_parser("paper", help="บัญชีเดโม (เงินจำลอง ราคาจริง) : run | new | status | compare")
    k.add_argument("action", choices=["run", "new", "status", "compare"])
    k.add_argument("--capital", type=float, default=10_000, help="ทุนเริ่มต้นของบัญชีเดโมใหม่")
    k.add_argument("--risk-pct", type=float, default=1.0)
    k.add_argument("--poll", type=float, default=15.0, help="worker เช็คราคาทุกกี่วินาที")
    k.add_argument("--interval", default="4h", choices=["15m", "1h", "4h", "1d"], help="timeframe ของบัญชี (new / compare)")
    k.set_defaults(fn=cmd_paper)

    s = sub.add_parser("serve", help="เปิด API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
