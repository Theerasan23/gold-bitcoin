"""entry_on=touch : เข้าเมื่อราคาแตะจุดระหว่างแท่ง · เข้าเพิ่มได้หลายไม้ · SL แยกต่อไม้ · ทิศกลับปิดทุกไม้"""

import numpy as np
import pytest

from goldbtc import backtest as bt
from goldbtc import paper as pp

H = 4 * 3_600_000
NAN = np.nan
#          open   high   low    close  level  exit
BARS = [(95.0, 97.0, 94.0, 96.0, 98.0, False),     # 0 ยังไม่แตะ 98
        (96.0, 102.0, 95.0, 101.0, 98.0, False),   # 1 แตะ 98 ระหว่างแท่ง -> ไม้ A ที่ 98 (ไม่ต้องรอแท่งปิด)
        (101.0, 101.5, 99.0, 100.0, 102.0, False),  # 2 สูงกว่าไม้ A แต่ไม่ถึง High เดิม 102 -> ไม่เข้า
        (100.0, 104.0, 99.5, 103.0, 102.0, False),  # 3 แตะ 102 -> ไม้ B (ไม้ A ยังถืออยู่)
        (103.0, 103.5, 96.0, 97.0, 104.0, False),   # 4 ลงถึง 96 : SL ไม้ B (97) โดน · ไม้ A (93) ยังอยู่
        (97.0, 98.0, 96.5, 97.5, 96.0, False),      # 5 จุดแตะ 96 แต่ต้องเหนือไม้ล่าสุด (102) ด้วย -> ไม่เข้า
        (97.0, 99.0, 96.8, 98.0, 104.0, True),      # 6 ทิศกลับ -> ปิดไม้ที่เหลือที่ราคาเปิด
        (98.0, 99.0, 97.0, 98.5, 104.0, False)]
RISK = 5.0


def _arrays():
    a = np.array([b[:5] for b in BARS])
    o, h, l, c, level = (a[:, k].copy() for k in range(5))
    ex = np.array([b[5] for b in BARS])
    n = len(BARS)
    return o, h, l, c, level, ex, n


def test_backtest_touch_entries_pyramid_and_separate_stops():
    o, h, l, c, level, ex, n = _arrays()
    f = np.zeros(n, bool)
    nan = np.full(n, NAN)
    T, eq, pos, _ = bt.simulate_touch(o, h, l, c, nan, level, nan, ~ex, f, ex, f, np.full(n, RISK), nan, nan,
                                      1, False, 3.0, 5.0, 10_000.0, 0.0, 0.0, 1.0, 100.0, 0, False)
    rows = [(int(r[0]), int(r[1]), r[3], r[4], bt.EXIT_NAMES[int(r[8])]) for r in T]
    assert rows == [(1, 6, 98.0, 97.0, "htf_reversal"), (3, 4, 102.0, 97.0, "stop")]
    assert list(pos) == [0, 1, 1, 2, 1, 1, 0, 0]
    assert T[0, 5] == pytest.approx(10_000 * 0.01 / RISK)   # เสี่ยง 1% ต่อไม้


def test_paper_engine_follows_same_rules():
    o, h, l, c, level, ex, n = _arrays()
    trades = []
    cfg = pp.PaperConfig(symbols=["AAAUSDT"], commission_pct=0.0, slippage_ticks=0, max_lev=100.0)
    eng = pp.Engine(cfg, sink=lambda k, r: trades.append(r) if k == "trades" else None, tick={"AAAUSDT": 0.01})
    held = []
    for i in range(n):
        t = i * H
        b = pp.Bar(t, o[i], h[i], l[i], c[i], NAN, level[i], NAN, not ex[i], False, bool(ex[i]), False, RISK)
        eng.begin_bar(t)
        eng.on_bar_open("AAAUSDT", b, t, o[i], o[i])
        eng.on_price("AAAUSDT", t, o[i], h[i], l[i], c[i])
        eng.on_bar_close("AAAUSDT", b)
        held.append(len(eng.of("AAAUSDT")))
    assert held == [0, 1, 1, 2, 1, 1, 0, 0]
    got = [(x["entry_bar"] // H, x["exit_time"] // H, x["entry_price"], x["exit_price"], x["reason"]) for x in trades]
    assert sorted(got) == [(1, 6, 98.0, 97.0, "trend_flip"), (3, 4, 102.0, 97.0, "stop")]


def test_paper_state_roundtrip_and_legacy_positions():
    cfg = pp.PaperConfig(symbols=["AAAUSDT"])
    legacy = {"cash": 9_990.0, "positions": {"AAAUSDT": {
        "symbol": "AAAUSDT", "side": 1, "qty": 0.5, "entry_price": 100.0, "ref_price": 99.9, "entry_time": 5,
        "entry_bar": 0, "risk": 4.0, "init_stop": 96.0, "stop": 97.0, "ext": 110.0, "trail": 97.0}},
        "pending": {}, "last_close": {"AAAUSDT": 105.0}, "eq_prev": 10_000.0}
    eng = pp.Engine(cfg, legacy)
    assert [u.stop for u in eng.of("AAAUSDT")] == [97.0]
    assert eng.camp["AAAUSDT"]["last_px"] == 100.0
    again = pp.Engine(cfg, eng.state())
    assert again.state() == eng.state()
