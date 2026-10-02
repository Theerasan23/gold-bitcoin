import numpy as np
import pytest

from goldbtc.portfolio import Sleeve, simulate_portfolio

H = 4 * 3_600_000


def _sleeve(name, sym, closes, trades):
    t = np.arange(len(closes), dtype=np.int64) * H
    return Sleeve(name, sym, t, np.array(closes, float), trades, np.zeros(len(trades)))


def test_risk_sizing_on_portfolio_equity():
    # เสี่ยง 1% ของ 10,000 = 100 · ระยะ SL 5 -> 20 หน่วย · ราคาขึ้น 10 = +200
    tr = {"entry_ms": 1 * H, "side": 1, "entry_px": 100.0, "risk": 5.0, "exits": [(3 * H, 110.0, 1.0)]}
    s = _sleeve("A", "AAA", [100, 100, 105, 110, 110], [tr])
    res = simulate_portfolio([s], 0, 5 * H, risk_pct=1.0, max_lev=100, comm_pct=0.0)
    assert res["equity"][2] == pytest.approx(10_100)   # mark-to-market ระหว่างถือ
    assert res["equity"][-1] == pytest.approx(10_200)


def test_notional_cap_shrinks_second_position():
    # ไม้แรกใช้ notional 80% ของพอร์ต ไม้ที่สองเหลือที่ว่างแค่ 20%
    a = {"entry_ms": 1 * H, "side": 1, "entry_px": 100.0, "risk": 1.25, "exits": [(4 * H, 100.0, 1.0)]}
    b = {"entry_ms": 2 * H, "side": 1, "entry_px": 50.0, "risk": 1.0, "exits": [(3 * H, 60.0, 1.0)]}
    s1 = _sleeve("A", "AAA", [100] * 6, [a])
    s2 = _sleeve("B", "BBB", [50] * 6, [b])
    res = simulate_portfolio([s1, s2], 0, 6 * H, risk_pct=1.0, max_lev=1.0, comm_pct=0.0)
    # ไม้ A : 100/1.25 = 80 หน่วย x 100 = 8,000 · ไม้ B ได้แค่ 2,000/50 = 40 หน่วย (แทน 100) -> +400
    assert res["equity"][-1] == pytest.approx(10_400)
    assert res["skipped"] == 0
