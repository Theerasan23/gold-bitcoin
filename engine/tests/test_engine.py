from datetime import datetime

import numpy as np
import polars as pl
import pytest

from goldbtc import backtest as bt
from goldbtc import indicators as ta
from goldbtc import structure as st
from goldbtc.config import Params, htf_resolution
from goldbtc.signals import htf_trend


# ---------------------------------------------------------------- indicators
def test_rma_seeds_with_sma():
    x = np.array([1.0, 2, 3, 4, 5])
    r = ta.rma(x, 3)
    assert np.isnan(r[:2]).all()
    assert r[2] == pytest.approx(2.0)
    assert r[3] == pytest.approx((4 + 2 * 2.0) / 3)


def test_rsi_bounds_and_extremes():
    up = np.arange(1, 60, dtype=float)
    assert ta.rsi(up, 14)[-1] == pytest.approx(100.0)
    rng = np.random.default_rng(0)
    r = ta.rsi(100 + rng.normal(0, 1, 500).cumsum(), 14)
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 100


def test_linreg_slope_on_line():
    c = 3.0 * np.arange(100) + 7
    slope, corr = ta.linreg_stats(c, 20)
    assert slope[-1] == pytest.approx(3.0)
    assert corr[-1] == pytest.approx(1.0)


def test_pivots_confirm_after_right_bars():
    h = np.array([1, 2, 3, 9, 3, 2, 1, 1], dtype=float)
    ph = ta.pivot_high(h, 2, 2)
    assert ph[5] == 9 and np.isnan(np.delete(ph, 5)).all()


def test_market_structure_flip():
    # ขึ้นทำยอด -> ย่อทำก้น -> ปิดหลุดก้น = กลับตัวลง
    c = np.array([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 12, 13, 14, 12, 9, 8], dtype=float)
    atr = np.full(c.size, 0.1)
    ph = ta.pivot_high(c, 1, 1)
    pl_ = ta.pivot_low(c, 1, 1)
    d, up, dn = st.market_structure(ph, pl_, c, atr, 0.0)
    assert d[7] == 1            # ทะลุยอดที่ 12
    assert dn.any() and d[-1] == -1


def test_htf_uses_previous_closed_bar_only():
    t = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 4, 23), "1h", eager=True) \
        .dt.replace_time_zone("UTC")
    close = np.r_[np.full(24, 100.0), np.full(24, 200.0), np.full(24, 300.0), np.full(24, 400.0)]
    _, _, slow = htf_trend(t, close, "1d", 1, 1)
    assert np.isnan(slow[:24]).all()       # วันแรกยังไม่มีวันก่อนหน้า
    assert (slow[24:48] == 100).all()      # วันที่ 2 เห็นแค่ราคาปิดวันที่ 1
    assert (slow[72:] == 300).all()        # ไม่แอบดูราคาวันปัจจุบัน


def test_htf_resolution_matches_pine():
    assert htf_resolution("4h", "1d") == "1d"
    assert htf_resolution("1d", "1d") == "1w"
    assert htf_resolution("1w", "1d") == "1M"


# ---------------------------------------------------------------- simulator
def _sim(o, h, l, c, long_sig, tp_mode=0, atr=1.0, extras=None, exit_mode=0, max_bars=0, exit_long=None, **kw):
    n = len(c)
    z = np.zeros(n, bool)
    a = dict(use_htf=False, exit_on_rev=False, cooldown=0, tp_mode=tp_mode, atr_mult_sl=2.0, trail_mult=3.0,
             tp1_r=2.0, tp1_frac=0.5, rr=3.0, use_be=False, capital=10_000.0, qty_pct=100.0, comm=0.0,
             slip=0.0, protect_fill=True, size_mode=0, risk_pct=1.0, max_lev=1.0)
    a.update(kw)
    ex = bt.nan_extras(n) if extras is None else extras
    xl = z if exit_long is None else np.array(exit_long, bool)
    return bt.simulate(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float),
                       np.full(n, atr), np.array(long_sig, bool), z, z, xl, z, z, *a.values(), *ex, exit_mode,
                       max_bars)


def test_entry_next_open_and_stop_hit():
    # สัญญาณแท่ง 0 -> เข้าที่ open แท่ง 1 = 100, SL = 100 - 2*ATR = 98
    o = [99, 100, 100, 99]
    h = [100, 101, 100.5, 99]
    l = [98.5, 99.5, 97, 95]
    c = [100, 100, 98, 96]
    T, eq, pos = _sim(o, h, l, c, [1, 0, 0, 0])
    assert T.shape[0] == 1
    assert T[0, 3] == 100 and T[0, 4] == 98
    assert T[0, 8] == bt.EXIT_STOP and T[0, 10] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    o = [99, 100, 95, 95]
    h = [100, 100.5, 96, 96]
    l = [98.5, 99.5, 94, 94]
    c = [100, 100, 95, 95]
    T, _, _ = _sim(o, h, l, c, [1, 0, 0, 0])
    assert T[0, 4] == 95  # ไม่ได้ราคา 98 เพราะเปิด gap ลงมาแล้ว


def test_fixed_tp_and_intrabar_order():
    # แท่ง 2 ชนทั้ง TP(106) และ SL(98) — open ใกล้ high กว่า -> ไป high ก่อน = TP
    o = [99, 100, 105.5, 100]
    h = [100, 101, 107, 100]
    l = [98.5, 99.5, 97, 100]
    c = [100, 100, 99, 100]
    T, _, _ = _sim(o, h, l, c, [1, 0, 0, 0], tp_mode=2)
    assert T[0, 8] == bt.EXIT_TP and T[0, 4] == 106


def test_mix_mode_partial_then_trailing_same_stop():
    o = [99, 100, 101, 103, 104, 99]
    h = [100, 101, 105, 106, 104.5, 99]
    l = [98.5, 99.5, 100.5, 102.5, 103, 90]
    c = [100, 100.5, 104.5, 105.5, 104, 92]
    T, _, _ = _sim(o, h, l, c, [1, 0, 0, 0, 0, 0], tp_mode=1)
    reasons = list(T[:, 8])
    assert bt.EXIT_TP1 in reasons
    assert T[:, 6].sum() == pytest.approx(1.0)   # ปิดครบ 100% ของไม้
    assert np.isclose(T[T[:, 8] == bt.EXIT_TP1, 4], 104).all()


def test_equity_consistent_with_trades():
    rng = np.random.default_rng(1)
    n = 400
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.5
    l = np.minimum(o, c) - 0.5
    sig = rng.random(n) < 0.05
    T, eq, _ = _sim(o, h, l, c, sig, comm=0.0005, slip=0.01)
    assert eq[-1] == pytest.approx(10_000 + T[:, 7].sum())


def test_params_reject_unknown():
    with pytest.raises(ValueError):
        Params().with_overrides(nope=1)


def test_risk_sizing_loses_risk_pct_on_stop():
    # เสี่ยง 1% : SL 2 ATR = 2 -> qty = 100 / 2 = 50 หน่วย, โดน SL = -100 (1% ของ 10,000)
    o = [99, 100, 100, 99]
    h = [100, 101, 100.5, 99]
    l = [98.5, 99.5, 97, 95]
    c = [100, 100, 98, 96]
    T, eq, _ = _sim(o, h, l, c, [1, 0, 0, 0], size_mode=1, risk_pct=1.0, max_lev=10.0)
    assert T[0, 5] == pytest.approx(50.0)
    assert eq[-1] == pytest.approx(9_900.0)


def test_risk_sizing_respects_leverage_cap():
    T, _, _ = _sim([99, 100, 100, 99], [100, 101, 100.5, 99], [98.5, 99.5, 97, 95], [100, 100, 98, 96],
                   [1, 0, 0, 0], size_mode=1, risk_pct=1.0, max_lev=0.2)
    assert T[0, 5] == pytest.approx(0.2 * 10_000 / 100)


def test_structural_stop_and_measured_target():
    # สัญญาณแท่ง 0 : SL ตามโครงสร้าง 1.0 ต่ำกว่าราคาปิด, เป้า 5.0 เหนือราคาปิด -> เข้า 100 SL 99 TP 105
    n = 4
    nan = np.full(n, np.nan)
    risk = nan.copy(); risk[0] = 1.0
    tgt = nan.copy(); tgt[0] = 5.0
    T, _, _ = _sim([99, 100, 101, 104], [100, 101, 106, 104], [98.5, 99.5, 100.5, 103], [100, 100.5, 104, 104],
                   [1, 0, 0, 0], tp_mode=3, extras=(risk, nan, tgt, nan, nan, nan))
    assert T[0, 9] == pytest.approx(1.0)       # risk = ระยะ SL ตามโครงสร้าง
    assert T[0, 8] == bt.EXIT_TP and T[0, 4] == pytest.approx(105.0)


def test_external_trailing_level():
    # trailing จากภายนอก (เช่น SAR) ยก SL จาก 98 ขึ้นมา 100.8 -> ออกที่ 100.8
    n = 5
    nan = np.full(n, np.nan)
    tr = nan.copy(); tr[1] = 99.0; tr[2] = 100.8
    T, _, _ = _sim([99, 100, 101, 102, 101], [100, 101, 102.5, 102.5, 101], [98.5, 99.5, 100.9, 101.5, 100],
                   [100, 100.5, 102, 102, 100.2], [1, 0, 0, 0, 0], extras=(nan, nan, nan, nan, tr, nan), exit_mode=1)
    assert T[0, 4] == pytest.approx(100.8) and T[0, 8] == bt.EXIT_TRAIL


def test_time_stop_and_signal_exit_without_trailing():
    o = [99, 100, 101, 102, 103, 104, 105]
    h = [100, 101, 102, 103, 104, 105, 106]
    l = [98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5]
    c = [100, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5]
    # time stop 3 แท่ง : เข้าแท่ง 1 -> ครบ 3 แท่งที่แท่ง 3 -> ออกที่ open แท่ง 4 = 103
    T, _, _ = _sim(o, h, l, c, [1, 0, 0, 0, 0, 0, 0], tp_mode=4, max_bars=3)
    assert T[0, 8] == bt.EXIT_TIME and T[0, 4] == 103
    # สัญญาณออก (เช่น ราคากลับถึงเส้นกลาง) แท่ง 2 -> ออกที่ open แท่ง 3 = 102
    T, _, _ = _sim(o, h, l, c, [1, 0, 0, 0, 0, 0, 0], tp_mode=4, exit_on_rev=True,
                   exit_long=[0, 0, 1, 0, 0, 0, 0])
    assert T[0, 8] == bt.EXIT_STRUCT and T[0, 4] == 102
