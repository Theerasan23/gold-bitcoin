from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from goldbtc.core import ENTRIES, CoreParams, net_r, raw_signals
from goldbtc.walkforward import make_windows


def _frame(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(rng.normal(0, 0.01, n).cumsum())
    o = np.r_[c[0], c[:-1]]
    t = pl.Series(np.arange(n, dtype=np.int64) * 4 * 3600 * 1000 + 1_577_836_800_000) \
        .cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")   # แท่ง 4h เริ่ม 2020-01-01
    return pl.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) * 1.002,
                         "low": np.minimum(o, c) * 0.998, "close": c, "volume": np.ones(n)})


def test_core_signals_do_not_look_ahead():
    # คำนวณจากข้อมูลเต็ม เทียบกับข้อมูลที่ตัดท้ายทิ้ง : ค่าในส่วนที่ซ้อนกันต้องเท่ากันทุกแท่ง
    df = _frame()
    for entry in ENTRIES:
        p = CoreParams(entry=entry, allow_short=True, regime="none")
        full = raw_signals(df, "4h", p)
        cut = raw_signals(df.head(2000), "4h", p)
        for k in ("long", "short"):
            assert (full[k][:2000] == cut[k]).all(), (entry, k)
        for k in ("risk_l", "tgt_l"):
            np.testing.assert_array_equal(full["raw"][k][:2000], cut["raw"][k], err_msg=f"{entry} {k}")
    # ตัวกรองทิศทุกแบบ ต้องใช้เฉพาะแท่ง TF ใหญ่ที่ปิดแล้ว
    full = raw_signals(df, "4h", CoreParams())
    cut = raw_signals(df.head(2000), "4h", CoreParams())
    for name, (up, down) in full["regimes"].items():
        assert (up[:2000] == cut["regimes"][name][0]).all(), name
        assert (down[:2000] == cut["regimes"][name][1]).all(), name
    np.testing.assert_array_equal(full["psar_l"][:2000], cut["psar_l"])


def test_windows_are_contiguous_and_out_of_sample():
    t = np.arange(0, 8 * 365 * 86_400_000, 3_600_000, dtype=np.int64) + 1_500_000_000_000
    w = make_windows(t, 36, 12)
    assert len(w) >= 4
    for (tr0, tr1, te0, te1), nxt in zip(w, w[1:] + [None]):
        assert tr0 < tr1 == te0 < te1          # train จบตรงที่ test เริ่ม
        if nxt:
            assert te1 == nxt[2]               # test ต่อกันพอดี ไม่ทับ ไม่เว้น
    d0 = datetime.fromtimestamp(w[0][2] / 1000, tz=timezone.utc)
    assert d0.day == 1 and d0.hour == 0


def test_net_r_groups_partial_exits():
    T = np.full((3, 11), np.nan)
    # entry_bar, exit_bar, side, entry_px, exit_px, qty, frac, pnl, reason, risk, r
    T[0] = [5, 9, 1, 100, 104, 0.5, 0.5, 2.0, 3, 2.0, 2.0]
    T[1] = [5, 12, 1, 100, 98, 0.5, 0.5, -1.0, 1, 2.0, -1.0]
    T[2] = [20, 25, 1, 100, 98, 1.0, 1.0, -2.0, 0, 2.0, -1.0]
    r = net_r(T)
    assert np.allclose(r, [0.5, -1.0])


def test_double_bottom_detects_w_breakout():
    from goldbtc import patterns as pt
    # ลง -> ก้น 1 (90) -> เด้ง 100 (neckline) -> ก้น 2 (90.5) -> ทะลุ 100
    path = [110, 105, 100, 95, 90, 95, 100, 95, 90.5, 95, 99, 101, 103]
    c = np.repeat(np.array(path, float), 3)
    h, l = c + 0.2, c - 0.2
    atr = np.full(c.size, 1.0)
    P, D0 = pt.swings(h, l, 2)
    sig, risk, tgt = pt.double_bottom(c, atr, P, D0, 1.0, 2.0)
    i = int(np.argmax(sig))
    assert sig.sum() == 1 and c[i] > 100 >= c[i - 1]
    assert c[i] - risk[i] == pytest.approx(89.8 - 0.25)          # SL ใต้ก้นต่ำสุด
    assert c[i] + tgt[i] == pytest.approx(100.2 + (100.2 - 89.8)) # เป้า = neckline + ความสูง
