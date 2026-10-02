"""รูปแบบกราฟ + แท่งเทียน ตามนิยามในบทความ masterthecrypto (docs/masterthecrypto-notes.md)

ทุกฟังก์ชันคืนสัญญาณฝั่งขาขึ้น (long) — ฝั่งขาลงได้จากการกลับด้านราคา (mirror):
    h' = -l, l' = -h, c' = -c, o' = -o   -> double top = double bottom ของราคากลับด้าน

รูปแบบกราฟคืน (สัญญาณ, ระยะ SL ตามโครงสร้าง, ระยะเป้า measured move) โดยวัดจากราคาปิดแท่งสัญญาณ
ใช้เฉพาะจุดกลับตัวที่ยืนยันแล้ว และเข้าเมื่อ "ปิดทะลุ" ระดับสำคัญครั้งแรก (แท่งก่อนหน้ายังไม่ทะลุ)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import indicators as ta
from .structure import zigzag

NAN = np.nan


def swings(h: np.ndarray, l: np.ndarray, depth: int):
    """จุดกลับตัวสลับยอด/ก้น : P[i, k] = ราคาจุดที่ k ย้อนหลัง ณ แท่ง i, D0 = ทิศของจุดล่าสุด (1 ยอด / -1 ก้น)"""
    ph = ta.pivot_high(h, depth, depth)
    pl = ta.pivot_low(l, depth, depth)
    P, _, D0, _ = zigzag(ph, pl, np.zeros(h.size), depth, True)
    return P, D0


@njit(cache=True)
def _emit(sig, risk, tgt, i, c, prev_c, level, stop, target):
    if c > level and prev_c <= level and stop < c:
        sig[i] = True
        risk[i] = c - stop
        tgt[i] = target - c


@njit(cache=True)
def double_bottom(c, atr, P, D0, tol_atr, min_h_atr):
    """W : ก้น 2 ก้นสูงใกล้กัน (ต่างไม่เกิน tol x ATR) คั่นด้วยยอด = neckline · เข้าเมื่อปิดเหนือ neckline
    SL ใต้ก้นต่ำสุด · เป้า = neckline + ความสูงรูปแบบ"""
    n = c.size
    sig = np.zeros(n, np.bool_)
    risk = np.full(n, NAN)
    tgt = np.full(n, NAN)
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        if D0[i] == -1:
            l2, nk, l1 = P[i, 0], P[i, 1], P[i, 2]
        else:   # ยอดล่าสุดยังต่ำกว่า neckline = ยังอยู่ในรูปแบบ
            l2, nk, l1 = P[i, 1], P[i, 2], P[i, 3]
            if not (P[i, 0] < nk):
                continue
        if np.isnan(l1):
            continue
        low = min(l1, l2)
        height = nk - low
        if abs(l1 - l2) <= tol_atr * a and height >= min_h_atr * a:
            _emit(sig, risk, tgt, i, c[i], c[i - 1], nk, low - 0.25 * a, nk + height)
    return sig, risk, tgt


@njit(cache=True)
def inverse_head_shoulders(c, atr, P, D0, tol_atr, min_h_atr):
    """ไหล่ซ้าย - หัว (ต่ำสุด) - ไหล่ขวา สูงใกล้กัน · neckline = ยอดที่สูงกว่าระหว่างสองยอด (แบบระดับ เผื่อความปลอดภัย)
    SL ใต้ไหล่ขวา · เป้า = neckline + (neckline - หัว)"""
    n = c.size
    sig = np.zeros(n, np.bool_)
    risk = np.full(n, NAN)
    tgt = np.full(n, NAN)
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        o = 0 if D0[i] == -1 else 1
        rs, h2, hd, h1, ls = P[i, o], P[i, o + 1], P[i, o + 2], P[i, o + 3], P[i, o + 4]
        if np.isnan(ls):
            continue
        nk = max(h1, h2)
        if o == 1 and not (P[i, 0] < nk):
            continue
        if hd < ls - 0.5 * a and hd < rs - 0.5 * a and abs(ls - rs) <= tol_atr * a and nk - hd >= min_h_atr * a:
            _emit(sig, risk, tgt, i, c[i], c[i - 1], nk, rs - 0.25 * a, nk + (nk - hd))
    return sig, risk, tgt


@njit(cache=True)
def ascending_triangle(c, atr, P, D0, tol_atr, min_h_atr):
    """ยอด 2 ยอดสูงใกล้กัน (แนวต้านแนวนอน) + ก้นยกตัว · เข้าเมื่อปิดเหนือแนวต้าน
    SL ใต้ก้นล่าสุด (ตามบทความ) · เป้า = แนวต้าน + ความสูงฐาน"""
    n = c.size
    sig = np.zeros(n, np.bool_)
    risk = np.full(n, NAN)
    tgt = np.full(n, NAN)
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        if D0[i] == -1:
            low_b, hi_b, low_a, hi_a = P[i, 0], P[i, 1], P[i, 2], P[i, 3]
        else:
            hi_b, low_b, hi_a, low_a = P[i, 0], P[i, 1], P[i, 2], P[i, 3]
        if np.isnan(low_a) or np.isnan(hi_a):
            continue
        res = max(hi_a, hi_b)
        height = res - low_a
        if abs(hi_a - hi_b) <= tol_atr * a and low_b > low_a + 0.25 * a and height >= min_h_atr * a:
            _emit(sig, risk, tgt, i, c[i], c[i - 1], res, low_b - 0.25 * a, res + height)
    return sig, risk, tgt


@njit(cache=True)
def bull_flag(h, l, c, atr, pole_atr, pole_bars, min_cons, max_cons):
    """เสาธง: ราคาพุ่งขึ้น >= pole_atr x ATR ภายใน pole_bars แท่ง แล้วพักแคบ ๆ min_cons..max_cons แท่ง
    (ย่อไม่เกินครึ่งเสา ไม่ขึ้นเกินยอดเสา) · เข้าเมื่อปิดเหนือกรอบธง · SL ใต้กรอบธง · เป้า = จุดทะลุ + ความยาวเสา"""
    n = c.size
    sig = np.zeros(n, np.bool_)
    risk = np.full(n, NAN)
    tgt = np.full(n, NAN)
    for i in range(max_cons + pole_bars + 1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        for m in range(min_cons, max_cons + 1):
            c_hi = h[i - m]
            c_lo = l[i - m]
            for j in range(i - m + 1, i):
                c_hi = max(c_hi, h[j])
                c_lo = min(c_lo, l[j])
            # เสา: ช่วงก่อนธง หา low ก่อน แล้ว high หลัง low
            s = i - m - pole_bars
            lo_idx = s
            for j in range(s, i - m):
                if l[j] < l[lo_idx]:
                    lo_idx = j
            top = h[lo_idx]
            for j in range(lo_idx, i - m):
                top = max(top, h[j])
            pole = top - l[lo_idx]
            if pole < pole_atr * a:
                continue
            if c_hi <= top + 0.25 * a and c_lo >= top - 0.5 * pole and (c_hi - c_lo) <= 0.5 * pole:
                if c[i] > c_hi and c[i - 1] <= c_hi:
                    sig[i] = True
                    risk[i] = c[i] - (c_lo - 0.25 * a)
                    tgt[i] = c_hi + pole - c[i]
                break
    return sig, risk, tgt


# ----------------------------------------------------------------------------
# แท่งเทียน (ฝั่งกลับตัวขึ้น) — ใช้คู่กับบริบท "เพิ่งย่อลงมา"
# ----------------------------------------------------------------------------
def bullish_candles(o, h, l, c, atr) -> dict[str, np.ndarray]:
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    green = c > o
    red = c < o
    o1, c1, h1, l1 = (np.r_[np.nan, x[:-1]] for x in (o, c, h, l))
    o2, c2 = (np.r_[np.nan, np.nan, x[:-2]] for x in (o, c))
    body1 = np.abs(c1 - o1)
    body2 = np.abs(c2 - o2)
    red1 = c1 < o1
    red2 = c2 < o2
    c4 = np.r_[np.full(4, np.nan), c[:-4]]
    declined = c1 < c4                                   # ก่อนหน้าราคาย่อลงมา
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    low5 = np.r_[np.full(4, np.nan), np.lib.stride_tricks.sliding_window_view(l, 5).min(axis=1)]

    engulf = green & red1 & (o <= c1) & (c >= o1) & (body > body1)
    hammer = green & (lower_wick >= 2 * body) & (upper_wick <= 0.25 * rng) & (l <= low5)
    morning = red2 & (body2 >= 0.5 * atr) & (body1 <= 0.3 * body2) & green & (c > (o2 + c2) / 2)
    piercing = red1 & (body1 >= 0.5 * atr) & green & (o <= c1) & (c > (o1 + c1) / 2) & (c < o1)
    return {
        "engulfing": engulf & declined,
        "hammer": hammer & declined,
        "morning_star": morning & declined,
        "piercing": piercing & declined,
    }


def mirror(o, h, l, c):
    """กลับด้านราคา : รูปแบบขาขึ้นของราคากลับด้าน = รูปแบบขาลงของราคาจริง"""
    return -o, -l, -h, -c
