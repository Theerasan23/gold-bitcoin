"""ส่วนที่ต้องจำสถานะข้ามแท่ง: แนวรับแนวต้าน, โครงสร้างราคา, ZigZag + Elliott, Fib, Divergence, Volume Profile

พอร์ตตรงจาก gold_bitcoin.pine (ทุกแท่งถือว่าเป็นแท่งปิดแล้ว = พฤติกรรม historical ของ Pine)
"""

from __future__ import annotations

import numpy as np
from numba import njit

NAN = np.nan
ZZ_CAP = 16  # Pine เก็บจุดกลับตัวไว้ 16 จุด


# ----------------------------------------------------------------------------
# แนวรับ - แนวต้าน (f_addLevel / f_nearest)
# ----------------------------------------------------------------------------
@njit(cache=True)
def sr_nearest(pv_high, pv_low, atr, close, merge_atr, max_n):
    n = close.size
    lv = np.empty(max_n + 2)
    cnt = 0
    near_res = np.full(n, NAN)
    near_sup = np.full(n, NAN)
    for i in range(n):
        tol = atr[i] * merge_atr
        for k in range(2):
            v = pv_high[i] if k == 0 else pv_low[i]
            if np.isnan(v):
                continue
            dup = False
            for j in range(cnt):
                if abs(lv[j] - v) <= tol:
                    dup = True
                    break
            if not dup:
                lv[cnt] = v
                cnt += 1
                if cnt > max_n:
                    for j in range(cnt - 1):
                        lv[j] = lv[j + 1]
                    cnt -= 1
        best_r = NAN
        best_s = NAN
        px = close[i]
        for j in range(cnt):
            v = lv[j]
            if v > px and (np.isnan(best_r) or v - px < best_r - px):
                best_r = v
            if v < px and (np.isnan(best_s) or px - v < px - best_s):
                best_s = v
        near_res[i] = best_r
        near_sup[i] = best_s
    return near_res, near_sup


# ----------------------------------------------------------------------------
# โครงสร้างราคา : Break of Structure / Change of Character
# ----------------------------------------------------------------------------
@njit(cache=True)
def market_structure(rv_ph, rv_pl, close, atr, buf):
    n = close.size
    sdir = np.zeros(n, np.int64)
    ch_up = np.zeros(n, np.bool_)
    ch_dn = np.zeros(n, np.bool_)
    hi = NAN
    lo = NAN
    d = 0
    for i in range(n):
        if not np.isnan(rv_ph[i]):
            hi = rv_ph[i]
        if not np.isnan(rv_pl[i]):
            lo = rv_pl[i]
        if not np.isnan(atr[i]):
            if d != 1 and not np.isnan(hi) and close[i] > hi + atr[i] * buf:
                ch_up[i] = d == -1
                d = 1
            elif d != -1 and not np.isnan(lo) and close[i] < lo - atr[i] * buf:
                ch_dn[i] = d == 1
                d = -1
        sdir[i] = d
    return sdir, ch_up, ch_dn


# ----------------------------------------------------------------------------
# ZigZag จาก pivot ที่ยืนยันแล้ว (f_zzAdd)
#   P[i, k] = ราคาจุดกลับตัวที่ k ย้อนหลัง (k=0 ล่าสุด) ณ แท่ง i
# ----------------------------------------------------------------------------
@njit(cache=True)
def zigzag(ph, pl, rsi, depth, on):
    n = ph.size
    zp = np.empty(ZZ_CAP + 1)
    zb = np.empty(ZZ_CAP + 1, np.int64)
    zd = np.empty(ZZ_CAP + 1, np.int64)
    zr = np.empty(ZZ_CAP + 1)
    cnt = 0
    P = np.full((n, 7), NAN)
    R = np.full((n, 4), NAN)
    D0 = np.zeros(n, np.int64)
    B0 = np.zeros(n, np.int64)
    for i in range(n):
        if on:
            for k in range(2):
                px = ph[i] if k == 0 else pl[i]
                if np.isnan(px):
                    continue
                d = 1 if k == 0 else -1
                bx = i - depth
                rs = rsi[bx] if bx >= 0 else NAN
                if cnt > 0 and zd[cnt - 1] == d:
                    prev = zp[cnt - 1]
                    if (d == 1 and px > prev) or (d == -1 and px < prev):
                        zp[cnt - 1] = px
                        zb[cnt - 1] = bx
                        zr[cnt - 1] = rs
                else:
                    zp[cnt] = px
                    zb[cnt] = bx
                    zd[cnt] = d
                    zr[cnt] = rs
                    cnt += 1
                    if cnt > ZZ_CAP:
                        for j in range(cnt - 1):
                            zp[j] = zp[j + 1]
                            zb[j] = zb[j + 1]
                            zd[j] = zd[j + 1]
                            zr[j] = zr[j + 1]
                        cnt -= 1
        for k in range(7):
            if cnt > k:
                P[i, k] = zp[cnt - 1 - k]
        for k in range(4):
            if cnt > k:
                R[i, k] = zr[cnt - 1 - k]
        if cnt > 0:
            D0[i] = zd[cnt - 1]
            B0[i] = zb[cnt - 1]
    return P, R, D0, B0


# ---- ตัวตรวจรูปแบบคลื่น (ตรงกับ f_imp* / f_corr* ใน Pine) ----
@njit(cache=True)
def _imp6(p0, p1, p2, p3, p4, p5, s, mn):
    w1 = (p1 - p0) * s
    w2 = (p1 - p2) * s
    w3 = (p3 - p2) * s
    w4 = (p3 - p4) * s
    w5 = (p5 - p4) * s
    r2 = w2 / w1 if w1 > 0 else 9.0
    r4 = w4 / w3 if w3 > 0 else 9.0
    ok_size = (not np.isnan(p0)) and w1 > mn and w3 > mn and w5 > 0
    ok1 = (p2 - p0) * s > 0
    ok2 = (p4 - p1) * s > 0
    ok3 = not (w3 < w1 and w3 < w5)
    ok_new = (p3 - p1) * s > 0 and (p5 - p3) * s > 0
    ok_ret = r2 > 0.15 and r2 < 0.95 and r4 > 0.10 and r4 < 0.85
    return ok_size and ok1 and ok2 and ok3 and ok_new and ok_ret


@njit(cache=True)
def _imp5(p0, p1, p2, p3, p4, s, mn, px):
    w1 = (p1 - p0) * s
    w2 = (p1 - p2) * s
    w3 = (p3 - p2) * s
    w4 = (p3 - p4) * s
    r2 = w2 / w1 if w1 > 0 else 9.0
    r4 = w4 / w3 if w3 > 0 else 9.0
    ok_size = (not np.isnan(p0)) and w1 > mn and w3 > mn
    ok1 = (p2 - p0) * s > 0
    ok2 = (p4 - p1) * s > 0
    ok_new = (p3 - p1) * s > 0
    ok_big3 = w3 >= w1 * 0.9
    ok_ret = r2 > 0.15 and r2 < 0.95 and r4 > 0.10 and r4 < 0.85
    ok_now = (px - p4) * s > mn * 0.15
    return ok_size and ok1 and ok2 and ok_new and ok_big3 and ok_ret and ok_now


@njit(cache=True)
def _imp4(p0, p1, p2, p3, s, mn, px):
    w1 = (p1 - p0) * s
    w2 = (p1 - p2) * s
    w3 = (p3 - p2) * s
    r2 = w2 / w1 if w1 > 0 else 9.0
    ok_size = (not np.isnan(p0)) and w1 > mn and w3 > mn
    ok1 = (p2 - p0) * s > 0
    ok_new = (p3 - p1) * s > 0
    ok_big3 = w3 >= w1 * 0.9
    ok_ret = r2 > 0.15 and r2 < 0.95
    ok_now = (px - p3) * s < -mn * 0.15 and (px - p1) * s > 0
    return ok_size and ok1 and ok_new and ok_big3 and ok_ret and ok_now


@njit(cache=True)
def _imp3(p0, p1, p2, s, mn, px):
    w1 = (p1 - p0) * s
    w2 = (p1 - p2) * s
    r2 = w2 / w1 if w1 > 0 else 9.0
    ok_size = (not np.isnan(p0)) and w1 > mn
    ok1 = (p2 - p0) * s > 0
    ok_ret = r2 > 0.20 and r2 < 0.95
    ok_now = (px - p2) * s > mn * 0.15
    return ok_size and ok1 and ok_ret and ok_now


@njit(cache=True)
def _prior(p_o, q1, q2, q3, s):
    ex = p_o if np.isnan(q1) else q1
    if not np.isnan(q2) and (q2 - ex) * s > 0:
        ex = q2
    if not np.isnan(q3) and (q3 - ex) * s > 0:
        ex = q3
    if np.isnan(p_o) or np.isnan(ex):
        return 0.0
    return (ex - p_o) * s


@njit(cache=True)
def _corr_a(p_top, p_prev, p1x, s, mn, px, prior, need_brk):
    gone = (px - p_top) * s
    ok_size = (not np.isnan(p_top)) and (not np.isnan(p1x)) and prior > mn * 2.0
    ok_top = (p_prev - p_top) * s > 0 and (p1x - p_top) * s > 0
    ok_go = gone > mn * 0.5
    ok_brk = (not need_brk) or (px - p1x) * s > 0
    ok_keep = gone < prior * 0.95
    return ok_size and ok_top and ok_go and ok_brk and ok_keep


@njit(cache=True)
def _corr_c(p_o, p_a, p_b, s, mn, px, prior):
    wa = (p_a - p_o) * s
    wb = (p_a - p_b) * s
    rb = wb / wa if wa > 0 else 9.0
    ok_size = (not np.isnan(p_o)) and wa > mn and prior > mn * 1.5
    ok_b = (p_b - p_o) * s > 0
    ok_ret = rb > 0.20 and rb < 1.05
    ok_big = prior > wa * 1.2
    ok_now = (px - p_b) * s > mn * 0.15
    ok_keep = (px - p_o) * s < prior
    return ok_size and ok_b and ok_ret and ok_big and ok_now and ok_keep


@njit(cache=True)
def _corr_e(p_o, p_a, p_b, p_c, s, mn, px, prior):
    wa = (p_a - p_o) * s
    wb = (p_a - p_b) * s
    wc = (p_c - p_b) * s
    rb = wb / wa if wa > 0 else 9.0
    rc = wc / wa if wa > 0 else 9.0
    ok_size = (not np.isnan(p_o)) and wa > mn and prior > mn * 1.5
    ok_b = (p_b - p_o) * s > 0
    ok_rb = rb > 0.20 and rb < 1.05
    ok_rc = rc > 0.50 and rc < 2.20
    ok_big = prior > wa * 1.2
    ok_deep = (p_c - p_a) * s > -mn * 0.30
    ok_keep = (p_c - p_o) * s < prior * 1.05
    ok_turn = (p_c - px) * s > mn * 0.20
    return ok_size and ok_b and ok_rb and ok_rc and ok_big and ok_deep and ok_keep and ok_turn


@njit(cache=True)
def elliott(P, D0, close, atr, math_dir, on, corr_on, corr_first, min_atr, min_conf):
    """คืน ewBias (1 / -1 / 0) ต่อแท่ง — ทิศที่โครงสร้างคลื่นหนุน (ผ่านเกณฑ์ความชัดแล้ว)"""
    n = close.size
    bias = np.zeros(n, np.int64)
    if not on:
        return bias
    for i in range(n):
        e0, e1, e2, e3, e4, e5, e6 = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        px = close[i]
        mn = atr[i] * min_atr
        sa = D0[i]
        sb = -sa
        ok6 = _imp6(e5, e4, e3, e2, e1, e0, sa, mn)
        ok5 = (not ok6) and _imp5(e4, e3, e2, e1, e0, sb, mn, px)
        pri_ce = _prior(e3, e4, e5, e6, sa)
        pri_cc = _prior(e2, e3, e4, e5, sb)
        pri_ca = _prior(e0, e1, e2, e3, sb)
        imp4_raw = (not ok6) and (not ok5) and _imp4(e3, e2, e1, e0, sa, mn, px)
        gate = corr_on and (not ok6) and (not ok5) and (corr_first or not imp4_raw)
        ok_ce = gate and _corr_e(e3, e2, e1, e0, sa, mn, px, pri_ce)
        ok_cc = gate and (not ok_ce) and _corr_c(e2, e1, e0, sb, mn, px, pri_cc)
        ok_ca = gate and (not ok_ce) and (not ok_cc) and _corr_a(e0, e1, e2, sb, mn, px, pri_ca, not corr_first)
        ok4 = imp4_raw and not ok_ce and not ok_cc and not ok_ca
        ok3 = (not ok6) and (not ok5) and (not ok4) and (not ok_ce) and (not ok_cc) and (not ok_ca) \
            and _imp3(e2, e1, e0, sb, mn, px)

        pts = 0
        raw = 0
        braw = 0
        size = 0.0
        if ok6:
            pts, raw, braw, size = 6, 80, -sa, (e0 - e5) * sa
        elif ok5:
            pts, raw, braw, size = 5, 72, sb, (e1 - e2) * sb
        elif ok4:
            pts, raw, braw, size = 4, 70, sa, (e0 - e1) * sa
        elif ok3:
            pts, raw, braw, size = 3, 68, sb, (e1 - e2) * sb
        elif ok_ce:
            pts, raw, braw, size = 4, 74, -sa, (e0 - e3) * sa
        elif ok_cc:
            wac = (e1 - e2) * sb
            dnc = ((px - e0) * sb) / wac if wac > 0 else 0.0
            pts, raw, braw, size = 3, 66, (0 if dnc >= 0.85 else sb), wac
        elif ok_ca:
            pts, raw, braw, size = 1, 55, sb, pri_ca
        if pts == 0:
            continue
        big = atr[i] > 0 and size > atr[i] * 2.5
        conf = min(100, raw + (10 if big else 0) + (10 if braw == math_dir[i] else 0))
        if conf >= min_conf:
            bias[i] = braw
    return bias


@njit(cache=True)
def fib_div(P, R, D0, B0, close, atr, min_atr, depth, fib_on, div_on, div_min_rsi):
    """โซนทอง Fib 0.5-0.618 ของขาล่าสุด + RSI divergence ที่จุดกลับตัว"""
    n = close.size
    g_long = np.zeros(n, np.bool_)
    g_short = np.zeros(n, np.bool_)
    bull = np.zeros(n, np.bool_)
    bear = np.zeros(n, np.bool_)
    for i in range(n):
        e0, e1, e2, e3 = P[i, 0], P[i, 1], P[i, 2], P[i, 3]
        px = close[i]
        if fib_on and not np.isnan(e0) and not np.isnan(e1):
            leg_up = e0 > e1
            hi = max(e0, e1)
            lo = min(e0, e1)
            rng = hi - lo
            if rng > atr[i] * min_atr:
                f5 = hi - 0.5 * rng if leg_up else lo + 0.5 * rng
                f6 = hi - 0.618 * rng if leg_up else lo + 0.618 * rng
                top = max(f5, f6)
                bot = min(f5, f6)
                if px >= bot and px <= top:
                    g_long[i] = leg_up
                    g_short[i] = not leg_up
        if div_on:
            d0 = D0[i]
            fresh = i - B0[i] <= depth * 3
            hi_new = e0 if d0 == 1 else e1
            hi_old = e2 if d0 == 1 else e3
            r_hi_new = R[i, 0] if d0 == 1 else R[i, 1]
            r_hi_old = R[i, 2] if d0 == 1 else R[i, 3]
            lo_new = e0 if d0 == -1 else e1
            lo_old = e2 if d0 == -1 else e3
            r_lo_new = R[i, 0] if d0 == -1 else R[i, 1]
            r_lo_old = R[i, 2] if d0 == -1 else R[i, 3]
            bear[i] = fresh and not np.isnan(hi_old) and not np.isnan(r_hi_old) \
                and hi_new > hi_old and r_hi_new < r_hi_old - div_min_rsi
            bull[i] = fresh and not np.isnan(lo_old) and not np.isnan(r_lo_old) \
                and lo_new < lo_old and r_lo_new > r_lo_old + div_min_rsi
    return g_long, g_short, bull, bear


# ----------------------------------------------------------------------------
# Volume Profile (POC / Value Area 70%) — คำนวณใหม่ทุก step แท่ง
# ----------------------------------------------------------------------------
@njit(cache=True)
def volume_profile(h, l, c, v, length, bins, step):
    n = c.size
    poc = np.full(n, NAN)
    vah = np.full(n, NAN)
    val = np.full(n, NAN)
    arr = np.zeros(bins)
    cur_poc = NAN
    cur_vah = NAN
    cur_val = NAN
    for i in range(n):
        if i >= 20 and i % step == 0:
            m = min(length, i + 1)
            hi = -np.inf
            lo = np.inf
            vol_sum = 0.0
            for k in range(m):
                j = i - k
                hi = max(hi, h[j])
                lo = min(lo, l[j])
                if not np.isnan(v[j]):
                    vol_sum += v[j]
            if hi > lo:
                use_vol = vol_sum > 0
                arr[:] = 0.0
                stp = (hi - lo) / bins
                for k in range(m):
                    j = i - k
                    px = (h[j] + l[j] + c[j]) / 3.0
                    vv = (0.0 if np.isnan(v[j]) else v[j]) if use_vol else 1.0
                    bi = int(np.floor((px - lo) / stp))
                    bi = max(0, min(bins - 1, bi))
                    arr[bi] += vv
                pi = int(np.argmax(arr))
                total = arr.sum()
                acc = arr[pi]
                lo_i = pi
                hi_i = pi
                guard = 0
                while acc < total * 0.7 and guard < bins * 2 and (lo_i > 0 or hi_i < bins - 1):
                    vl = arr[lo_i - 1] if lo_i > 0 else -1.0
                    vh = arr[hi_i + 1] if hi_i < bins - 1 else -1.0
                    if vh >= vl:
                        hi_i += 1
                        acc += vh
                    else:
                        lo_i -= 1
                        acc += vl
                    guard += 1
                cur_poc = lo + (pi + 0.5) * stp
                cur_vah = lo + (hi_i + 1) * stp
                cur_val = lo + lo_i * stp
        poc[i] = cur_poc
        vah[i] = cur_vah
        val[i] = cur_val
    return poc, vah, val
