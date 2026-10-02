"""อินดิเคเตอร์พื้นฐาน — เขียนตามนิยามของ ta.* ใน Pine v5 แล้วคอมไพล์ด้วย Numba

ค่า na ของ Pine = NaN, ส่วน bool ที่เทียบกับ na ได้ False เหมือน Pine
"""

from __future__ import annotations

import numpy as np
from numba import njit

NAN = np.nan


@njit(cache=True)
def sma(x, n):
    out = np.full(x.size, NAN)
    s = 0.0
    nan_cnt = 0
    for i in range(x.size):
        v = x[i]
        if np.isnan(v):
            nan_cnt += 1
        else:
            s += v
        if i >= n:
            old = x[i - n]
            if np.isnan(old):
                nan_cnt -= 1
            else:
                s -= old
        if i >= n - 1 and nan_cnt == 0:
            out[i] = s / n
    return out


@njit(cache=True)
def ema(x, n):
    """ta.ema : เริ่มจากค่าแรกที่ไม่ใช่ na"""
    out = np.full(x.size, NAN)
    a = 2.0 / (n + 1)
    prev = NAN
    for i in range(x.size):
        v = x[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = v if np.isnan(prev) else a * v + (1 - a) * prev
        out[i] = prev
    return out


@njit(cache=True)
def rma(x, n):
    """ta.rma : เริ่มด้วย SMA ของ n ค่าแรก แล้วค่อย smooth ด้วย alpha = 1/n"""
    out = np.full(x.size, NAN)
    prev = NAN
    s = 0.0
    c = 0
    for i in range(x.size):
        v = x[i]
        if np.isnan(prev):
            if np.isnan(v):
                s = 0.0
                c = 0
                continue
            s += v
            c += 1
            if c == n:
                prev = s / n
                out[i] = prev
        else:
            if not np.isnan(v):
                prev = (v + (n - 1) * prev) / n
            out[i] = prev
    return out


@njit(cache=True)
def stdev(x, n):
    """ta.stdev (biased = population)"""
    out = np.full(x.size, NAN)
    for i in range(n - 1, x.size):
        m = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            if np.isnan(x[j]):
                ok = False
                break
            m += x[j]
        if not ok:
            continue
        m /= n
        ss = 0.0
        for j in range(i - n + 1, i + 1):
            d = x[j] - m
            ss += d * d
        out[i] = np.sqrt(ss / n)
    return out


@njit(cache=True)
def true_range(h, l, c, handle_na):
    out = np.empty(c.size)
    out[0] = h[0] - l[0] if handle_na else NAN
    for i in range(1, c.size):
        out[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return out


@njit(cache=True)
def atr(h, l, c, n):
    return rma(true_range(h, l, c, True), n)


@njit(cache=True)
def rsi(c, n):
    up = np.full(c.size, NAN)
    dn = np.full(c.size, NAN)
    for i in range(1, c.size):
        ch = c[i] - c[i - 1]
        up[i] = max(ch, 0.0)
        dn[i] = max(-ch, 0.0)
    ru = rma(up, n)
    rd = rma(dn, n)
    out = np.full(c.size, NAN)
    for i in range(c.size):
        if np.isnan(ru[i]) or np.isnan(rd[i]):
            continue
        if rd[i] == 0:
            out[i] = 100.0
        elif ru[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ru[i] / rd[i])
    return out


@njit(cache=True)
def dmi(h, l, c, n, smooth):
    size = c.size
    pdm = np.full(size, NAN)
    mdm = np.full(size, NAN)
    for i in range(1, size):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        mdm[i] = down if (down > up and down > 0) else 0.0
    trur = rma(true_range(h, l, c, False), n)
    rp = rma(pdm, n)
    rm = rma(mdm, n)
    plus = np.full(size, NAN)
    minus = np.full(size, NAN)
    dx = np.full(size, NAN)
    for i in range(size):
        if np.isnan(trur[i]) or trur[i] == 0 or np.isnan(rp[i]) or np.isnan(rm[i]):
            continue
        plus[i] = 100.0 * rp[i] / trur[i]
        minus[i] = 100.0 * rm[i] / trur[i]
        s = plus[i] + minus[i]
        dx[i] = abs(plus[i] - minus[i]) / (s if s != 0 else 1.0)
    adx = rma(dx, smooth)
    for i in range(size):
        adx[i] *= 100.0
    return plus, minus, adx


@njit(cache=True)
def macd(c, fast, slow, sig):
    line = ema(c, fast) - ema(c, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


@njit(cache=True)
def crossover(a, b):
    out = np.zeros(a.size, np.bool_)
    for i in range(1, a.size):
        out[i] = a[i] > b[i] and a[i - 1] <= b[i - 1]
    return out


@njit(cache=True)
def barssince(cond):
    out = np.full(cond.size, NAN)
    last = -1
    for i in range(cond.size):
        if cond[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


@njit(cache=True)
def pivot_high(src, left, right):
    """ta.pivothigh : ค่าออกที่แท่งยืนยัน (ช้ากว่าจุดจริง right แท่ง)"""
    out = np.full(src.size, NAN)
    for i in range(left + right, src.size):
        p = i - right
        v = src[p]
        ok = True
        for j in range(p - left, p):
            if src[j] >= v:
                ok = False
                break
        if ok:
            for j in range(p + 1, i + 1):
                if src[j] > v:
                    ok = False
                    break
        if ok:
            out[i] = v
    return out


@njit(cache=True)
def pivot_low(src, left, right):
    out = np.full(src.size, NAN)
    for i in range(left + right, src.size):
        p = i - right
        v = src[p]
        ok = True
        for j in range(p - left, p):
            if src[j] <= v:
                ok = False
                break
        if ok:
            for j in range(p + 1, i + 1):
                if src[j] < v:
                    ok = False
                    break
        if ok:
            out[i] = v
    return out


@njit(cache=True)
def linreg_stats(c, n):
    """คืน (ความชันต่อแท่ง, correlation กับเวลา) ของหน้าต่าง n แท่ง
    ความชัน = ta.linreg(c,n,0) - ta.linreg(c,n,1),  corr = ta.correlation(c, bar_index, n)"""
    size = c.size
    slope = np.full(size, NAN)
    corr = np.full(size, NAN)
    xm = (n - 1) / 2.0
    sxx = 0.0
    for k in range(n):
        sxx += (k - xm) ** 2
    for i in range(n - 1, size):
        ym = 0.0
        for k in range(n):
            ym += c[i - n + 1 + k]
        ym /= n
        sxy = 0.0
        syy = 0.0
        for k in range(n):
            dy = c[i - n + 1 + k] - ym
            sxy += (k - xm) * dy
            syy += dy * dy
        slope[i] = sxy / sxx
        if syy > 0:
            corr[i] = sxy / np.sqrt(sxx * syy)
    return slope, corr


@njit(cache=True)
def holt(c, alpha, beta):
    lvl = np.empty(c.size)
    trd = np.empty(c.size)
    pl_ = NAN
    pt = 0.0
    for i in range(c.size):
        prev_l = c[i] if np.isnan(pl_) else pl_
        cur_l = alpha * c[i] + (1 - alpha) * (prev_l + pt)
        cur_t = beta * (cur_l - prev_l) + (1 - beta) * pt
        lvl[i] = cur_l
        trd[i] = cur_t
        pl_ = cur_l
        pt = cur_t
    return lvl, trd


@njit(cache=True)
def efficiency_ratio(c, n):
    out = np.zeros(c.size)
    for i in range(n, c.size):
        vol = 0.0
        for j in range(i - n + 1, i + 1):
            vol += abs(c[j] - c[j - 1])
        out[i] = abs(c[i] - c[i - n]) / vol if vol > 0 else 0.0
    return out


@njit(cache=True)
def kama(c, er):
    fast = 2.0 / 3.0
    slow = 2.0 / 31.0
    out = np.empty(c.size)
    prev = NAN
    for i in range(c.size):
        sc = (er[i] * (fast - slow) + slow) ** 2
        prev = c[i] if np.isnan(prev) else prev + sc * (c[i] - prev)
        out[i] = prev
    return out


@njit(cache=True)
def highest_prev(x, n):
    """สูงสุดของ n แท่งก่อนหน้า (ไม่รวมแท่งปัจจุบัน) = ta.highest(x, n)[1]"""
    out = np.full(x.size, NAN)
    for i in range(n, x.size):
        m = x[i - n]
        for j in range(i - n + 1, i):
            if x[j] > m:
                m = x[j]
        out[i] = m
    return out


@njit(cache=True)
def lowest_prev(x, n):
    out = np.full(x.size, NAN)
    for i in range(n, x.size):
        m = x[i - n]
        for j in range(i - n + 1, i):
            if x[j] < m:
                m = x[j]
        out[i] = m
    return out


@njit(cache=True)
def stochastic(h, l, c, k_len, k_smooth, d_len):
    """Full stochastic : %K ดิบ -> SMA k_smooth = %K, SMA d_len ของ %K = %D"""
    n = c.size
    raw = np.full(n, NAN)
    for i in range(k_len - 1, n):
        hh = h[i - k_len + 1]
        ll = l[i - k_len + 1]
        for j in range(i - k_len + 2, i + 1):
            hh = max(hh, h[j])
            ll = min(ll, l[j])
        raw[i] = 100.0 * (c[i] - ll) / (hh - ll) if hh > ll else 50.0
    k = sma(raw, k_smooth)
    return k, sma(k, d_len)


@njit(cache=True)
def obv(c, v):
    out = np.zeros(c.size)
    for i in range(1, c.size):
        vv = 0.0 if np.isnan(v[i]) else v[i]
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + vv
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - vv
        else:
            out[i] = out[i - 1]
    return out


@njit(cache=True)
def psar_next(h, l, start, step, max_af):
    """Parabolic SAR (Wilder) — คืนค่า SAR ที่จะใช้กับ 'แท่งถัดไป' (คำนวณจากข้อมูลถึงแท่งปัจจุบัน) และโหมดขาขึ้น/ลง"""
    n = h.size
    nxt = np.full(n, NAN)
    bull = np.zeros(n, np.bool_)
    if n < 3:
        return nxt, bull
    up = True
    sar = l[0]
    ep = h[0]
    af = start
    for i in range(1, n):
        s = sar + af * (ep - sar)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up = False
                s = ep
                ep = l[i]
                af = start
            elif h[i] > ep:
                ep = h[i]
                af = min(af + step, max_af)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up = True
                s = ep
                ep = h[i]
                af = start
            elif l[i] < ep:
                ep = l[i]
                af = min(af + step, max_af)
        sar = s
        # ค่าสำหรับแท่ง i+1 : ห้ามเกินกรอบของแท่ง i และ i-1
        q = sar + af * (ep - sar)
        if up:
            q = min(q, l[i], l[i - 1])
        else:
            q = max(q, h[i], h[i - 1])
        nxt[i] = q
        bull[i] = up
    return nxt, bull
