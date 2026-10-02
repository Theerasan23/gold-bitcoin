"""รวมอินดิเคเตอร์ทั้งหมดเป็นคะแนน 100 และสัญญาณเข้า (ตรงกับส่วน 'ระบบให้คะแนน' ใน Pine)"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import indicators as ta
from . import structure as st
from .config import Params, htf_resolution

# น้ำหนักคะแนน (รวม 100)
W_HTF, W_MATH, W_WAVE, W_ADX, W_MACD, W_RSI, W_VOL, W_SR, W_CONF, W_EXT = 22, 18, 10, 9, 8, 8, 8, 8, 5, 4


def _polars_every(interval: str) -> str:
    return "1mo" if interval == "1M" else interval


def htf_trend(times: pl.Series, close: np.ndarray, htf: str, fast: int, slow: int):
    """EMA ของ TF ใหญ่จากแท่ง 'ที่ปิดแล้ว' เท่านั้น (= request.security(..., x[1], lookahead_on))"""
    keys = times.dt.truncate(_polars_every(htf)).to_numpy()
    uniq, first_idx = np.unique(keys, return_index=True)
    # ราคาปิดของแต่ละแท่ง TF ใหญ่ = close ของแท่งสุดท้ายในช่วงนั้น
    last_idx = np.r_[first_idx[1:] - 1, len(keys) - 1]
    hc = close[last_idx]
    ef = ta.ema(hc, fast)
    es = ta.ema(hc, slow)
    pos = np.searchsorted(uniq, keys) - 1  # แท่ง TF ใหญ่ก่อนหน้า
    valid = pos >= 0
    p = np.where(valid, pos, 0)
    h_close = np.where(valid, hc[p], np.nan)
    h_fast = np.where(valid, ef[p], np.nan)
    h_slow = np.where(valid, es[p], np.nan)
    up = (h_close > h_slow) & (h_fast > h_slow)
    down = (h_close < h_slow) & (h_fast < h_slow)
    return up, down, h_slow


def _to_ms(day: str | None) -> float:
    if day is None:
        return np.inf
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000


def compute(df: pl.DataFrame, interval: str, p: Params) -> dict[str, np.ndarray]:
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    v = df["volume"].to_numpy().astype(np.float64)
    t_ms = df["open_time"].dt.epoch("ms").to_numpy()

    atr = ta.atr(h, l, c, p.atr_len)
    macd, sig, hist = ta.macd(c, p.macd_fast, p.macd_slow, p.macd_signal)
    rsi = ta.rsi(c, p.rsi_len)
    di_p, di_m, adx = ta.dmi(h, l, c, p.adx_len, p.adx_smooth)
    vol_ma = ta.sma(v, p.vol_len)

    # ---- คณิตศาสตร์วิเคราะห์แนวโน้ม ----
    _, holt_trd = ta.holt(c, p.holt_alpha, p.holt_beta)
    er = ta.efficiency_ratio(c, p.er_len)
    kama = ta.kama(c, er)
    slope, corr = ta.linreg_stats(c, p.reg_len)
    corr = np.nan_to_num(corr)
    r2 = corr ** 2
    sd = ta.stdev(c, p.z_len)
    mean = ta.sma(c, p.z_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (c - mean) / sd, 0.0)
        slope_atr = np.where(atr > 0, slope / atr, 0.0)
    slope_atr = np.nan_to_num(slope_atr)
    dir_sum = np.sign(holt_trd) + np.sign(np.nan_to_num(slope)) + np.sign(c - kama)
    math_dir = np.where(np.abs(dir_sum) >= 2, np.sign(dir_sum), 0).astype(np.int64)
    str_comp = np.minimum(np.abs(slope_atr) / p.slope_ref, 1.0)
    math_conf = np.round(100 * (0.35 * er + 0.35 * r2 + 0.30 * str_comp) * (np.abs(dir_sum) / 3.0))

    # ---- TF ใหญ่ ----
    htf = htf_resolution(interval, p.htf)
    htf_up, htf_down, htf_slow = htf_trend(df["open_time"], c, htf, p.htf_fast, p.htf_slow)

    # ---- แนวรับแนวต้าน ----
    pv_h = ta.pivot_high(h, p.piv_left, p.piv_right)
    pv_l = ta.pivot_low(l, p.piv_left, p.piv_right)
    near_res, near_sup = st.sr_nearest(pv_h, pv_l, atr, c, p.sr_merge_atr, p.sr_max)
    with np.errstate(invalid="ignore", divide="ignore"):
        res_room = np.where(np.isnan(near_res), 99.0, (near_res - c) / atr)
        sup_room = np.where(np.isnan(near_sup), 99.0, (c - near_sup) / atr)

    # ---- โครงสร้างราคา ----
    rv_h = ta.pivot_high(h, p.rev_len, p.rev_len)
    rv_l = ta.pivot_low(l, p.rev_len, p.rev_len)
    struct_dir, choch_up, choch_dn = st.market_structure(rv_h, rv_l, c, atr, p.rev_buf)

    # ---- Elliott / Fib / Divergence ----
    ew_h = ta.pivot_high(h, p.ew_depth, p.ew_depth)
    ew_l = ta.pivot_low(l, p.ew_depth, p.ew_depth)
    P, R, D0, B0 = st.zigzag(ew_h, ew_l, rsi, p.ew_depth, p.ew_on)
    ew_bias = st.elliott(P, D0, c, atr, math_dir, p.ew_on, p.ew_corr_on, p.ew_corr_first,
                         p.ew_min_atr, p.ew_min_conf)
    g_long, g_short, bull_div, bear_div = st.fib_div(P, R, D0, B0, c, atr, p.ew_min_atr, p.ew_depth,
                                                      p.fib_on, p.div_on, p.div_min_rsi)

    # ---- Volume Profile ----
    if p.vp_on:
        poc, vah, val = st.volume_profile(h, l, c, v, p.vp_len, p.vp_bins, p.vp_step)
        near_poc = np.abs(c - poc) <= atr * 0.5
        near_val = np.abs(c - val) <= atr * 0.75
        near_vah = np.abs(c - vah) <= atr * 0.75
    else:
        poc = np.full(c.size, np.nan)
        near_poc = near_val = near_vah = np.zeros(c.size, bool)

    conf_l = g_long.astype(int) + bull_div.astype(int) + (near_val | near_poc).astype(int)
    conf_s = g_short.astype(int) + bear_div.astype(int) + (near_vah | near_poc).astype(int)

    # ---- คะแนน ----
    hist_prev = np.r_[np.nan, hist[:-1]]
    has_vol = v > 0
    vol_ok = np.where(has_vol, v > vol_ma * p.vol_mult, True)

    f_l_htf = htf_up if p.use_htf else np.ones(c.size, bool)
    f_s_htf = htf_down if p.use_htf else np.ones(c.size, bool)
    f_l_math = (math_dir == 1) & (math_conf >= p.conf_min)
    f_s_math = (math_dir == -1) & (math_conf >= p.conf_min)
    f_l_ext = z <= p.z_max
    f_s_ext = z >= -p.z_max
    f_l_macd = (macd > sig) & (hist > hist_prev)
    f_s_macd = (macd < sig) & (hist < hist_prev)
    f_l_rsi = (rsi >= p.rsi_long_min) & (rsi <= p.rsi_long_max)
    f_s_rsi = (rsi <= p.rsi_short_max) & (rsi >= p.rsi_short_min)
    f_l_adx = (adx >= p.adx_min) & (di_p > di_m)
    f_s_adx = (adx >= p.adx_min) & (di_m > di_p)
    f_l_sr = (res_room >= p.sr_min_atr) if p.use_sr_filter else np.ones(c.size, bool)
    f_s_sr = (sup_room >= p.sr_min_atr) if p.use_sr_filter else np.ones(c.size, bool)
    wave_l = np.where(ew_bias == 1, W_WAVE, np.where(ew_bias == 0, 5, 0))
    wave_s = np.where(ew_bias == -1, W_WAVE, np.where(ew_bias == 0, 5, 0))
    conf_pts_l = np.where(conf_l >= 2, W_CONF, np.where(conf_l == 1, 3, 0))
    conf_pts_s = np.where(conf_s >= 2, W_CONF, np.where(conf_s == 1, 3, 0))

    long_score = (W_HTF * f_l_htf + W_MATH * f_l_math + wave_l + W_MACD * f_l_macd + W_RSI * f_l_rsi
                  + W_ADX * f_l_adx + W_VOL * vol_ok + W_SR * f_l_sr + conf_pts_l + W_EXT * f_l_ext)
    short_score = (W_HTF * f_s_htf + W_MATH * f_s_math + wave_s + W_MACD * f_s_macd + W_RSI * f_s_rsi
                   + W_ADX * f_s_adx + W_VOL * vol_ok + W_SR * f_s_sr + conf_pts_s + W_EXT * f_s_ext)

    # ---- เงื่อนไขเข้า (ส่วนที่ไม่ขึ้นกับสถานะไม้ — cooldown / position อยู่ใน backtest) ----
    cross_up = np.nan_to_num(ta.barssince(ta.crossover(macd, sig)), nan=9999) <= p.cross_lookback
    cross_dn = np.nan_to_num(ta.barssince(ta.crossover(sig, macd)), nan=9999) <= p.cross_lookback
    in_range = (t_ms >= _to_ms(p.start)) & (t_ms <= _to_ms(p.end))
    ew_long_ok = (ew_bias != -1) if p.ew_filter else np.ones(c.size, bool)
    ew_short_ok = (ew_bias != 1) if p.ew_filter else np.ones(c.size, bool)
    struct_long_ok = (struct_dir != -1) if p.rev_filter else np.ones(c.size, bool)
    struct_short_ok = (struct_dir != 1) if p.rev_filter else np.ones(c.size, bool)

    long_sig = (p.allow_long & in_range & cross_up & (long_score >= p.min_score) & ew_long_ok & struct_long_ok)
    short_sig = (p.allow_short & in_range & cross_dn & (short_score >= p.min_score) & ew_short_ok & struct_short_ok)

    return {
        "time_ms": t_ms, "open": o, "high": h, "low": l, "close": c, "volume": v,
        "atr": atr, "rsi": rsi, "adx": adx, "z": z, "kama": kama,
        "math_dir": math_dir, "math_conf": math_conf,
        "htf_up": htf_up, "htf_down": htf_down, "htf_ema_slow": htf_slow,
        "near_res": near_res, "near_sup": near_sup,
        "struct_dir": struct_dir, "choch_up": choch_up, "choch_dn": choch_dn,
        "ew_bias": ew_bias, "poc": poc,
        "long_score": long_score.astype(np.int64), "short_score": short_score.astype(np.int64),
        "long_sig": long_sig, "short_sig": short_sig,
    }
