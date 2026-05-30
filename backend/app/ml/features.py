from scipy.signal import argrelextrema
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from core.config import TrainingConfig

def features(spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close, trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s):
    config = TrainingConfig()
    np.random.seed(config.SEED)
    idx = spy_close.index

    def align(s, idx):
        return s.reindex(idx).ffill().bfill()

    # ── Helper: ADX (Average Directional Index) ───────────────────────────────────
    def compute_adx(high, low, close, period=14):
        """
        Causal ADX — quantifies trend strength regardless of direction.
        Returns: adx, plus_di, minus_di as pd.Series aligned to close.index
        """
        h = high.reindex(close.index).ffill()
        l = low.reindex(close.index).ffill()
        c = close

        tr   = pd.concat([h - l,
                        (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs()], axis=1).max(axis=1)
        dm_p = (h - h.shift(1)).clip(lower=0).where((h - h.shift(1)) > (l.shift(1) - l), 0)
        dm_m = (l.shift(1) - l).clip(lower=0).where((l.shift(1) - l) > (h - h.shift(1)), 0)

        atr   = tr.ewm(span=period, adjust=False).mean()
        di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
        adx   = dx.ewm(span=period, adjust=False).mean()
        return adx, di_p, di_m

    # ── Helper: swing high/low proximity ─────────────────────────────────────────
    def swing_proximity(series, order=10, window=252):
        """
        Returns distance from recent swing high and swing low as % of price.
        Uses argrelextrema on a rolling basis (causal approximation).
        """
        arr = series.values
        n   = len(arr)
        dist_high = np.full(n, np.nan)
        dist_low  = np.full(n, np.nan)
        for i in range(window, n):
            seg = arr[max(0, i - window): i]
            local_max = argrelextrema(seg, np.greater, order=order)[0]
            local_min = argrelextrema(seg, np.less,    order=order)[0]
            if len(local_max):
                dist_high[i] = (arr[i] - seg[local_max[-1]]) / (seg[local_max[-1]] + 1e-9)
            if len(local_min):
                dist_low[i]  = (arr[i] - seg[local_min[-1]]) / (seg[local_min[-1]] + 1e-9)
        return (pd.Series(dist_high, index=series.index),
                pd.Series(dist_low,  index=series.index))

    # ── Build master feature DataFrame ────────────────────────────────────────────
    feat = pd.DataFrame(index=idx)

    p   = spy_close
    vol = spy_vol.reindex(idx).ffill()
    tf  = trend_fast.reindex(idx)
    ts  = trend_slow.reindex(idx)
    cy  = cycle_comp.reindex(idx)
    ns  = noise_comp.reindex(idx)

    # ── [2a] Trend structure features ─────────────────────────────────────────────
    feat["trend_fast_slope_5"]  = tf.diff(5)  / (tf.shift(5)  + 1e-9)
    feat["trend_fast_slope_20"] = tf.diff(20) / (tf.shift(20) + 1e-9)
    feat["trend_slow_slope_20"] = ts.diff(20) / (ts.shift(20) + 1e-9)
    feat["trend_slow_slope_60"] = ts.diff(60) / (ts.shift(60) + 1e-9)
    feat["trend_accel"]         = tf.diff().diff()   # 2nd derivative of fast trend
    feat["trend_accel_slow"]    = ts.diff().diff()

    # Fast vs slow trend relationship (inflection signal)
    feat["trend_fast_vs_slow"]  = (tf - ts) / (ts + 1e-9)   # + = fast above slow = bullish
    feat["trend_cross_signal"]  = np.sign(tf - ts)           # direction of fast-slow spread
    feat["trend_cross_change"]  = feat["trend_cross_signal"].diff()  # when this ≠ 0 = crossover

    feat["price_vs_fast"]       = (p - tf) / (tf + 1e-9)
    feat["price_vs_slow"]       = (p - ts) / (ts + 1e-9)

    # ── [5] Cycle component features ──────────────────────────────────────────────
    feat["cycle_level"]         = cy / (p + 1e-9)   # cycle as % of price
    feat["cycle_slope"]         = cy.diff(5)
    feat["cycle_zscore"]        = (cy - cy.rolling(60).mean()) / (cy.rolling(60).std() + 1e-9)
    feat["noise_abs_20"]        = ns.abs().rolling(20).mean() / (p + 1e-9)  # relative noise

    # ── [2b] ADX — trend strength ─────────────────────────────────────────────────
    adx, di_p, di_m = compute_adx(spy_high, spy_low, p)
    di_bull = (di_p.reindex(idx).ffill() > di_m.reindex(idx).ffill()).astype(int)
    feat["adx_14"]              = adx
    feat["adx_zscore_60"]       = (adx - adx.rolling(60).mean()) / (adx.rolling(60).std() + 1e-9)
    feat["di_diff"]             = di_p - di_m          # + = bullish directional pressure
    feat["adx_trend_strength"]  = adx / 25.0           # normalised (ADX>25 = trending)

    # ── [2c] MA stack — multi-timeframe alignment ──────────────────────────────────
    ma5   = p.rolling(5).mean()
    ma20  = p.rolling(20).mean()
    ma50  = p.rolling(50).mean()
    ma200 = p.rolling(200).mean()

    feat["ma_cross_5_20"]       = ma5  / (ma20  + 1e-9) - 1
    feat["ma_cross_20_50"]      = ma20 / (ma50  + 1e-9) - 1
    feat["ma_cross_50_200"]     = ma50 / (ma200 + 1e-9) - 1
    feat["price_vs_200d"]       = (p - ma200) / (ma200 + 1e-9)  # key trend filter
    feat["ma_stack_score"]      = (                              # +1 per aligned condition
        (ma5 > ma20).astype(int) +
        (ma20 > ma50).astype(int) +
        (ma50 > ma200).astype(int)
    ).astype(float) / 3.0   # 0=fully bearish stack, 1=fully bullish stack

    # ── [2d] Support/Resistance proximity ─────────────────────────────────────────
    dist_swing_high, dist_swing_low = swing_proximity(p, order=10, window=252)
    feat["dist_swing_high"]     = dist_swing_high
    feat["dist_swing_low"]      = dist_swing_low
    feat["dist_52w_high"]       = (p - p.rolling(252).max()) / (p.rolling(252).max() + 1e-9)
    feat["dist_52w_low"]        = (p - p.rolling(252).min()) / (p.rolling(252).min() + 1e-9)

    # Bollinger band position (dynamic S/R)
    bb_mid = p.rolling(20).mean()
    bb_std = p.rolling(20).std()
    feat["bb_position"]         = (p - bb_mid) / (2 * bb_std + 1e-9)
    feat["bb_width"]            = (2 * bb_std) / (bb_mid + 1e-9)  # volatility of range

    # ── [2e] Standard price/vol features (kept from v1) ───────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f"spy_ret_{w}d"]  = p.pct_change(w)
        feat[f"spy_vol_{w}d"]  = p.pct_change().rolling(w).std()

    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"]              = 100 - (100 / (1 + gain / (loss + 1e-9)))
    feat["vol_ratio_20"]        = vol / (vol.rolling(20).mean() + 1e-9)

    # ── [6a] Cross-asset features ─────────────────────────────────────────────────
    qqq = align(qqq_close, idx)
    tlt = align(tlt_close, idx)

    feat["qqq_ret_20d"]         = qqq.pct_change(20)
    feat["tlt_ret_20d"]         = tlt.pct_change(20)
    feat["spy_qqq_spread"]      = p.pct_change(20) - qqq.pct_change(20)
    feat["spy_tlt_spread"]      = p.pct_change(20) - tlt.pct_change(20)

    # ── [6b] Trend concordance — are all indices trending together? ───────────────
    def ma_trend_signal(s, fast=20, slow=60):
        """+1 if fast MA > slow MA (uptrend), -1 otherwise."""
        return np.sign(s.rolling(fast).mean() - s.rolling(slow).mean())

    tc_spy = ma_trend_signal(p)
    tc_qqq = ma_trend_signal(qqq)
    tc_tlt = ma_trend_signal(tlt)

    feat["trend_concordance"]   = (tc_spy + tc_qqq) / 2.0  # -1 to +1
    feat["equity_bond_diverge"] = tc_spy - tc_tlt   # equity and bond trend divergence

    # ── [6c] Risk-on/off composite ────────────────────────────────────────────────
    vix   = align(vix_s,  idx)
    yr10  = align(yr10_s, idx)
    yr2   = align(yr2_s,  idx)
    cpi   = cpi_s.resample("D").ffill().reindex(idx).ffill().bfill()
    vix_z  = (vix  - vix.rolling(60).mean())  / (vix.rolling(60).std()  + 1e-9)
    tlt_z  = (tlt  - tlt.rolling(60).mean())  / (tlt.rolling(60).std()  + 1e-9)
    # Risk-off when VIX elevated + MOVE elevated + TLT rallying
    feat["risk_off_composite"]  = (vix_z - tlt_z) / 2.0   # + = risk-off
    feat["risk_off_direction"]  = feat["risk_off_composite"].diff(5)

    # ── [6d] Macro features ───────────────────────────────────────────────────────
    feat["vix_level"]           = vix
    feat["vix_change_5d"]       = vix.pct_change(5)
    feat["vix_zscore_60d"]      = vix_z
    feat["yield_spread"]        = yr10 - yr2
    feat["yield_spread_ch5"]    = (yr10 - yr2).diff(5)
    feat["yield_10yr"]          = yr10
    feat["cpi_yoy"]             = cpi
    feat["cpi_change_3m"]       = cpi - cpi.shift(63)

    n_features_base = feat.shape[1]
    print(f"  Base features: {n_features_base}")

    # ── [3] DELTA FEATURES — 1-step changes at each timestep ──────────────────────
    # These are critical for the GRU to detect dynamics rather than just levels.
    # We compute deltas for the most informative features and store them separately;
    # they will be stacked onto the sequence at sequence-build time.
    DELTA_COLS = [
        "trend_fast_slope_5", "trend_fast_vs_slow", "trend_cross_signal",
        "adx_14", "di_diff", "ma_stack_score", "bb_position",
        "price_vs_fast", "price_vs_slow", "cycle_level", "cycle_zscore",
        "vix_level", "yield_spread", "risk_off_composite",
        "trend_concordance", "spy_ret_5d", "rsi_14"
    ]
    DELTA_COLS = [c for c in DELTA_COLS if c in feat.columns]

    delta_feat = feat[DELTA_COLS].diff(1)
    delta_feat.columns = [f"Δ{c}" for c in DELTA_COLS]
    feat = pd.concat([feat, delta_feat], axis=1)

    n_features = feat.shape[1]
    print(f"  Total features (base + deltas): {n_features}")
    print(f"  Delta features added: {len(DELTA_COLS)}")

    # ── [7] REGIME FEATURE — ADX regime for conditioning the fusion head ──────────
    # Discretised into 3 bins: Weak(0) / Moderate(1) / Strong(2) trend
    adx_aligned = adx.reindex(idx).ffill().bfill()
    adx_regime  = pd.cut(adx_aligned, bins=[-np.inf, 20, 35, np.inf],
                        labels=[0, 1, 2]).astype(float)
    
    return feat, adx_aligned, adx_regime, di_bull
