from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TREND-STATE LABELS  [CHANGE 1]
#
#   Instead of 20-day forward return buckets, label each day with its
#   CURRENT trend state based on the decomposed signal:
#
#   State 3 — Trending Up:    fast > slow, ADX > 20, DI+ > DI-
#   State 2 — Transition Up:  fast > slow but ADX weak OR recent crossover up
#   State 1 — Transition Down: fast < slow but ADX weak OR recent crossover dn
#   State 0 — Trending Down:  fast < slow, ADX > 20, DI- > DI+
#
#   This is strictly causal — all inputs are computed from historical data only.
# ══════════════════════════════════════════════════════════════════════════════

def label_generate(feat, adx_aligned, adx_regime, trend_fast, trend_slow, di_bull, idx, train_frac, val_frac):
    fast_above_slow = (trend_fast > trend_slow).reindex(idx).astype(int)
    adx_strong      = (adx_aligned > 20).astype(int)

    # Combine into 4-class trend state
    def build_trend_state(fast_above, adx_str, di_b):
        state = np.zeros(len(fast_above), dtype=int)
        for i in range(len(fast_above)):
            fa = fast_above.iloc[i]
            ad = adx_str.iloc[i]
            db = di_b.iloc[i]
            if fa == 1 and ad == 1 and db == 1:
                state[i] = 3   # Trending Up
            elif fa == 1:
                state[i] = 2   # Transition Up (fast above slow but weak/mixed)
            elif fa == 0 and ad == 1 and db == 0:
                state[i] = 0   # Trending Down
            else:
                state[i] = 1   # Transition Down
        return pd.Series(state, index=fast_above.index)

    labels_raw = build_trend_state(fast_above_slow, adx_strong, di_bull)

    # ── Label smoothing: require a state to persist ≥5 days before recording ──────
    # A rolling mode with window=5 suppresses single-day flickers where the trend
    # state oscillates due to borderline ADX / DI readings.  This does NOT look
    # forward — pd.Series.rolling is applied left-to-right and mode() only sees the
    # current window of *past* labels (including the current day).
    labels = (
        labels_raw
        .rolling(10, min_periods=1)
        .apply(lambda x: pd.Series(x).mode()[0], raw=False)
        .astype(int)
    )
    n_changed = (labels != labels_raw).sum()
    print(f"  Label smoothing (window=10): {n_changed} days changed "
        f"({n_changed/len(labels)*100:.1f}% of samples)")

    STATE_NAMES = {0: "Trending-Down", 1: "Trans-Down", 2: "Trans-Up", 3: "Trending-Up"}

    valid_idx    = feat.dropna().index.intersection(labels.dropna().index)
    feat_clean   = feat.loc[valid_idx]
    labels_clean = labels.loc[valid_idx]
    regime_clean = adx_regime.loc[valid_idx]

    split_tr  = int(len(valid_idx) * train_frac) # TRAIN_FRAC
    split_val = int(len(valid_idx) * (train_frac + val_frac)) # VAL_FRAC
    train_idx = valid_idx[:split_tr]
    val_idx   = valid_idx[split_tr:split_val]
    test_idx  = valid_idx[split_val:]

    vc    = labels_clean.value_counts().sort_index()
    naive = float(vc.max() / vc.sum())

    print(f"  Total valid samples : {len(valid_idx)}")
    print(f"  Train (50%)         : {len(train_idx)}  ({train_idx[0].date()} → {train_idx[-1].date()})")
    print(f"  Val   (25%)         : {len(val_idx)}  ({val_idx[0].date()} → {val_idx[-1].date()})")
    print(f"  Test  (25%)         : {len(test_idx)}  ({test_idx[0].date()} → {test_idx[-1].date()})")
    for lbl, cnt in vc.items():
        print(f"    {STATE_NAMES[lbl]:15s} ({lbl}): {cnt:5d}  ({cnt/len(labels_clean)*100:.1f}%)")
    print(f"  Naive baseline      : {naive:.4f}")

    return feat_clean, labels_clean, regime_clean, split_tr, split_val
