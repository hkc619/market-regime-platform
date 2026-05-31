from scipy.signal import butter, filtfilt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: IMPROVED SIGNAL DECOMPOSITION  [CHANGES 1 + 5]
#   - Dual-EWM trend (fast + slow) for inflection detection
#   - Cycle component: residual split into cycle + noise via 2nd EWM pass
#   - Butterworth kept for visualization only
# ══════════════════════════════════════════════════════════════════════════════

def butterworth_noncausal(arr, cutoff=0.04, order=4):
    """Non-causal Butterworth for visualization only."""
    b, a = butter(order, cutoff, btype="low", analog=False)
    return filtfilt(b, a, arr)

def dual_ewm_decomposition(series):
    """
    Causal decomposition into 3 components:
      trend_fast  : EWM(span=20)  — tracks medium-term trend
      trend_slow  : EWM(span=60)  — tracks long-term structural trend
      cycle       : trend_fast - trend_slow  — medium-freq oscillation
      noise       : price - trend_fast       — high-freq residual
    All strictly causal (ewm adjust=False).
    """
    arr   = pd.Series(series.values.astype(float), index=series.index)
    t_fast = arr.ewm(span=20, adjust=False).mean()
    t_slow = arr.ewm(span=60, adjust=False).mean()
    cycle  = t_fast - t_slow
    noise  = arr - t_fast
    return t_fast, t_slow, cycle, noise

