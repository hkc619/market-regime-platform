import os
from dotenv import load_dotenv

from data_loader import load_ohlcv
from preprocessing import butterworth_noncausal, dual_ewm_decomposition


load_dotenv()
data_path = os.getenv("DATA_PATH")
print("\n" + "=" * 70)
print("  SECTION 1: LOADING DATA")
print("=" * 70)
spy_close = load_ohlcv(data_path, "SPY", "Close")

print("\n" + "=" * 70)
print("  SECTION 2: DUAL-SCALE CAUSAL SIGNAL DECOMPOSITION  [v2: +cycle]")
print("=" * 70)

spy_arr    = spy_close.values.astype(float)
trend_viz  = butterworth_noncausal(spy_arr)  # viz only

trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

print(f"  trend_fast (EWM-20) σ  : ${trend_fast.std():.2f}")
print(f"  trend_slow (EWM-60) σ  : ${trend_slow.std():.2f}")
print(f"  cycle component σ      : ${cycle_comp.std():.2f}")
print(f"  noise component σ      : ${noise_comp.std():.2f}")

print("\n" + "=" * 70)
print("  SECTION 3: FEATURE ENGINEERING  [v2: +ADX +S/R +delta +composite]")
print("=" * 70)

decompose(spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close,
           trend_fast, trend_slow, cycle_comp, noise_comp, 
           vix_s, yr10_s, yr2_s, cpi_s)

