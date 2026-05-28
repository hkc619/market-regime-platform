import os
from app.core.config import AppConfig

from backend.app.ml.data_loader import load_ohlcv, load_macro_daily, load_cpi
from backend.app.ml.decomposition import butterworth_noncausal, dual_ewm_decomposition
from backend.app.ml.features import features
from backend.app.ml.dataset import build_dataset
from backend.app.ml.model import DualCNNGRUFusion
from backend.app.ml.training import train_model


data_path = os.getenv("DATA_PATH")
print("\n" + "=" * 70)
print("  SECTION 1: LOADING DATA")
print("=" * 70)

spy_close = load_ohlcv(data_path, "SPY", "Close")
spy_vol =  load_ohlcv(data_path, "SPY", "Volumn")
spy_high = load_ohlcv(data_path, "SPY", "High")
spy_low = load_ohlcv(data_path, "SPY", "Low")
qqq_close = load_ohlcv(data_path, "SPY", "Close")
tlt_close = load_ohlcv(data_path, "SPY", "Close")

vix_s, yr10_s, yr2_s = load_macro_daily(data_path)
cpi_s = load_cpi(data_path)

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

feat, adx_aligned, adx_regime = features(spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close,
                            trend_fast, trend_slow, cycle_comp, noise_comp, 
                            vix_s, yr10_s, yr2_s, cpi_s)

print("\n" + "=" * 70)
print("  SECTION 4: TREND-STATE LABELS  [v2: current regime, not fwd return]")
print("=" * 70)


print("\n" + "=" * 70)
print("  SECTION 6: DUAL-SCALE SEQUENCE DATASET  [v2: 20-day + 60-day]")
print("=" * 70)

print("\n" + "=" * 70)
print("  SECTION 7: DUAL-SCALE CNN-GRU + REGIME CONDITIONING  [v2]")
print("=" * 70)


print("\n" + "=" * 70)
print("  SECTION 8: TRAINING MODELS")
print("=" * 70)

def main():
    if __name__ == "main":
        config = AppConfig()