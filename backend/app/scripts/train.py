import os
import sys
sys.path.append('../../app')
from core.config import DataConfig, TrainingConfig


from ml.data_loader import load_ohlcv, load_macro_daily, load_cpi
from ml.decomposition import dual_ewm_decomposition
from ml.features import features
from ml.label_generation import label_generate
from ml.dataset import build_dataset
from ml.training import train_model

def train():

    data_path = DataConfig().data_path
    config = TrainingConfig()
    
    print("\n" + "=" * 70)
    print("  SECTION 1: LOADING DATA")
    print("=" * 70)

    spy_close = load_ohlcv(data_path, "SPY", "Close")
    spy_vol =  load_ohlcv(data_path, "SPY", "Volume")
    spy_high = load_ohlcv(data_path, "SPY", "High")
    spy_low = load_ohlcv(data_path, "SPY", "Low")
    qqq_close = load_ohlcv(data_path, "QQQ", "Close")
    tlt_close = load_ohlcv(data_path, "TLT", "Close")

    vix_s, yr10_s, yr2_s = load_macro_daily(data_path)
    cpi_s = load_cpi(data_path)

    print("\n" + "=" * 70)
    print("  SECTION 2: DUAL-SCALE CAUSAL SIGNAL DECOMPOSITION  [v2: +cycle]")
    print("=" * 70)

    spy_arr    = spy_close.values.astype(float)
    #trend_viz  = butterworth_noncausal(spy_arr)  # viz only

    trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

    print(f"  trend_fast (EWM-20) σ  : ${trend_fast.std():.2f}")
    print(f"  trend_slow (EWM-60) σ  : ${trend_slow.std():.2f}")
    print(f"  cycle component σ      : ${cycle_comp.std():.2f}")
    print(f"  noise component σ      : ${noise_comp.std():.2f}")

    print("\n" + "=" * 70)
    print("  SECTION 3: FEATURE ENGINEERING  [v2: +ADX +S/R +delta +composite]")
    print("=" * 70)

    feat, adx_aligned, adx_regime, di_bull = features(spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close,
                                trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s)

    print("\n" + "=" * 70)
    print("  SECTION 4: TREND-STATE LABELS  [v2: current regime, not fwd return]")
    print("=" * 70)
    idx = spy_close.index

    feat_clean, labels_clean, regime_clean, split_tr, split_val, scaler = label_generate(
        feat, adx_aligned, adx_regime, trend_fast, trend_slow, 
        di_bull, idx, train_frac = config.TRAIN_FRAC, val_frac = config.VAL_FRAC
        )

    print("\n" + "=" * 70)
    print("  SECTION 6: DUAL-SCALE SEQUENCE DATASET  [v2: 20-day + 60-day]")
    print("=" * 70)

    train_ds, val_ds, test_ds, n_feat, class_weights = build_dataset(
        feat_clean, labels_clean, regime_clean, 
        split_tr, split_val, seq_len_s=config.SEQ_LEN_S, seq_len_m=config.SEQ_LEN_M, scaler=scaler
        )

    print("\n" + "=" * 70)
    print("  SECTION 7: DUAL-SCALE CNN-GRU + REGIME CONDITIONING  [v2]")
    print("=" * 70)


    print("\n" + "=" * 70)
    print("  SECTION 8: TRAINING MODELS")
    print("=" * 70)
    results_nn = {}
    for mode in ["cnn_only", "gru_only", "dual_cnn", "fusion"]:
        model, acc, f1, preds, trues, history = train_model(train_ds, val_ds, test_ds, 
                mode=mode, epochs=config.EPOCHS, patience=config.PATIENCE, 
                lr=config.LR, device=config.DEVICE, batch_size=config.BATCH_SIZE, n_feat=n_feat, class_weights=class_weights)
        results_nn[mode] = {
            "model": model, "acc": acc, "f1": f1,
            "preds": preds, "trues": trues, "history": history
    
        }


if __name__ == "__main__":
    train()
    print("finish")
