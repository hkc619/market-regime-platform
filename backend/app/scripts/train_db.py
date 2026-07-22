from datetime import date
from pathlib import Path

from app.db.session import get_db
from app.ml.data_loader_db import load_training_data_from_db
from app.ml.metadata import build_model_metadata_entry, append_model_metadata
from app.core.config import TrainingConfig
from app.ml.decomposition import dual_ewm_decomposition
from app.ml.features import features
from app.ml.label_generation import label_generate
from app.ml.dataset import build_dataset
from app.ml.training import train_model


today = date.today()

MODEL_OUTPUT_PATH = Path(f'models/cnn_gru_{today}.pt')
METADATA_OUTPUT_PATH = Path(f'models/metadata_{today}.json')

def train():

    db = ""
    config = TrainingConfig()

    try:
        
        print("\n" + "=" * 70)
        print("  SECTION 1: LOADING DATA")
        print("=" * 70)

        raw_data = load_training_data_from_db(
            db=db,
            target_ticker="SPY",
            support1="QQQ",
            support2="TLT",
            start_date=None,
            end_date=None,
        )

        ticker_close = raw_data.ticker_prices["close"]
        ticker_vol =  raw_data.ticker_prices["volumn"]
        ticker_high = raw_data.ticker_prices["high"]
        ticker_low = raw_data.ticker_prices["low"]
        sup1_close = raw_data.sup1_close
        sup2_close = raw_data.sup2_close

        vix_s = raw_data.macro_daily["vix"]
        yr10_s = raw_data.macro_daily["yield_10yr"]
        yr2_s = raw_data.macro_daily["yield_2yr"]
        cpi_s = raw_data.macro_monthly


        print("\n" + "=" * 70)
        print("  SECTION 2: DUAL-SCALE CAUSAL SIGNAL DECOMPOSITION  [v2: +cycle]")
        print("=" * 70)

        trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(ticker_close)

        print(f"  trend_fast (EWM-20) σ  : ${trend_fast.std():.2f}")
        print(f"  trend_slow (EWM-60) σ  : ${trend_slow.std():.2f}")
        print(f"  cycle component σ      : ${cycle_comp.std():.2f}")
        print(f"  noise component σ      : ${noise_comp.std():.2f}")

        print("\n" + "=" * 70)
        print("  SECTION 3: FEATURE ENGINEERING  [v2: +ADX +S/R +delta +composite]")
        print("=" * 70)

        feat, adx_aligned, adx_regime, di_bull = features(
            ticker_close, ticker_vol, ticker_high, ticker_low,
            sup1_close, sup2_close,
            trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s
            )

        print("\n" + "=" * 70)
        print("  SECTION 4: TREND-STATE LABELS  [v2: current regime, not fwd return]")
        print("=" * 70)

        idx = ticker_close.index

        feat_clean, labels_clean, regime_clean, split_tr, split_val, scaler = label_generate(
            feat, adx_aligned, adx_regime, trend_fast, trend_slow, 
            di_bull, idx, train_frac = config.TRAIN_FRAC, val_frac = config.VAL_FRAC
            )
        print("\n regime_clean shape: ", regime_clean.shape)
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
        for mode in ["fusion"]:
            model, metrics, preds, trues, history = train_model(
                train_ds, val_ds, test_ds, 
                mode=mode, epochs=config.EPOCHS, patience=config.PATIENCE, 
                lr=config.LR, device=config.DEVICE, batch_size=config.BATCH_SIZE, n_feat=n_feat, class_weights=class_weights
                )
            results_nn[mode] = {
                "model": model, "acc": acc, "f1": f1,
                "preds": preds, "trues": trues, "history": history
        
            }
        

        metadata_entry = build_model_metadata_entry(
            version="v3-db",
            model_name="CNN-GRU Fusion v3 DB",
            checkpoint_path="models/fusion_v3_db.pth",
            scaler_path="models/scaler_v3_db.pkl",
            target_asset="SPY",
            assets=["SPY", "QQQ", "TLT"],
            raw_data_start_date=raw_data.target_prices.index.min().date(),
            raw_data_end_date=raw_data.target_prices.index.max().date(),
            training_rows=len(raw_data.target_prices),
            feature_rows=len(feature_df),
            test_rows=len(trues),
            metrics=metrics,
        )

        append_model_metadata(
            metadata_path="models/metadata.json",
            model_entry=metadata_entry,
)

        print(f"Saved metadata to {METADATA_OUTPUT_PATH}")
    finally:
        db.close()


if __name__ == "__main__":
    train()
    print("finish")
