import numpy as np
import pandas as pd
import torch
import pickle
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')

from ml.decomposition import dual_ewm_decomposition
from ml.features import features
from ml.model import DualCNNGRUFusion

from core.config import ModelConfig

def load_data(df):
    spy_close = df["SPY-Close"]
    spy_vol =  df["SPY-Volume"]
    spy_high = df["SPY-High"]
    spy_low = df["SPY-Low"]
    qqq_close = df["QQQ-Close"]
    tlt_close = df["TLT-Close"]

    vix_s, yr10_s, yr2_s = load_macro_daily(data_path)
    cpi_s = load_cpi(data_path)

    trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

    feat, adx_aligned, adx_regime, di_bull = features(spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close,
                                trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s)
    valid_idx = feat.dropna().index
    feat_clean = feat.loc[valid_idx]
    regime_clean = adx_regime.loc[valid_idx].values.astype(np.int64)
    latest_regime = regime_clean[-1]
    if len(feat_clean) < 60:
        return {"error": "Valid data length after feature engineering is less than 60 days."}
    latest_60_feat = feat_clean.values[-60:]
    return latest_60_feat, latest_regime
    


def predict_proba(latest_60_feat, latest_regime):

    config = ModelConfig()
    device = config.DEVICE

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    latest_20_scaled = latest_60_scaled[-20:]
    latest_60_scaled = scaler.transform(latest_60_feat)

    Xm_tensor = torch.tensor(latest_60_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    Xs_tensor = torch.tensor(latest_20_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    r_tensor = torch.tensor([latest_regime], dtype=torch.long).to(device)

    model = DualCNNGRUFusion(Xs_tensor.shape[2], mode="dual_cnn").to(device)
    model.load_state_dict(torch.load("dual_cnn_v2.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(Xs_tensor, Xm_tensor, r_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)

    STATE_NAMES = {0: "Trending-Down", 1: "Trans-Down", 2: "Trans-Up", 3: "Trending-Up"}


    return {
        "predicted_state": STATE_NAMES[predicted_class],
        "probabilities": {
            STATE_NAMES: float(probabilities),
            STATE_NAMES[21]: float(probabilities[21]),
            STATE_NAMES[22]: float(probabilities[22]),
            STATE_NAMES[2]: float(probabilities[2]),
        }
    }



def predict_status():
    pass