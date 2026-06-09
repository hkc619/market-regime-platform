import numpy as np
import torch
import pickle
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')

from ml.decomposition import dual_ewm_decomposition
from ml.label_generation import generate_predict_label
from ml.features import features
from ml.model import DualCNNGRUFusion
from ml.data_loader import load_ohlcv, load_macro_daily, load_cpi

from core.config import ModelConfig

def load_data(data_path):
    print("loading data...")
    spy_close = load_ohlcv(data_path, "SPY", "Close")
    spy_vol =  load_ohlcv(data_path, "SPY", "Volume")
    spy_high = load_ohlcv(data_path, "SPY", "High")
    spy_low = load_ohlcv(data_path, "SPY", "Low")
    qqq_close = load_ohlcv(data_path, "QQQ", "Close")
    tlt_close = load_ohlcv(data_path, "TLT", "Close")

    ticker_close, ticker_vol, ticker_high, ticker_low = spy_close, spy_vol, spy_high, spy_low
    sup1_close, sup2_close = qqq_close, tlt_close

    vix_s, yr10_s, yr2_s = load_macro_daily(data_path)
    cpi_s = load_cpi(data_path)

    trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

    feat, adx_aligned, adx_regime, di_bull = features(ticker_close, ticker_vol, ticker_high, ticker_low,
                                sup1_close, sup2_close,
                                trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s)
    idx = ticker_close.index

    feat_clean, regime_clean = generate_predict_label(feat, adx_aligned, adx_regime, trend_fast, trend_slow, di_bull, idx)
    print("regime_clean.shape: ", regime_clean.shape)
    latest_regime = regime_clean.values[-1].astype(np.int64)
    
    if len(feat_clean) < 60:
        return {"error": "Valid data length after feature engineering is less than 60 days."}
    latest_60_feat = feat_clean.values[-60:]
    return latest_60_feat, latest_regime, idx
    


def predict_proba(latest_60_feat, latest_regime, idx):
    print("predicting...")

    config = ModelConfig()
    device = config.DEVICE

    with open(f"{config.model_path}scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    
    latest_60_scaled = scaler.transform(latest_60_feat)
    latest_20_scaled = latest_60_scaled[-20:]

    Xm_tensor = torch.tensor(latest_60_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    Xs_tensor = torch.tensor(latest_20_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    r_tensor = torch.tensor([latest_regime], dtype=torch.long).to(device)

    model = DualCNNGRUFusion(Xs_tensor.shape[2], mode="dual_cnn").to(device)
    model.load_state_dict(torch.load(f"{config.model_path}dual_cnn_v2.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(Xs_tensor, Xm_tensor, r_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)

    STATE_NAMES = {0: "Trending-Down", 1: "Trans-Down", 2: "Trans-Up", 3: "Trending-Up"}

    
    return {
        "last day: ": idx[-1].strftime("%Y-%m-%d"),
        "predicted_state": STATE_NAMES[predicted_class],
        "probabilities": {
            STATE_NAMES[0]: float(probabilities[0][0]),
            STATE_NAMES[1]: float(probabilities[0][1]),
            STATE_NAMES[2]: float(probabilities[0][2]),
            STATE_NAMES[3]: float(probabilities[0][3]),
        }
    }

def load_data(data_path):
    print("loading data...")
    spy_close = load_ohlcv(data_path, "SPY", "Close")
    spy_vol =  load_ohlcv(data_path, "SPY", "Volume")
    spy_high = load_ohlcv(data_path, "SPY", "High")
    spy_low = load_ohlcv(data_path, "SPY", "Low")
    qqq_close = load_ohlcv(data_path, "QQQ", "Close")
    tlt_close = load_ohlcv(data_path, "TLT", "Close")

    ticker_close, ticker_vol, ticker_high, ticker_low = spy_close, spy_vol, spy_high, spy_low
    sup1_close, sup2_close = qqq_close, tlt_close

    vix_s, yr10_s, yr2_s = load_macro_daily(data_path)
    cpi_s = load_cpi(data_path)

    trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

    feat, adx_aligned, adx_regime, di_bull = features(ticker_close, ticker_vol, ticker_high, ticker_low,
                                sup1_close, sup2_close,
                                trend_fast, trend_slow, cycle_comp, noise_comp, vix_s, yr10_s, yr2_s, cpi_s)
    idx = ticker_close.index

    feat_clean, regime_clean = generate_predict_label(feat, adx_aligned, adx_regime, trend_fast, trend_slow, di_bull, idx)
    print("regime_clean.shape: ", regime_clean.shape)
    latest_regime = regime_clean.values[-1].astype(np.int64)
    
    if len(feat_clean) < 60:
        return {"error": "Valid data length after feature engineering is less than 60 days."}
    latest_60_feat = feat_clean.values[-60:]
    return latest_60_feat, latest_regime, idx
    


def predict_proba(latest_60_feat, latest_regime, idx):
    print("predicting...")

    config = ModelConfig()
    device = config.DEVICE

    with open(f"{config.model_path}scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    
    latest_60_scaled = scaler.transform(latest_60_feat)
    latest_20_scaled = latest_60_scaled[-20:]

    Xm_tensor = torch.tensor(latest_60_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    Xs_tensor = torch.tensor(latest_20_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    r_tensor = torch.tensor([latest_regime], dtype=torch.long).to(device)

    model = DualCNNGRUFusion(Xs_tensor.shape[2], mode="dual_cnn").to(device)
    model.load_state_dict(torch.load(f"{config.model_path}dual_cnn_v2.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(Xs_tensor, Xm_tensor, r_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)

    STATE_NAMES = {0: "Trending-Down", 1: "Trans-Down", 2: "Trans-Up", 3: "Trending-Up"}

    
    return {
        "last day: ": idx[-1].strftime("%Y-%m-%d"),
        "predicted_state": STATE_NAMES[predicted_class],
        "probabilities": {
            STATE_NAMES[0]: float(probabilities[0][0]),
            STATE_NAMES[1]: float(probabilities[0][1]),
            STATE_NAMES[2]: float(probabilities[0][2]),
            STATE_NAMES[3]: float(probabilities[0][3]),
        }
    }


