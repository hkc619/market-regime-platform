import numpy as np

from app.ml.decomposition import dual_ewm_decomposition
from app.ml.features import features
from app.ml.label_generation import generate_predict_label
from app.core.config import ModelInputBundle

class InsufficientFeatureDataError(Exception):
    pass


def build_latest_model_input(raw):
    trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(
        raw.ticker_close
    )

    # print(raw.ticker_close.head())
    # print(raw.vix_s.head())
    # print(raw.cpi_s.head())
    # print("trend_fast:", cycle_comp.head())
    # print("trend_slow:", noise_comp.head())

    feat, adx_aligned, adx_regime, di_bull = features(
        ticker_close=raw.ticker_close,
        ticker_vol=raw.ticker_vol,
        ticker_high=raw.ticker_high,
        ticker_low=raw.ticker_low,
        sup1_close=raw.sup0_close,
        sup2_close=raw.sup1_close,
        trend_fast=trend_fast,
        trend_slow=trend_slow,
        cycle_comp=cycle_comp,
        noise_comp=noise_comp,
        vix_s=raw.vix_s,
        yr10_s=raw.yr10_s,
        yr2_s=raw.yr2_s,
        cpi_s=raw.cpi_s,
    )
    
    feat_clean, regime_clean = generate_predict_label(
        feat,
        adx_aligned,
        adx_regime,
        trend_fast,
        trend_slow,
        di_bull,
        raw.idx,
    )

    if len(feat_clean) < 60:
        raise InsufficientFeatureDataError(
            f"Only {len(feat_clean)} valid feature rows after feature engineering."
        )

    latest_60_feat = feat_clean.values[-60:]
    latest_regime = regime_clean.values[-1].astype(np.int64)

    return ModelInputBundle(
        ticker=raw.ticker,
        latest_60_feat=latest_60_feat,
        latest_regime=latest_regime,
        feature_rows=len(feat_clean),
        feature_dim=feat_clean.shape[1],
        start_date=feat_clean.index[-60],
        end_date=feat_clean.index[-1],
    )