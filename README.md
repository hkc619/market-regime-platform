# AI-Powered Market Regime Classification Platform
A full-stack ML inference platform that decomposes equity price signals into low-frequency trend and residual components, classifies market trend regimes using a CNN-GRU model, and provides REST APIs, backtesting, and dashboard visualization for financial time-series analysis. ////
SPY

## System Architecture:


## Current Status 7/8
Completed:
- Upsert-based market data update into market_prices
- data_update_log table for recording refresh results
- Refresh result handling for:
  - success
  - up_to_date
  - no_new_data
  - failed

## Current Status 7/1
Completed:
- PostgreSQL schema established
- OHLCV data imported
- macro_daily data imported
- FastAPI connected to database
- DB-backed data_service implemented
- raw market rows converted to inference-ready Series
- feature engineering pipeline connected
- model input generated successfully
- POST endpoint can return prediction result

## Current Status 6/1

- Refactored the original CNN-GRU notebook into modular Python components.
- Built a local inference pipeline that loads Excel market data and predicts the next-day market regime.
- Current model outputs four regime probabilities:
  - Trending-Down
  - Trans-Down
  - Trans-Up
  - Trending-Up
```json
 "base_feature_columns": [
    "trend_fast_slope_5",
    "trend_fast_slope_20",
    "trend_slow_slope_20",
    "trend_slow_slope_60",
    "trend_accel",
    "trend_accel_slow",
    "trend_fast_vs_slow",
    "trend_cross_signal",
    "trend_cross_change",
    "price_vs_fast",
    "price_vs_slow",

    "cycle_level",
    "cycle_slope",
    "cycle_zscore",
    "noise_abs_20",

    "adx_14",
    "adx_zscore_60",
    "di_diff",
    "adx_trend_strength",

    "ma_cross_5_20",
    "ma_cross_20_50",
    "ma_cross_50_200",
    "price_vs_200d",
    "ma_stack_score",

    "dist_swing_high",
    "dist_swing_low",
    "dist_52w_high",
    "dist_52w_low",
    "bb_position",
    "bb_width",

    "spy_ret_5d",
    "spy_vol_5d",
    "spy_ret_10d",
    "spy_vol_10d",
    "spy_ret_20d",
    "spy_vol_20d",
    "spy_ret_60d",
    "spy_vol_60d",
    "rsi_14",
    "vol_ratio_20",

    "qqq_ret_20d",
    "tlt_ret_20d",
    "spy_qqq_spread",
    "spy_tlt_spread",

    "trend_concordance",
    "equity_bond_diverge",

    "risk_off_composite",
    "risk_off_direction",

    "vix_level",
    "vix_change_5d",
    "vix_zscore_60d",
    "yield_spread",
    "yield_spread_ch5",
    "yield_10yr",
    "cpi_yoy",
    "cpi_change_3m"
  ],
  "delta_feature_columns": [
    "trend_fast_slope_5",
    "trend_fast_vs_slow",
    "trend_cross_signal",
    "adx_14",
    "di_diff",
    "ma_stack_score",
    "bb_position",
    "price_vs_fast",
    "price_vs_slow",
    "cycle_level",
    "cycle_zscore",
    "vix_level",
    "yield_spread",
    "risk_off_composite",
    "trend_concordance",
    "spy_ret_5d",
    "rsi_14"
  ],
  "feature_generation": {
    "uses_low_frequency_trend": true,
    "uses_cycle_component": true,
    "uses_noise_component": true,
    "uses_adx_features": true,
    "uses_moving_average_stack": true,
    "uses_support_resistance_features": true,
    "uses_bollinger_band_features": true,
    "uses_cross_asset_features": true,
    "uses_macro_features": true,
    "uses_delta_features": true
  },
  "feature_groups": {
    "trend_structure": [
      "trend_fast_slope_5",
      "trend_fast_slope_20",
      "trend_slow_slope_20",
      "trend_slow_slope_60",
      "trend_accel",
      "trend_accel_slow",
      "trend_fast_vs_slow",
      "trend_cross_signal",
      "trend_cross_change",
      "price_vs_fast",
      "price_vs_slow"
    ],
    "cycle_and_noise": [
      "cycle_level",
      "cycle_slope",
      "cycle_zscore",
      "noise_abs_20"
    ],
    "adx_trend_strength": [
      "adx_14",
      "adx_zscore_60",
      "di_diff",
      "adx_trend_strength"
    ],
    "moving_average_stack": [
      "ma_cross_5_20",
      "ma_cross_20_50",
      "ma_cross_50_200",
      "price_vs_200d",
      "ma_stack_score"
    ],
    "support_resistance": [
      "dist_swing_high",
      "dist_swing_low",
      "dist_52w_high",
      "dist_52w_low",
      "bb_position",
      "bb_width"
    ],
    "price_volume": [
      "spy_ret_5d",
      "spy_vol_5d",
      "spy_ret_10d",
      "spy_vol_10d",
      "spy_ret_20d",
      "spy_vol_20d",
      "spy_ret_60d",
      "spy_vol_60d",
      "rsi_14",
      "vol_ratio_20"
    ],
    "cross_asset": [
      "qqq_ret_20d",
      "tlt_ret_20d",
      "spy_qqq_spread",
      "spy_tlt_spread",
      "xlf_xle_rel",
      "trend_concordance",
      "equity_bond_diverge"
    ],
    "risk_on_off": [
      "risk_off_composite",
      "risk_off_direction"
    ],
    "macro": [
      "vix_level",
      "vix_change_5d",
      "vix_zscore_60d",
      "yield_spread",
      "yield_spread_ch5",
      "yield_10yr",
      "gold_ret_20d",
      "gold_spy_ratio",
      "move_level",
      "move_zscore_60d",
      "cpi_yoy",
      "cpi_change_3m"
    ]
  }
```

## How to Run Local Prediction

```bash
python3 backend/app/scripts/predict.py
```

## Example output

```json
{
    "last day: ": "2025-12-31",
    "predicted_state": "Trans-Up",
    "probabilities": {
        "Trending-Down": 0.004556646570563316,
        "Trans-Down": 0.000556760118342936,
        "Trans-Up": 0.9535136818885803,
        "Trending-Up": 0.04137295112013817
    }
}
```

## API Documentation

FastAPI automatically generates interactive Swagger documentation from the Pydantic request/response schemas.

Run the backend:

```bash
uvicorn backend.app.main:app --reload