# AI-Powered Market Regime Classification Platform

A market regime classification backend built with FastAPI, PostgreSQL, and a CNN-GRU model. The project focuses on SPY inference, prediction history, market and macro data refresh, and historical classification replay. A frontend dashboard and Docker deployment are planned.

## Current Progress — 2026-09-09

**Status: backend integration and validation in progress.** The main backend layers are implemented, but the complete workflow has not yet been validated against real database and external data services following the recent fixes.

### Implemented Backend Components

- API, service, repository, and external data provider layers.
- PostgreSQL operations for market prices, daily/monthly macro data, prediction history, and refresh logs.
- Model loading at application startup, inference input preparation, and prediction persistence.
- Latest prediction and prediction history endpoints.
- Historical classification replay with regime distribution, coverage, skip reasons, and confidence summaries.
- Health endpoints, request IDs, request latency logging, and application error handling.

These components are at different stages of validation; implementation does not imply that every endpoint currently works end to end.

### Recent Fixes

- **`b71be83` — Prediction input and history routing:** made `as_of_date` optional for latest inference, required it for historical mode, and resolved the endpoint/service name collision that caused recursive history queries.
- **`2b51a08` — API exception handling:** preserved known `AppError` and `HTTPException` responses across five endpoints, removed unreachable exception handlers, and returned generic HTTP 500 responses for unexpected failures.
- Added explicit HTTP 422 handling for invalid backtest dates and reversed date ranges.
- Added regression tests for success responses, service call arguments, error status/code preservation, and history queries through the real service.

### Validation Status

The latest targeted regression run completed with **69 passed and 3 warnings**:

```bash
cd backend
python -m pytest -q \
  app/tests/test_api_exception_handling.py \
  app/tests/test_prediction_history_api.py \
  app/tests/test_prediction_history_service.py \
  app/tests/test_error_mapping.py
```

Run this command in the configured backend environment. These tests isolate database and external service dependencies. This result is not a full-suite pass or proof of live end-to-end functionality. Other tests still have known failures, and the earlier code review excluded `ml`, so model methodology and evaluation validity remain to be reviewed.

### Remaining Work

- Repair the yfinance provider invocation and normalize ordinary/MultiIndex data columns consistently.
- Fix refresh response contracts, empty-data handling, and remaining service exception classification issues.
- Validate historical window boundaries, supporting-asset alignment, and CPI availability/filling.
- Make stored model versions match the model actually loaded.
- Establish a complete test baseline with real PostgreSQL and inference integration tests, followed by CI and browser E2E tests.
- Add Docker Compose, database migrations, reproducible configuration, and demo data.
- Build the frontend dashboard for predictions, history, replay, and data management.

The current backtest functionality summarizes historical classifications; it does not establish trading-strategy returns or profitability. Timing and data-vintage limitations must be addressed before claiming absence of look-ahead bias.

See the [Project Completion Roadmap](notes/20260908.md) for phases, checklists, and acceptance criteria.

## System Architecture

```text
Planned frontend dashboard
          |
          v
FastAPI routes → Services → Repositories → PostgreSQL
                    |
                    +→ Data providers (Yahoo Finance / FRED)
                    |
                    +→ Feature preparation → Scaler / CNN-GRU inference
```

The frontend and Docker environment are planned work. The diagram describes the backend structure and intended frontend connection.

## Earlier Development Milestones

The entries below record earlier implementation progress and legacy examples. They are not current end-to-end validation results.

### 2026-07-08

Implemented at this milestone:
- Upsert-based market data update into market_prices
- data_update_log table for recording refresh results
- Refresh result handling for:
  - success
  - up_to_date
  - no_new_data
  - failed

### 2026-07-01

Implemented at this milestone:
- PostgreSQL schema established
- OHLCV data imported
- macro_daily data imported
- FastAPI connected to database
- DB-backed data_service implemented
- raw market rows converted to inference-ready Series
- feature engineering pipeline connected
- model input generated successfully
- POST endpoint can return prediction result

### 2026-06-01

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

## Legacy Excel Prediction Script

This earlier entry point depends on local Excel files and path configuration. It is retained for reference and has not been revalidated; the active API uses database-backed prediction services.

```bash
python3 backend/app/scripts/predict.py
```

## Historical Example Output

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

The backend currently requires an existing PostgreSQL schema and data, installed backend dependencies, and `backend/.env` configured with `DATABASE_URL`, `FRED_API_KEY`, and `METADATA_PATH`. Model metadata must reference accessible checkpoint and scaler files. Automated setup through Docker and migrations is planned.

From the repository root, with the backend Python environment activated:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Open [Swagger UI](http://127.0.0.1:8000/docs) after startup. Check `/api/v1/health` for model status and `/api/v1/health/db` for database connectivity. The service can start with a degraded model state; startup alone does not establish readiness for prediction.
