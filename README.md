# AI-Powered Market Regime Classification Platform
A full-stack ML inference platform that decomposes equity price signals into low-frequency trend and residual components, classifies market trend regimes using a CNN-GRU model, and provides REST APIs, backtesting, and dashboard visualization for financial time-series analysis. ////
SPY

## Current Status 6/1

- Refactored the original CNN-GRU notebook into modular Python components.
- Built a local inference pipeline that loads Excel market data and predicts the next-day market regime.
- Current model outputs four regime probabilities:
  - Trending-Down
  - Trans-Down
  - Trans-Up
  - Trending-Up

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