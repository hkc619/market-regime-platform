from datetime import date, datetime
from unittest.mock import MagicMock

from app.repositories.prediction_repository import get_latest_prediction_by_ticker, get_prediction_history_by_ticker

def test_get_latest_prediction_by_ticker_returns_row():
    db = MagicMock()

    fake_row = {
        "prediction_id": 1,
        "ticker": "SPY",
        "as_of_date": date(2026, 7, 17),
        "predicted_class": 0,
        "predicted_regime": "Trending-Up",
        "confidence": 0.82,
        "model_version": "v2",
        "created_at": datetime(2026, 7, 17, 21, 30),
    }

    db.execute.return_value.mappings.return_value.first.return_value = fake_row

    result = get_latest_prediction_by_ticker(
        db=db,
        ticker="spy",
    )

    assert result["ticker"] == "SPY"
    assert result["predicted_regime"] == "Trending-Up"

def test_get_prediction_history_by_ticker_returns_rows():
    db = MagicMock()

    fake_rows = [
        {
            "prediction_id": 1,
            "ticker": "SPY",
            "as_of_date": date(2026, 7, 17),
            "predicted_class": 0,
            "predicted_regime": "Trending-Up",
            "confidence": 0.82,
            "probabilities": {
                "Trending-Up": 0.82,
                "Transition-Up": 0.10,
                "Transition-Down": 0.05,
                "Trending-Down": 0.03,
            },
            "model_version": "v2",
            "created_at": datetime(2026, 7, 17, 21, 30),
        }
    ]

    db.execute.return_value.mappings.return_value.all.return_value = fake_rows

    result = get_prediction_history_by_ticker(
        db=db,
        ticker="spy",
        limit=20,
    )
    

    assert len(result) == 1
    assert result[0]["ticker"] == "SPY"
    assert result[0]["predicted_regime"] == "Trending-Up"

    db.execute.assert_called_once()


def test_get_prediction_history_by_ticker_returns_empty_list():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []

    result = get_prediction_history_by_ticker(
        db=db,
        ticker="SPY",
        limit=20,
    )

    assert result == []