from datetime import date, datetime
from unittest.mock import MagicMock

from app.services.prediction_service import get_prediction_history


def test_get_prediction_history_success(mocker):
    db = MagicMock()

    mocker.patch(
        "app.services.prediction_service.get_prediction_history_by_ticker",
        return_value=[
            {
                "prediction_id": 1,
                "ticker": "SPY",
                "as_of_date": date(2026, 7, 17),
                "predicted_class": 0,
                "predicted_regime": "Trending-Up",
                "confidence": 0.82,
                "prob_trending_down": 0.03,
                "prob_transition_down": 0.05,
                "prob_transition_up": 0.10,
                "prob_trending_up": 0.82,
                "model_version": "v2",
                "created_at": datetime(2026, 7, 17, 21, 30),
            }
        ],
    )

    result = get_prediction_history(
        db=db,
        ticker="spy",
        limit=20,
    )

    assert result.ticker == "SPY"
    assert result.count == 1
    assert result.results[0].predicted_regime == "Trending-Up"
    assert result.results[0].confidence == 0.82
    assert result.results[0].probabilities["Trending-Up"] == 0.82


def test_get_prediction_history_empty_result(mocker):
    db = MagicMock()

    mocker.patch(
        "app.services.prediction_service.get_prediction_history_by_ticker",
        return_value=[],
    )

    result = get_prediction_history(
        db=db,
        ticker="SPY",
        limit=20,
    )

    assert result.ticker == "SPY"
    assert result.count == 0
    assert result.results == []


def test_get_prediction_history_parses_json_string_probabilities(mocker):
    db = MagicMock()

    mocker.patch(
        "app.services.prediction_service.get_prediction_history_by_ticker",
        return_value=[
            {
                "prediction_id": 1,
                "ticker": "SPY",
                "as_of_date": date(2026, 7, 17),
                "predicted_class": 0,
                "predicted_regime": "Trending-Up",
                "confidence": 0.82,
                "prob_trending_down": 0.03,
                "prob_transition_down": 0.05,
                "prob_transition_up": 0.10,
                "prob_trending_up": 0.82,
                "model_version": "v2",
                "created_at": datetime(2026, 7, 17, 21, 30),
            }
        ],
    )

    result = get_prediction_history(
        db=db,
        ticker="SPY",
        limit=20,
    )
    
    assert result.count == 1
    assert result.results[0].probabilities["Trending-Up"] == 0.82