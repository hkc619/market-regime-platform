from datetime import date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.prediction_db import router
from app.db.session import get_db
from app.schemas.prediction import PredictionHistoryResponse, PredictionHistoryItem


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: object()
    with TestClient(app) as test_client:
        yield test_client


def test_get_prediction_history_api_success(client, mocker):
    service = mocker.patch(
        "app.api.prediction_db.get_prediction_history_service",
        return_value=PredictionHistoryResponse(
            ticker="SPY",
            count=1,
            results=[
                PredictionHistoryItem(
                    prediction_id=1,
                    ticker="SPY",
                    as_of_date=date(2026, 7, 17),
                    predicted_class=3,
                    predicted_regime="Trending-Up",
                    confidence=0.82,
                    probabilities={
                        "Trending-Up": 0.82,
                        "Transition-Up": 0.10,
                        "Transition-Down": 0.05,
                        "Trending-Down": 0.03,
                    },
                    model_version="v2",
                    created_at=datetime(2026, 7, 17, 21, 30),
                )
            ],
        ),
    )

    response = client.get(
        "/api/v1/predictions/history",
        params={"ticker": "SPY", "limit": 20},
    )

    assert response.status_code == 200
    service.assert_called_once_with(db=mocker.ANY, ticker="SPY", limit=20)

    data = response.json()

    assert data["ticker"] == "SPY"
    assert data["count"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["predicted_regime"] == "Trending-Up"


def test_get_prediction_history_api_empty_result(client, mocker):
    mocker.patch(
        "app.api.prediction_db.get_prediction_history_service",
        return_value=PredictionHistoryResponse(
            ticker="SPY",
            count=0,
            results=[],
        ),
    )

    response = client.get(
        "/api/v1/predictions/history",
        params={"ticker": "SPY"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ticker"] == "SPY"
    assert data["count"] == 0
    assert data["results"] == []


def test_get_prediction_history_api_limit_validation(client):
    response = client.get(
        "/api/v1/predictions/history",
        params={"ticker": "SPY", "limit": 0},
    )

    assert response.status_code == 422


def test_get_prediction_history_api_calls_real_service(client, mocker):
    repository = mocker.patch(
        "app.services.prediction_service.get_prediction_history_by_ticker",
        return_value=[],
    )

    response = client.get(
        "/api/v1/predictions/history",
        params={"ticker": "spy", "limit": 10},
    )

    assert response.status_code == 200
    assert response.json() == {"ticker": "SPY", "count": 0, "results": []}
    repository.assert_called_once_with(db=mocker.ANY, ticker="SPY", limit=10)
