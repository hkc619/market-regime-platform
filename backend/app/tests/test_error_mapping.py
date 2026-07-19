from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handler import app_error_handler
from app.core.exceptions import (
    AppError,
    TickerNotFoundError,
    InsufficientRawDataError,
    ExternalDataFetchError,
    InvalidExternalDataError,
)


def create_test_app() -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(AppError, app_error_handler)

    @app.get("/raise/ticker-not-found")
    def raise_ticker_not_found():
        raise TickerNotFoundError("Ticker INVALID was not found.")

    @app.get("/raise/insufficient-raw-data")
    def raise_insufficient_raw_data():
        raise InsufficientRawDataError("SPY requires at least 312 raw rows.")

    @app.get("/raise/external-fetch-error")
    def raise_external_fetch_error():
        raise ExternalDataFetchError("Failed to fetch data from yfinance.")

    @app.get("/raise/invalid-external-data")
    def raise_invalid_external_data():
        raise InvalidExternalDataError("External data is missing required columns.")

    return app


def test_ticker_not_found_maps_to_404():
    client = TestClient(create_test_app())

    response = client.get("/raise/ticker-not-found")

    assert response.status_code == 404
    assert response.json() == {
        "status": "error",
        "error_code": "TICKER_NOT_FOUND",
        "message": "Ticker INVALID was not found.",
    }


def test_insufficient_raw_data_maps_to_422():
    client = TestClient(create_test_app())

    response = client.get("/raise/insufficient-raw-data")

    assert response.status_code == 422
    assert response.json()["error_code"] == "INSUFFICIENT_RAW_DATA"


def test_external_fetch_error_maps_to_502():
    client = TestClient(create_test_app())

    response = client.get("/raise/external-fetch-error")

    assert response.status_code == 502
    assert response.json()["error_code"] == "EXTERNAL_DATA_FETCH_ERROR"


def test_invalid_external_data_maps_to_502():
    client = TestClient(create_test_app())

    response = client.get("/raise/invalid-external-data")

    assert response.status_code == 502
    assert response.json()["error_code"] == "INVALID_EXTERNAL_DATA"