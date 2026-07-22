# tests/test_market_data_validation.py

import pandas as pd
import pytest
from unittest.mock import MagicMock
from datetime import date, datetime

from app.core.exceptions import InvalidExternalDataError
from app.services.data_refresh_service import validate_market_data_df, refresh_market_data


@pytest.fixture
def valid_market_df():
    return pd.DataFrame(
        [
            {
                "date": "2026-07-15",
                "open": 620.0,
                "high": 625.0,
                "low": 618.0,
                "close": 623.0,
                "volume": 50000000,
            },
            {
                "date": "2026-07-16",
                "open": 623.0,
                "high": 628.0,
                "low": 621.0,
                "close": 627.0,
                "volume": 52000000,
            },
        ]
    )


def test_validate_market_data_success(valid_market_df):
    result = validate_market_data_df(valid_market_df)

    assert len(result) == 2
    assert "date" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_validate_market_data_missing_columns_raises_error(valid_market_df):
    df = valid_market_df.drop(columns=["close"])

    with pytest.raises(InvalidExternalDataError) as exc_info:
        validate_market_data_df(df)

    assert "missing required columns" in exc_info.value.message
    assert "close" in exc_info.value.message


def test_validate_market_data_invalid_dates_raises_error(valid_market_df):
    df = valid_market_df.copy()
    df.loc[0, "date"] = "not-a-date"

    with pytest.raises(InvalidExternalDataError) as exc_info:
        validate_market_data_df(df)

    assert "invalid date" in exc_info.value.message


def test_validate_market_data_non_positive_price_raises_error(valid_market_df):
    df = valid_market_df.copy()
    df.loc[0, "close"] = 0

    with pytest.raises(InvalidExternalDataError) as exc_info:
        validate_market_data_df(df)

    assert "non-positive" in exc_info.value.message
    assert "close" in exc_info.value.message


def test_validate_market_data_negative_volume_raises_error(valid_market_df):
    df = valid_market_df.copy()
    df.loc[0, "volume"] = -100

    with pytest.raises(InvalidExternalDataError) as exc_info:
        validate_market_data_df(df)

    assert "negative volume" in exc_info.value.message


def test_validate_market_data_empty_df_returns_empty_df():
    df = pd.DataFrame()

    result = validate_market_data_df(df)

    assert result.empty

class FakeMarketProviderSuccess:
    def fetch_market_data(self, ticker: str, *args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "2026-07-15",
                    "open": 620.0,
                    "high": 625.0,
                    "low": 618.0,
                    "close": 623.0,
                    "volume": 50000000,
                },
                {
                    "date": "2026-07-16",
                    "open": 623.0,
                    "high": 628.0,
                    "low": 621.0,
                    "close": 627.0,
                    "volume": 52000000,
                },
            ]
        )


class FakeMarketProviderEmpty:
    def fetch_market_data(self, ticker: str, *args, **kwargs):
        return pd.DataFrame()


class FakeMarketProviderInvalidData:
    def fetch_market_data(self, ticker: str, *args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "not-a-date",
                    "open": 620.0,
                    "high": 625.0,
                    "low": 618.0,
                    "close": 623.0,
                    "volume": 50000000,
                }
            ]
        )
    
def test_refresh_market_success_calls_upsert_and_commit(mocker):
    db = MagicMock()
    provider = FakeMarketProviderSuccess()

    mocker.patch(
        "app.services.data_refresh_service.get_latest_ohlcv",
        return_value=date(2026, 7, 14),
    )


    mock_upsert = mocker.patch(
        "app.services.data_refresh_service.upsert_market_prices",
        return_value=2,
    )

    result = refresh_market_data(
        db=db,
        ticker="SPY",
        provider=provider,
    )

    assert result["ticker"] == "SPY"
    assert result["status"] == "success"
    assert result["rows_fetched"] == 2
    assert result["rows_inserted_or_updated"] == 2

    mock_upsert.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_not_called()

def test_refresh_market_returns_no_new_data_when_provider_empty(mocker):
    db = MagicMock()
    provider = FakeMarketProviderEmpty()

    mocker.patch(
        "app.services.data_refresh_service.get_latest_ohlcv",
        return_value=date(2026, 7, 14),
    )

    mock_upsert = mocker.patch(
        "app.services.data_refresh_service.upsert_market_prices",
        return_value=0,
    )

    result = refresh_market_data(
        db=db,
        ticker="SPY",
        provider=provider,
    )

    assert result["ticker"] == "SPY"
    assert result["status"] == "no_new_data"
    assert result["rows_fetched"] == 0
    assert result["rows_inserted_or_updated"] == 0

    mock_upsert.assert_not_called()
    db.commit.assert_not_called()
    db.rollback.assert_not_called()

def test_refresh_market_rolls_back_on_invalid_external_data(mocker):
    db = MagicMock()
    provider = FakeMarketProviderInvalidData()

    mocker.patch(
        "app.services.data_refresh_service.get_latest_ohlcv",
        return_value=date(2026, 7, 14),
    )

    mock_upsert = mocker.patch(
        "app.services.data_refresh_service.upsert_market_prices",
        return_value=1,
    )

    with pytest.raises(InvalidExternalDataError):
        refresh_market_data(
            db=db,
            ticker="SPY",
            provider=provider,
        )

    mock_upsert.assert_not_called()
    db.commit.assert_not_called()
    db.rollback.assert_called_once()

def test_refresh_market_rolls_back_when_upsert_fails(mocker):
    db = MagicMock()
    provider = FakeMarketProviderSuccess()

    mocker.patch(
        "app.services.data_refresh_service.get_latest_ohlcv",
        return_value=date(2026, 7, 14),
    )

    mocker.patch(
        "app.services.data_refresh_service.upsert_market_prices",
        side_effect=Exception("database write failed"),
    )

    with pytest.raises(Exception) as exc_info:
        refresh_market_data(
            db=db,
            ticker="SPY",
            provider=provider,
        )

    assert "database write failed" in str(exc_info.value)

    db.commit.assert_not_called()
    db.rollback.assert_called_once()