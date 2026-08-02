from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

import app.services.backtest_data_service as sut


# ============================================================
# get_start_end_rows
# ============================================================

def test_get_start_end_rows_normalizes_ticker_and_reverses_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    repository_rows = [
        {"date": date(2025, 1, 3)},
        {"date": date(2025, 1, 2)},
    ]

    repository_mock = MagicMock(
        return_value=repository_rows
    )

    monkeypatch.setattr(
        sut,
        "get_rows_between_start_end",
        repository_mock,
    )

    result = sut.get_start_end_rows(
        ticker=" spy ",
        db=db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    repository_mock.assert_called_once_with(
        db=db,
        ticker="SPY",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    assert result == [
        {"date": date(2025, 1, 2)},
        {"date": date(2025, 1, 3)},
    ]


def test_get_start_end_rows_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    monkeypatch.setattr(
        sut,
        "get_rows_between_start_end",
        MagicMock(return_value=[]),
    )

    result = sut.get_start_end_rows(
        ticker="SPY",
        db=db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    assert result == []


def test_get_start_end_rows_propagates_repository_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    repository_mock = MagicMock(
        side_effect=RuntimeError("database failure")
    )

    monkeypatch.setattr(
        sut,
        "get_rows_between_start_end",
        repository_mock,
    )

    with pytest.raises(
        RuntimeError,
        match="database failure",
    ):
        sut.get_start_end_rows(
            ticker="SPY",
            db=db,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )


# ============================================================
# get_range_ticker_prices
# ============================================================

def test_get_range_ticker_prices_uses_period_count_plus_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    start_to_end_rows = [
        {"date": date(2025, 1, 2)},
        {"date": date(2025, 1, 3)},
        {"date": date(2025, 1, 6)},
    ]

    full_window_rows = [
        {"date": date(2025, 1, 6)},
        {"date": date(2025, 1, 3)},
        {"date": date(2025, 1, 2)},
        {"date": date(2024, 12, 31)},
    ]

    start_end_mock = MagicMock(
        return_value=start_to_end_rows
    )

    ticker_window_mock = MagicMock(
        return_value=full_window_rows
    )

    monkeypatch.setattr(
        sut,
        "get_start_end_rows",
        start_end_mock,
    )

    monkeypatch.setattr(
        sut,
        "get_ticker_window_range",
        ticker_window_mock,
    )

    result = sut.get_range_ticker_prices(
        db=db,
        ticker="spy",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        lookback=312,
    )

    start_end_mock.assert_called_once_with(
        ticker="spy",
        db=db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    ticker_window_mock.assert_called_once_with(
        db=db,
        ticker="spy",
        end_date=date(2025, 1, 31),
        range=315,
    )

    assert result == [
        {"date": date(2024, 12, 31)},
        {"date": date(2025, 1, 2)},
        {"date": date(2025, 1, 3)},
        {"date": date(2025, 1, 6)},
    ]


def test_get_range_ticker_prices_uses_only_lookback_when_period_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    start_end_mock = MagicMock(
        return_value=[]
    )

    ticker_window_mock = MagicMock(
        return_value=[
            {"date": date(2024, 12, 31)},
            {"date": date(2024, 12, 30)},
        ]
    )

    monkeypatch.setattr(
        sut,
        "get_start_end_rows",
        start_end_mock,
    )

    monkeypatch.setattr(
        sut,
        "get_ticker_window_range",
        ticker_window_mock,
    )

    result = sut.get_range_ticker_prices(
        db=db,
        ticker="SPY",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        lookback=312,
    )

    ticker_window_mock.assert_called_once_with(
        db=db,
        ticker="SPY",
        end_date=date(2025, 1, 31),
        range=312,
    )

    assert result == [
        {"date": date(2024, 12, 30)},
        {"date": date(2024, 12, 31)},
    ]


def test_get_range_ticker_prices_supports_custom_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    monkeypatch.setattr(
        sut,
        "get_start_end_rows",
        MagicMock(
            return_value=[
                {"date": date(2025, 1, 2)},
                {"date": date(2025, 1, 3)},
            ]
        ),
    )

    ticker_window_mock = MagicMock(
        return_value=[]
    )

    monkeypatch.setattr(
        sut,
        "get_ticker_window_range",
        ticker_window_mock,
    )

    sut.get_range_ticker_prices(
        db=db,
        ticker="SPY",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        lookback=100,
    )

    ticker_window_mock.assert_called_once_with(
        db=db,
        ticker="SPY",
        end_date=date(2025, 1, 31),
        range=102,
    )


def test_get_range_ticker_prices_reverses_repository_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    monkeypatch.setattr(
        sut,
        "get_start_end_rows",
        MagicMock(
            return_value=[
                {"date": date(2025, 1, 2)}
            ]
        ),
    )

    monkeypatch.setattr(
        sut,
        "get_ticker_window_range",
        MagicMock(
            return_value=[
                {"date": date(2025, 1, 3)},
                {"date": date(2025, 1, 2)},
                {"date": date(2024, 12, 31)},
            ]
        ),
    )

    result = sut.get_range_ticker_prices(
        db=db,
        ticker="SPY",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    assert result == [
        {"date": date(2024, 12, 31)},
        {"date": date(2025, 1, 2)},
        {"date": date(2025, 1, 3)},
    ]


def test_get_range_ticker_prices_propagates_window_repository_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = MagicMock()

    monkeypatch.setattr(
        sut,
        "get_start_end_rows",
        MagicMock(
            return_value=[
                {"date": date(2025, 1, 2)}
            ]
        ),
    )

    monkeypatch.setattr(
        sut,
        "get_ticker_window_range",
        MagicMock(
            side_effect=RuntimeError(
                "failed to load ticker window"
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="failed to load ticker window",
    ):
        sut.get_range_ticker_prices(
            db=db,
            ticker="SPY",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )
