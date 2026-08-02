from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import app.services.backtest_service as sut
from app.core.exceptions import (
    InsufficientRawDataError,
    ModelInferenceError,
    NoSuccessfulBacktestPredictionsError,
    InvalidPredictionError,
)
from app.schemas.backtest import BacktestPredictionItem


# ============================================================
# Test data helpers
# ============================================================

STATE_NAMES = {
    0: "Trending-Down",
    1: "Transition-Down",
    2: "Transition-Up",
    3: "Trending-Up",
}


def make_weekday_dates(start: date, count: int) -> list[date]:
    """Return deterministic weekday-only dates."""
    dates: list[date] = []
    current = start

    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    return dates


def make_price_rows(
    count: int,
    *,
    ticker: str = "SPY",
    start: date = date(2024, 1, 2),
) -> list[dict[str, Any]]:
    dates = make_weekday_dates(start, count)

    return [
        {
            "ticker": ticker,
            "date": row_date,
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "adjusted_close": 100.5 + index,
            "volume": 1_000_000 + index,
        }
        for index, row_date in enumerate(dates)
    ]


def make_prediction(
    as_of_date: date,
    *,
    predicted_class: int = 3,
    confidence: float = 0.8,
) -> dict[str, Any]:
    predicted_regime = STATE_NAMES[predicted_class]
    remaining = 1.0 - confidence
    other_probability = remaining / 3

    probabilities = {
        regime: (
            confidence
            if regime == predicted_regime
            else other_probability
        )
        for regime in STATE_NAMES.values()
    }

    return {
        "ticker": "SPY",
        "as_of_date": as_of_date,
        "predicted_class": predicted_class,
        "predicted_regime": predicted_regime,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def make_prediction_item(
    as_of_date: date,
    *,
    predicted_class: int = 3,
    confidence: float = 0.8,
) -> BacktestPredictionItem:
    result = make_prediction(
        as_of_date,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    return BacktestPredictionItem(
        as_of_date=result["as_of_date"],
        predicted_class=result["predicted_class"],
        predicted_regime=result["predicted_regime"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture
def db() -> MagicMock:
    return MagicMock()


@pytest.fixture
def model_state() -> SimpleNamespace:
    return SimpleNamespace(
        model=object(),
        metadata={
            "model_config": {
                "scaler_path": "fake-scaler.pkl",
            }
        },
        device="cpu",
        model_version="v2",
    )


@pytest.fixture
def backtest_rows() -> list[dict[str, Any]]:
    # Three complete candidate windows:
    # first candidate is at position RAW_LOOKBACK_DAYS - 1.
    total = sut.RAW_LOOKBACK_DAYS + 2
    return make_price_rows(total)


@pytest.fixture
def candidate_date_rows(
    backtest_rows: list[dict[str, Any]],
) -> list[dict[str, date]]:
    first_candidate_position = sut.RAW_LOOKBACK_DAYS - 1

    return [
        {"date": row["date"]}
        for row in backtest_rows[first_candidate_position:]
    ]


def patch_data_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ticker_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, date]],
) -> None:
    support_rows = [dict(row) for row in ticker_rows]

    monkeypatch.setattr(
        sut,
        "get_range_ticker_prices",
        lambda **kwargs: ticker_rows,
    )

    monkeypatch.setattr(
        sut,
        "get_latest_support_window",
        lambda **kwargs: support_rows,
    )

    monkeypatch.setattr(
        sut,
        "get_macro_daily_window",
        lambda **kwargs: [],
    )

    monkeypatch.setattr(
        sut,
        "get_macro_monthly_window",
        lambda **kwargs: [],
    )

    monkeypatch.setattr(
        sut,
        "get_rows_between_start_end",
        lambda **kwargs: candidate_rows,
    )


def run_backtest(
    *,
    db: MagicMock,
    model_state: SimpleNamespace,
    start_date: date,
    end_date: date,
):
    return sut.backtest_for_range(
        model_state=model_state,
        db=db,
        ticker="SPY",
        sup0="QQQ",
        sup1="TLT",
        start_date=start_date,
        end_date=end_date,
        request_id="test-request-id",
    )


# ============================================================
# predict_for_date tests
# ============================================================

def test_predict_for_date_success(
    monkeypatch: pytest.MonkeyPatch,
    model_state: SimpleNamespace,
) -> None:
    raw = SimpleNamespace(
        ticker_close=[1.0] * sut.RAW_LOOKBACK_DAYS,
    )

    model_input = SimpleNamespace(
        latest_60_feat=SimpleNamespace(shape=(60, 73)),
        end_date=date(2025, 1, 31),
    )

    monkeypatch.setattr(
        "app.services.inference_db_service.prepare_backtest_inference_input",
        lambda **kwargs: raw,
    )

    monkeypatch.setattr(
        "app.services.features_input_service.build_latest_model_input",
        lambda raw_input: model_input,
    )

    monkeypatch.setattr(
        "app.services.inference.predict_proba",
        lambda *args, **kwargs: {
            "predicted_class": 3,
            "predicted_regime": "Trending-Up",
            "confidence": 0.8,
            "probabilities": {
                "Trending-Down": 0.05,
                "Transition-Down": 0.05,
                "Transition-Up": 0.10,
                "Trending-Up": 0.80,
            },
        },
    )

    result = sut.predict_for_date(
        ticker="spy",
        model_state=model_state,
        request_id="request-1",
        ticker_rows=[],
        sup0_rows=[],
        sup1_rows=[],
        macro_daily_rows=[],
        macro_monthly_rows=[],
    )

    assert result == {
        "ticker": "SPY",
        "as_of_date": date(2025, 1, 31),
        "predicted_class": 3,
        "predicted_regime": "Trending-Up",
        "confidence": 0.8,
        "probabilities": {
            "Trending-Down": 0.05,
            "Transition-Down": 0.05,
            "Transition-Up": 0.10,
            "Trending-Up": 0.80,
        },
    }


def test_predict_for_date_raises_insufficient_raw_data(
    monkeypatch: pytest.MonkeyPatch,
    model_state: SimpleNamespace,
) -> None:
    raw = SimpleNamespace(
        ticker_close=[1.0] * (sut.RAW_LOOKBACK_DAYS - 1),
    )

    monkeypatch.setattr(
        "app.services.inference_db_service.prepare_backtest_inference_input",
        lambda **kwargs: raw,
    )

    with pytest.raises(InsufficientRawDataError):
        sut.predict_for_date(
            ticker="SPY",
            model_state=model_state,
            request_id="request-1",
            ticker_rows=[],
            sup0_rows=[],
            sup1_rows=[],
            macro_daily_rows=[],
            macro_monthly_rows=[],
        )


def test_predict_for_date_wraps_model_error(
    monkeypatch: pytest.MonkeyPatch,
    model_state: SimpleNamespace,
) -> None:
    raw = SimpleNamespace(
        ticker_close=[1.0] * sut.RAW_LOOKBACK_DAYS,
    )

    model_input = SimpleNamespace(
        latest_60_feat=SimpleNamespace(shape=(60, 73)),
        end_date=date(2025, 1, 31),
    )

    monkeypatch.setattr(
        "app.services.inference_db_service.prepare_backtest_inference_input",
        lambda **kwargs: raw,
    )

    monkeypatch.setattr(
        "app.services.features_input_service.build_latest_model_input",
        lambda raw_input: model_input,
    )

    def raise_model_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("fake model failure")

    monkeypatch.setattr(
        "app.services.inference.predict_proba",
        raise_model_error,
    )

    with pytest.raises(ModelInferenceError):
        sut.predict_for_date(
            ticker="SPY",
            model_state=model_state,
            request_id="request-1",
            ticker_rows=[],
            sup0_rows=[],
            sup1_rows=[],
            macro_daily_rows=[],
            macro_monthly_rows=[],
        )


# ============================================================
# build_backtest_summary tests
# ============================================================

def test_build_backtest_summary_success() -> None:
    predictions = [
        make_prediction_item(
            date(2025, 1, 2),
            predicted_class=3,
            confidence=0.8,
        ),
        make_prediction_item(
            date(2025, 1, 3),
            predicted_class=2,
            confidence=0.7,
        ),
        make_prediction_item(
            date(2025, 1, 6),
            predicted_class=2,
            confidence=0.4,
        ),
    ]

    candidate_dates = [
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
    ]

    result = sut.build_backtest_summary(
        ticker="SPY",
        requested_start_date=date(2025, 1, 1),
        requested_end_date=date(2025, 1, 10),
        candidate_dates=candidate_dates,
        predictions=predictions,
        skip_reasons={
            "insufficient_market_rows": 0,
            "candidate_date_missing": 0,
            "prediction_failed": 1,
        },
    )

    assert result.ticker == "SPY"
    assert result.num_candidate_dates == 4
    assert result.num_predictions == 3
    assert result.skipped_dates == 1
    assert result.coverage_rate == pytest.approx(0.75)

    assert result.actual_prediction_start_date == date(2025, 1, 2)
    assert result.actual_prediction_end_date == date(2025, 1, 6)

    assert result.regime_distribution == {
        "Trending-Down": 0,
        "Transition-Down": 0,
        "Transition-Up": 2,
        "Trending-Up": 1,
    }

    assert sum(
        result.regime_distribution.values()
    ) == result.num_predictions

    assert sum(
        result.regime_distribution_pct.values()
    ) == pytest.approx(1.0, abs=1e-4)

    assert result.confidence_summary.average == pytest.approx(
        0.6333,
        abs=1e-4,
    )
    assert result.confidence_summary.minimum == pytest.approx(0.4)
    assert result.confidence_summary.maximum == pytest.approx(0.8)
    assert result.confidence_summary.low_confidence_count == 1


def test_build_backtest_summary_rejects_empty_predictions() -> None:
    with pytest.raises(NoSuccessfulBacktestPredictionsError):
        sut.build_backtest_summary(
            ticker="SPY",
            requested_start_date=date(2025, 1, 1),
            requested_end_date=date(2025, 1, 10),
            candidate_dates=[date(2025, 1, 2)],
            predictions=[],
            skip_reasons={
                "insufficient_market_rows": 1,
                "candidate_date_missing": 0,
                "prediction_failed": 0,
            },
        )


def test_build_backtest_summary_rejects_unknown_class() -> None:
    invalid_prediction = BacktestPredictionItem(
        as_of_date=date(2025, 1, 2),
        predicted_class=99,
        predicted_regime="Unknown",
        confidence=0.8,
        probabilities={"Unknown": 0.8},
    )

    with pytest.raises(InvalidPredictionError):
        sut.build_backtest_summary(
            ticker="SPY",
            requested_start_date=date(2025, 1, 1),
            requested_end_date=date(2025, 1, 10),
            candidate_dates=[date(2025, 1, 2)],
            predictions=[invalid_prediction],
            skip_reasons={
                "insufficient_market_rows": 0,
                "candidate_date_missing": 0,
                "prediction_failed": 0,
            },
        )


def test_build_backtest_summary_rejects_class_regime_mismatch() -> None:
    invalid_prediction = BacktestPredictionItem(
        as_of_date=date(2025, 1, 2),
        predicted_class=3,
        predicted_regime="Transition-Up",
        confidence=0.8,
        probabilities={
            "Trending-Down": 0.05,
            "Transition-Down": 0.05,
            "Transition-Up": 0.10,
            "Trending-Up": 0.80,
        },
    )

    with pytest.raises(InvalidPredictionError):
        sut.build_backtest_summary(
            ticker="SPY",
            requested_start_date=date(2025, 1, 1),
            requested_end_date=date(2025, 1, 10),
            candidate_dates=[date(2025, 1, 2)],
            predictions=[invalid_prediction],
            skip_reasons={
                "insufficient_market_rows": 0,
                "candidate_date_missing": 0,
                "prediction_failed": 0,
            },
        )


# ============================================================
# backtest_for_range tests
# ============================================================

def test_backtest_for_range_success(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    def fake_predict_for_date(
        ticker: str,
        *,
        ticker_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return make_prediction(
            ticker_rows[-1]["date"],
            predicted_class=3,
            confidence=0.8,
        )

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        fake_predict_for_date,
    )

    result = run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    assert result.num_candidate_dates == len(candidate_date_rows)
    assert result.num_predictions == len(candidate_date_rows)
    assert result.skipped_dates == 0
    assert result.coverage_rate == pytest.approx(1.0)
    assert len(result.predictions) == len(candidate_date_rows)


def test_backtest_uses_exact_raw_lookback_size(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    observed_window_sizes: list[int] = []

    def fake_predict_for_date(
        ticker: str,
        *,
        ticker_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed_window_sizes.append(len(ticker_rows))
        return make_prediction(ticker_rows[-1]["date"])

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        fake_predict_for_date,
    )

    run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    assert observed_window_sizes
    assert all(
        size == sut.RAW_LOOKBACK_DAYS
        for size in observed_window_sizes
    )


def test_backtest_windows_advance_one_row(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    captured_windows: list[list[date]] = []

    def fake_predict_for_date(
        ticker: str,
        *,
        ticker_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_windows.append(
            [row["date"] for row in ticker_rows]
        )
        return make_prediction(ticker_rows[-1]["date"])

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        fake_predict_for_date,
    )

    run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    assert len(captured_windows) >= 2

    for previous, current in zip(
        captured_windows,
        captured_windows[1:],
    ):
        assert previous[1:] == current[:-1]


def test_backtest_prediction_dates_match_candidate_dates(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    def fake_predict_for_date(
        ticker: str,
        *,
        ticker_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return make_prediction(ticker_rows[-1]["date"])

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        fake_predict_for_date,
    )

    result = run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    expected_dates = [
        row["date"]
        for row in candidate_date_rows
    ]

    actual_dates = [
        prediction.as_of_date
        for prediction in result.predictions
    ]

    assert actual_dates == expected_dates


def test_backtest_rejects_invalid_date_range(
    db: MagicMock,
    model_state: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError):
        run_backtest(
            db=db,
            model_state=model_state,
            start_date=date(2025, 2, 1),
            end_date=date(2025, 1, 1),
        )


def test_backtest_rejects_no_candidate_dates(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=[],
    )

    with pytest.raises(NoSuccessfulBacktestPredictionsError):
        run_backtest(
            db=db,
            model_state=model_state,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )


def test_backtest_counts_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    failed_date = candidate_date_rows[1]["date"]

    def fake_predict_for_date(
        ticker: str,
        *,
        ticker_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        as_of_date = ticker_rows[-1]["date"]

        if as_of_date == failed_date:
            raise ModelInferenceError("fake failure")

        return make_prediction(as_of_date)

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        fake_predict_for_date,
    )

    result = run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    assert result.num_candidate_dates == len(candidate_date_rows)
    assert result.num_predictions == len(candidate_date_rows) - 1
    assert result.skipped_dates == 1
    assert result.skip_reasons["prediction_failed"] == 1


def test_backtest_probability_invariants(
    monkeypatch: pytest.MonkeyPatch,
    db: MagicMock,
    model_state: SimpleNamespace,
    backtest_rows: list[dict[str, Any]],
    candidate_date_rows: list[dict[str, date]],
) -> None:
    patch_data_loaders(
        monkeypatch,
        ticker_rows=backtest_rows,
        candidate_rows=candidate_date_rows,
    )

    monkeypatch.setattr(
        sut,
        "predict_for_date",
        lambda ticker, *, ticker_rows, **kwargs: make_prediction(
            ticker_rows[-1]["date"],
            predicted_class=2,
            confidence=0.7,
        ),
    )

    result = run_backtest(
        db=db,
        model_state=model_state,
        start_date=candidate_date_rows[0]["date"],
        end_date=candidate_date_rows[-1]["date"],
    )

    expected_states = set(STATE_NAMES.values())

    for prediction in result.predictions:
        assert set(prediction.probabilities) == expected_states
        assert sum(
            prediction.probabilities.values()
        ) == pytest.approx(1.0, abs=1e-5)
        assert all(
            0.0 <= probability <= 1.0
            for probability in prediction.probabilities.values()
        )
        assert prediction.confidence == pytest.approx(
            max(prediction.probabilities.values())
        )
        assert (
            STATE_NAMES[prediction.predicted_class]
            == prediction.predicted_regime
        )
