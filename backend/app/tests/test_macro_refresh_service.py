from datetime import date
from unittest.mock import MagicMock
import pandas as pd

from app.services.macro_refresh_service import build_daily_macro_wide_df, normalize_monthly_date_to_month_end, MacroDataService

def test_build_daily_macro_wide_df():
    long_df = pd.DataFrame(
        [
            {
                "date": date(2025, 12, 15),
                "feature_name": "VIX",
                "value": 15.2,
            },
            {
                "date": date(2025, 12, 15),
                "feature_name": "Yield10yr",
                "value": 4.12,
            },
            {
                "date": date(2025, 12, 15),
                "feature_name": "Yield2yr",
                "value": 3.68,
            },
            {
                "date": date(2025, 12, 16),
                "feature_name": "VIX",
                "value": 16.1,
            },
            {
                "date": date(2025, 12, 16),
                "feature_name": "Yield10yr",
                "value": 4.15,
            },
            {
                "date": date(2025, 12, 16),
                "feature_name": "Yield2yr",
                "value": 3.70,
            },
        ]
    )

    result = build_daily_macro_wide_df(long_df)

    assert list(result.columns) == ["date", "vix", "yield_10yr", "yield_2yr", "source"]
    assert len(result) == 2

    first_row = result.iloc[0]

    assert first_row["date"] == date(2025, 12, 15)
    assert first_row["vix"] == 15.2
    assert first_row["yield_10yr"] == 4.12
    assert first_row["yield_2yr"] == 3.68
    assert first_row["source"] == "FRED"


def test_normalize_monthly_date_to_month_end():
    df = pd.DataFrame(
        [
            {"date": "2025-10-01", "value": 3.1},
            {"date": "2025-11-01", "value": 2.9},
            {"date": "2025-12-01", "value": 2.7},
        ]
    )

    result = normalize_monthly_date_to_month_end(df)

    assert result.loc[0, "date"] == date(2025, 10, 31)
    assert result.loc[1, "date"] == date(2025, 11, 30)
    assert result.loc[2, "date"] == date(2025, 12, 31)


class FakeFredClient:
    def fetch_series(
        self,
        series_id: str,
        observation_start,
        observation_end=None,
        units: str = "lin",
    ) -> pd.DataFrame:
        values_by_series = {
            "VIXCLS": [15.2],
            "DGS10": [4.12],
            "DGS2": [3.68],
        }

        return pd.DataFrame(
            {
                "date": [date(2025, 12, 15)],
                "value": values_by_series[series_id],
                "realtime_start": [date(2025, 12, 16)],
                "realtime_end": [date(2025, 12, 16)],
            }
        )


def test_refresh_daily_macro_returns_wide_dataframe():
    db = MagicMock()
    service = MacroDataService(fred_client=FakeFredClient())

    result = service.refresh_daily_macro(db)

    assert list(result.columns) == ["date", "vix", "yr10", "yr2", "source"]
    assert len(result) == 2

    assert result.loc[0, "vix"] == 15.2
    assert result.loc[0, "yr10"] == 4.12
    assert result.loc[0, "yr2"] == 3.68
    assert result.loc[0, "source"] == "FRED"

def test_refresh_daily_macro_calls_upsert_and_commit(mocker):
    db = MagicMock()

    service = MacroDataService(fred_client=FakeFredClient())

    mocker.patch.object(
        service,
        "get_latest_daily_macro_date",
        return_value=date(2025, 12, 18),
    )

    mock_upsert = mocker.patch.object(
        service,
        "upsert_daily_macro",
        return_value=1,
    )

    result = service.refresh_daily_macro(db)

    assert result["status"] == "success"
    assert result["rows_fetched"] == 1
    assert result["rows_upserted"] == 1

    mock_upsert.assert_called_once()
    db.commit.assert_called_once()