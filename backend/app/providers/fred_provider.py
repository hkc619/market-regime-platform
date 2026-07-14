import warnings
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import date
from typing import Any

from app.core.exceptions import ExternalDataFetchError, NoNewMarketDataError
from app.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

class FredClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not set")

    def fetch_series(
        self,
        series_id: str,
        observation_start: date | str,
        observation_end: date | str | None = None,
        units: str="lin",
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": str(observation_start),
            "sort_order": "asc",
            "units": units,
        }

        if observation_end:
            params["observation_end"] = str(observation_end)

        response = requests.get(FRED_BASE_URL, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()
        observations = data.get("observations", [])

        df = pd.DataFrame(observations)

        if df.empty:
            return pd.DataFrame(
                columns=["series_id", "date", "value", "realtime_start", "realtime_end"]
            )

        df["series_id"] = series_id
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["realtime_start"] = pd.to_datetime(df["realtime_start"]).dt.date
        df["realtime_end"] = pd.to_datetime(df["realtime_end"]).dt.date

        # FRED missing value is often "."
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")

        return df[["series_id", "date", "value", "realtime_start", "realtime_end"]]