from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.providers.fred_provider import FredClient
from app.repositories.macro_refresh_repository import get_latest_macro_daily, upsert_macro_daily, get_latest_macro_monthly, upsert_macro_monthly
from app.repositories.data_update_log_repository import create_data_update_log

logger = get_logger(__name__)

DAILY_MACRO_SERIES = {
    "VIX": {
        "series_id": "VIXCLS",
        "frequency": "daily",
        "units": "lin",
    },
    "Yield10yr": {
        "series_id": "DGS10",
        "frequency": "daily",
        "units": "lin",
    },
    "Yield2yr": {
        "series_id": "DGS2",
        "frequency": "daily",
        "units": "lin",
    },
}

FRED_MACRO_M_SERIES = {
    "CPI_YOY": {
        "series_id": "CPIAUCSL",
        "frequency": "monthly",
        "units": "pc1",
    },
}

def build_daily_macro_wide_df(long_df: pd.DataFrame) -> pd.DataFrame:
    wide_df = (
        long_df
        .pivot_table(
            index="date",
            columns="feature_name",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .rename(
            columns={
                "VIX": "vix",
                "Yield10yr": "yield_10yr",
                "Yield2yr": "yield_2yr",
            }
        )
    )

    wide_df["source"] = "FRED"

    wide_df = wide_df[["date", "vix", "yield_10yr", "yield_2yr", "source"]]

    wide_df = wide_df.sort_values("date").reset_index(drop=True)

    return wide_df

def normalize_monthly_date_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = (df["date"] + pd.offsets.MonthEnd(0)).dt.date

    return df


class MacroDataService:
    def __init__(self, fred_client: FredClient):
        self.fred_client = fred_client

    def get_latest_daily_date(self, db: Session) -> date | None:
        result = get_latest_macro_daily(db)

        return result
    
    def get_latest_monthly_date(self, db: Session) -> date | None:
        result = get_latest_macro_monthly(db)

        return result

    def refresh_daily_macro(
            self, 
            db: Session, 
            request_id: str | None = None,
        ) -> dict:

        latest_before = self.get_latest_daily_date(db)


        if latest_before:
            # 多抓 1 天是為了處理假日、缺值、資料修正
            observation_start = latest_before - timedelta(days=1)
        else:
            observation_start = date(1990, 1, 1)

        end_date = date.today()

        all_frames: list[pd.DataFrame] = []

        for feature_name, config in DAILY_MACRO_SERIES.items():
            df = self.fred_client.fetch_series(
                series_id=config["series_id"],
                observation_start=observation_start,
                units=config["units"]
            )

            if df.empty:
                continue

            df["feature_name"] = feature_name
            df["series_id"] = config["series_id"]
            df["frequency"] = config["frequency"]
            df["units"] = config["units"]
            df["source"] = "FRED"

            all_frames.append(df)

        if not all_frames:
            return {
                "status": "no_data",
                "message": "No daily macro data fetched from FRED.",
                "observation_start": observation_start,
                "rows_fetched": 0,
                "rows_upserted": 0,
            }
        
        long_df = pd.concat(all_frames, ignore_index=True)

        wide_df = build_daily_macro_wide_df(long_df)

        affected_rows = upsert_macro_daily(db, wide_df)

        create_data_update_log(
            db=db,
            request_id=request_id,
            data_type="macro_daily",
            source="FRED",
            ticker="macro_daily",
            start_date=latest_before,
            end_date=end_date,
            status="success",
            rows_fetched=len(wide_df),
            rows_inserted_or_updated=affected_rows,
            error_message=None,
        )

        latest_after = self.get_latest_daily_date(db)

        logger.info(
        "Macro daily refresh completed | request_id=%s | rows=%s | latest_before=%s | latest_after=%s",
        request_id,
        affected_rows,
        latest_before,
        latest_after,
        )

        return {
            "type_of_macro": "daily",
            "latest_before": latest_before,
            "latest_after": latest_after,
            "rows_fetched": len(wide_df),
            "rows_inserted_or_updated": affected_rows,
            "status": "success",
            "message": "Macro daily refreshed successfully.",
        }
    
    def refresh_monthly_macro(
            self, 
            db: Session, 
            request_id: str | None = None,
        ) -> dict:
        latest_before = self.get_latest_monthly_date(db)

        end_date = date.today()

        if latest_before:
            # 重點：先轉回該月月初，再往前抓幾個月
            # 不要直接用 2025-12-31 往前，否則可能 miss 掉 2025-10-01 這種 FRED monthly observation
            observation_start = latest_before.replace(day=1) - relativedelta(months=3)
        else:
            observation_start = date(1990, 1, 1)

        all_frames: list[pd.DataFrame] = []

        for feature_name, config in FRED_MACRO_M_SERIES.items():
            df = self.fred_client.fetch_series(
                series_id=config["series_id"],
                observation_start=observation_start,
                units=config["units"],
            )

            if df.empty:
                continue

            df = normalize_monthly_date_to_month_end(df)

            df["feature_name"] = feature_name
            df["series_id"] = config["series_id"]
            df["frequency"] = config["frequency"]
            df["units"] = config["units"]
            df["source"] = "FRED"

            all_frames.append(df)

        if not all_frames:
            return pd.DataFrame()

        monthly_df = pd.concat(all_frames, ignore_index=True)
        monthly_df = df.dropna(subset=['value'])
        monthly_df = monthly_df.rename(columns={"value": "cpi_yoy_level"})
        
        monthly_df = monthly_df[["date", "cpi_yoy_level", "source"]]

        # print(monthly_df.head())
        # print("rows:", len(monthly_df))
        # print("date range:", monthly_df["date"].min(), "to", monthly_df["date"].max())

        affected_rows = upsert_macro_monthly(db, monthly_df)

        create_data_update_log(
            db=db,
            request_id=request_id,
            data_type="macro_monthly",
            source="FRED",
            ticker="macro_monthly",
            start_date=latest_before,
            end_date=end_date,
            status="success",
            rows_fetched=len(monthly_df),
            rows_inserted_or_updated=affected_rows,
            error_message=None,
        )

        latest_after = self.get_latest_daily_date(db)

        logger.info(
        "Macro monthly refresh completed | request_id=%s | rows=%s | latest_before=%s | latest_after=%s",
        request_id,
        affected_rows,
        latest_before,
        latest_after,
        )

        return {
            "type_of_macro": "monthly",
            "latest_before": latest_before,
            "latest_after": latest_after,
            "rows_fetched": len(monthly_df),
            "rows_inserted_or_updated": affected_rows,
            "status": "success",
            "message": "Macro monthly refreshed successfully.",
        }

   

# if __name__ == "__main__":
#     try:
        
#         ins = MacroDataService(FredClient())
#         ins.refresh_monthly_macro()

#     finally:
#         pass
