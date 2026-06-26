from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


RAW_FILE = Path("/Users/hkc619/Documents/PY/project/market-regime-platform/data/raw/marco_monthly.csv")
STAGING_TABLE = "_staging_macro_monthly"


COLUMN_MAPPING = {
    "date": "period",
    "cpi_yoy_level": "cpi_yoy_level",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_macro_monthly() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE)
    df = normalize_columns(df)

    # 把 date 改成 period
    if "date" not in df.columns:
        raise ValueError("macro_monthly must contain a Date column.")

    df["period"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp("M")
        .dt.date
    )

    # 欄位名稱標準化後，只保留 DB 需要的欄位
    available_cols = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    df = df[available_cols + ["period"]].copy()
    
    # date 已經轉成 period，所以不要重複留 date
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    numeric_cols = ["cpi_yoy_level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["period"])
    df = df.drop_duplicates(subset=["period"], keep="last")
    df = df.sort_values("period")

    df["source"] = RAW_FILE.name

    return df


def validate_macro_monthly(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("macro_monthly is empty.")

    if df["period"].isna().any():
        raise ValueError("macro_monthly contains missing period.")

    if df["period"].duplicated().any():
        duplicates = df.loc[df["period"].duplicated(keep=False), "period"]
        raise ValueError(f"macro_monthly contains duplicated periods:\n{duplicates}")


def upsert_macro_monthly(df: pd.DataFrame) -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    engine = create_engine(database_url)

    insert_cols = [
        "period",
        "cpi_yoy_level",
        "source"
    ]

    # 如果某些欄位目前沒有，就自動補成 NA
    for col in insert_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df_to_insert = df[insert_cols].copy()

    with engine.begin() as conn:
        df_to_insert.to_sql(
            STAGING_TABLE,
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        conn.execute(
            text(
                f"""
                INSERT INTO macro_monthly (
                    period,
                    cpi_yoy_level,
                    source
                )
                SELECT
                    period,
                    cpi_yoy_level,
                    source
                FROM {STAGING_TABLE}
                ON CONFLICT (period)
                DO UPDATE SET
                    cpi_yoy_level = EXCLUDED.cpi_yoy_level,
                    source = EXCLUDED.source,
                    updated_at = NOW();
                """
            )
        )

        conn.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE};"))


def main() -> None:
    df = load_macro_monthly()
    validate_macro_monthly(df)

    print(df.head())
    print(df.tail())
    print(df.isna().sum())

    upsert_macro_monthly(df)

    print(f"Upserted {len(df)} rows into macro_monthly.")


if __name__ == "__main__":
    main()