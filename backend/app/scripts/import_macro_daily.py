from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


RAW_FILE = Path("/Users/hkc619/Documents/PY/project/market-regime-platform/data/raw/marco_daily.csv")
STAGING_TABLE = "_staging_macro_daily"


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


def load_macro_daily() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE)
    df = normalize_columns(df)

    # transform column name into：
    # Date      -> date
    # Vix Index -> vix_index
    # 10 Yr     -> 10_yr
    # 2 Yr      -> 2_yr

    column_mapping = {
        "date": "date",
        "vix_index": "vix",
        "10_yr": "yield_10yr",
        "2_yr": "yield_2yr",
    }

    missing = set(column_mapping.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in macro_daily file: {missing}")

    df = df[list(column_mapping.keys())].copy()
    df = df.rename(columns=column_mapping)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    numeric_cols = ["vix", "yield_10yr", "yield_2yr"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 移除完全沒有 date 的 row
    df = df.dropna(subset=["date"])

    # 如果同一天有重複資料，保留最後一筆
    df = df.drop_duplicates(subset=["date"], keep="last")

    # 日期排序
    df = df.sort_values("date").reset_index(drop=True)

    df["source"] = RAW_FILE.name

    return df

def validate_macro_daily(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("macro_daily is empty.")

    if df["date"].isna().any():
        raise ValueError("macro_daily contains missing dates.")

    if df["date"].duplicated().any():
        duplicates = df.loc[df["date"].duplicated(keep=False), "date"]
        raise ValueError(f"macro_daily contains duplicated dates:\n{duplicates}")

    value_cols = ["vix", "yield_10yr", "yield_2yr"]

    all_missing = df[value_cols].isna().all(axis=1)
    if all_missing.any():
        bad_rows = df.loc[all_missing, ["date"] + value_cols]
        raise ValueError(f"macro_daily has rows with all macro values missing:\n{bad_rows}")


def upsert_macro_daily(df: pd.DataFrame) -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    engine = create_engine(database_url)

    insert_cols = ["date", "vix", "yield_10yr", "yield_2yr", "source"]
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
                INSERT INTO macro_daily (
                    date,
                    vix,
                    yield_10yr,
                    yield_2yr,
                    source
                )
                SELECT
                    date,
                    vix,
                    yield_10yr,
                    yield_2yr,
                    source
                FROM {STAGING_TABLE}
                ON CONFLICT (date)
                DO UPDATE SET
                    vix = EXCLUDED.vix,
                    yield_10yr = EXCLUDED.yield_10yr,
                    yield_2yr = EXCLUDED.yield_2yr,
                    source = EXCLUDED.source,
                    updated_at = NOW();
                """
            )
        )

        conn.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE};"))


def main() -> None:
    df = load_macro_daily()
    validate_macro_daily(df)

    print(df.head())
    print(df.tail())
    print(df.isna().sum())

    upsert_macro_daily(df)

    print(f"Upserted {len(df)} rows into macro_daily.")


if __name__ == "__main__":
    main()